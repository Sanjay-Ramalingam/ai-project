import cv2
import numpy as np
import easyocr
from rapidfuzz import fuzz, process

class HandwritingEvaluator:
    def __init__(self, use_gpu=False):
        """
        Initializes the AI Evaluator. 
        Note: The OCR reader is set to None (Lazy Loading) to speed up initial startup.
        """
        self.reader = None
        self.use_gpu = use_gpu
        self.reports = []

    def _init_ocr(self):
        """Internal method to load EasyOCR models only when processing starts."""
        if self.reader is None:
            print("Initializing AI OCR Engine (Loading Models)...")
            self.reader = easyocr.Reader(['en'], gpu=self.use_gpu)

    def segment_content(self, line_crops):
        """
        Detects question numbers in the left margin and groups lines into 
        distinct answer blocks. Returns a dictionary: { "Q1": "text...", "Q2": "text..." }
        """
        self._init_ocr()
        structured_data = {}
        current_q = "General" # Default bucket for text before any question number
        
        

        for line in line_crops:
            if line.shape[0] < 15: continue
            
            h, w = line.shape
            # Analyze only the leftmost 12% of the line to find the Question Number
            margin_box = line[:, :int(w * 0.12)]
            
            # Fast OCR on the margin only
            margin_results = self.reader.readtext(margin_box, detail=0)
            
            # Check if margin contains a digit (e.g., '1.', '2)', 'Q3')
            if margin_results and any(char.isdigit() for char in margin_results[0]):
                # Extract digits to create a clean ID (e.g., "1", "2")
                q_id = "".join(filter(str.isdigit, margin_results[0]))
                current_q = f"Q{q_id}"
                if current_q not in structured_data:
                    structured_data[current_q] = ""

            # Transcribe the full line
            # Optimization: 50% resize for the heavy recognition pass
            small_line = cv2.resize(line, (w//2, h//2))
            color_line = cv2.cvtColor(small_line, cv2.COLOR_GRAY2BGR)
            full_line_text = self.reader.readtext(color_line, detail=0, paragraph=True)
            
            if full_line_text:
                if current_q not in structured_data:
                    structured_data[current_q] = ""
                structured_data[current_q] += " " + " ".join(full_line_text)
                
        return structured_data

    def extract_key_from_pdf(self, processor, key_pdf_path):
        """
        Reads the Answer Key PDF and segments it by question number 
        to create a Master reference map.
        """
        print(f"--- Processing Master Answer Key: {key_pdf_path} ---")
        processor.load_pdf(key_pdf_path)
        master_key_map = {}
        
        

        for page in processor.pages:
            binary = processor.clean_page(page)
            lines = processor.extract_lines(binary)
            # Map this page's content by question
            page_map = self.segment_content(lines)
            master_key_map.update(page_map)
            
        return master_key_map

    def grade_paper(self, student_map, master_key_map):
        """
        Performs the final evaluation by comparing student questions 
        against corresponding master key questions.
        """
        results = {}
        
        for q_num, student_text in student_map.items():
            if q_num in master_key_map:
                key_text = master_key_map[q_num]
                
                # Automatically extract unique technical terms (len > 4) from the key
                key_words = list(set([w.strip(".,") for w in key_text.lower().split() if len(w) > 4]))
                # Assign 10 points per keyword found
                weights = {word: 10 for word in key_words}
                
                # Perform fuzzy matching to allow for handwriting/OCR variations
                score, matches = self.compare_to_key(student_text, weights)
                results[q_num] = {"score": score, "matches": matches}
            else:
                results[q_num] = {"score": 0.0, "status": "No matching question in Answer Key"}
                
        return results

    def compare_to_key(self, student_text, weight_map):
        """
        Fuzzy comparison between student's answer and expected keywords.
        """
        if not student_text or not weight_map: 
            return 0.0, []

        

        student_words = student_text.lower().split()
        earned = 0
        found = []

        for term, weight in weight_map.items():
            # Using WRatio for better matching on varying handwriting lengths
            match = process.extractOne(term, student_words, scorer=fuzz.WRatio)
            
            # 80% similarity threshold to account for OCR 'near misses'
            if match and match[1] >= 80:
                earned += weight
                found.append(f"{term} ({int(match[1])}%)")

        total = sum(weight_map.values())
        final_score = (earned / total * 100) if total > 0 else 0
        return round(final_score, 2), found

    def calculate_neatness(self, line_crops):
        """Evaluates physical presentation (height/margin consistency)."""
        if not line_crops: return 0
        
        # 1. Height Consistency
        heights = np.array([img.shape[0] for img in line_crops])
        h_sigma = np.std(heights)
        consistency = max(0, 100 - (h_sigma * 1.2))

        # 2. Left Margin Alignment
        margins = []
        for line in line_crops:
            coords = np.argwhere(line > 0)
            if coords.size > 0:
                margins.append(np.min(coords[:, 1]))
        
        m_sigma = np.std(margins) if margins else 20
        alignment = max(0, 100 - (m_sigma * 0.7))

        return round((consistency * 0.6) + (alignment * 0.4), 2)

    def detect_slant(self, line_crops):
        """Calculates handwriting slant angle using Hough Transform."""
        slants = []
        for line in line_crops:
            edges = cv2.Canny(line, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 40, minLineLength=30, maxLineGap=10)
            if lines is not None:
                angles = [np.degrees(np.arctan2(l[0][3]-l[0][1], l[0][2]-l[0][0])) for l in lines]
                valid_angles = [a for a in angles if abs(a) < 45]
                if valid_angles: slants.append(np.median(valid_angles))
        return round(np.mean(slants), 2) if slants else 0

    def estimate_content(self, line_crops):
        """Rough word count using morphological dilation and blob counting."""
        word_count = 0
        for line in line_crops:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 2))
            dilated = cv2.dilate(line, kernel, iterations=1)
            num_labels, _ = cv2.connectedComponents(dilated)
            word_count += max(0, num_labels - 1)
        return word_count