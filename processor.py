import cv2
import numpy as np
import os
from pdf2image import convert_from_path

class DocumentProcessor:
    def __init__(self):
        self.pages = []

    def load_pdf(self, pdf_path):
        """Loads PDF and converts pages to OpenCV BGR format."""
        try:
            # Using 300 DPI ensures OCR accuracy later
            pil_images = convert_from_path(pdf_path, dpi=300)
            self.pages = [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) for img in pil_images]
            print(f"Successfully loaded {len(self.pages)} pages.")
        except Exception as e:
            print(f"Error loading PDF: {e}")

    def clean_page(self, image):
        """Pre-processes image: Denoising and Adaptive Thresholding."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Bilateral filter removes paper texture while keeping pen strokes sharp
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Adaptive Gaussian Thresholding handles uneven lighting/shadows
        
        binary = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 41, 10
        )
        
        # Morphological opening to remove 'salt and pepper' noise
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return binary

    def extract_lines(self, binary_image):
        """Segments the page into individual line strips using horizontal projection."""
        # Summing pixels horizontally to find 'peaks' of text
        
        row_sums = np.sum(binary_image, axis=1)
        
        threshold = np.max(row_sums) * 0.02
        ink_rows = np.where(row_sums > threshold)[0]
        
        if len(ink_rows) == 0:
            return []

        # Grouping consecutive rows into fragments
        diffs = np.diff(ink_rows)
        breaks = np.where(diffs > 15)[0]
        
        raw_intervals = []
        start = ink_rows[0]
        for b in breaks:
            raw_intervals.append((start, ink_rows[b]))
            start = ink_rows[b+1]
        raw_intervals.append((start, ink_rows[-1]))

        # Smart Merge: Join fragments if they are vertically close (handles g, j, y descenders)
        merged_intervals = []
        if raw_intervals:
            curr_start, curr_end = raw_intervals[0]
            for i in range(1, len(raw_intervals)):
                next_start, next_end = raw_intervals[i]
                
                # If gap is < 40px, it's the same line
                if (next_start - curr_end) < 40:
                    curr_end = next_end
                else:
                    merged_intervals.append((curr_start, curr_end))
                    curr_start, curr_end = next_start, next_end
            merged_intervals.append((curr_start, curr_end))

        # Crop and pad for OCR readiness
        line_crops = []
        for s, e in merged_intervals:
            if (e - s) > 30:
                padding = 20
                y1 = max(0, s - padding)
                y2 = min(binary_image.shape[0], e + padding)
                line_crops.append(binary_image[y1:y2, :])
        
        return line_crops

    def save_data(self, page_num, lines, output_dir="output"):
        """Saves extracted line strips as individual images for records."""
        path = os.path.join(output_dir, f"page_{page_num}")
        if not os.path.exists(path):
            os.makedirs(path)
            
        for i, line in enumerate(lines):
            file_name = os.path.join(path, f"line_{i}.png")
            cv2.imwrite(file_name, line)