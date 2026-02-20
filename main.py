import cv2
import os
import json
import numpy as np
from processor import DocumentProcessor
from evaluator import HandwritingEvaluator

# --- CONFIGURATION ---
STUDENT_PDF = "student_script.pdf"
ANSWER_KEY_PDF = "answer_key.pdf"
FAST_MODE = False  # Disable to get full content grading (slow on CPU)
# ---------------------

def run_paper_evaluator(student_path, key_path):
    # 1. Initialize Components
    processor = DocumentProcessor()
    # Lazy loads EasyOCR only when transcription starts
    evaluator = HandwritingEvaluator(use_gpu=False) 
    
    print("="*60)
    print("          AUTOMATED ACADEMIC PAPER EVALUATOR")
    print("="*60)

    # --- PHASE 1: MASTER KEY LEARNING ---
    # The AI reads the Answer Key PDF to create a reference map { "Q1": "text", ... }
    if not os.path.exists(key_path):
        print(f"Error: Answer key not found at {key_path}")
        return

    # In FAST_MODE, skip intensive answer key OCR processing
    if FAST_MODE:
        print("FAST_MODE: Skipping answer key OCR extraction")
        master_key_map = {"Q1": "sample", "Q2": "sample", "Q3": "sample"}
    else:
        print("Loading Answer Key... (this may take several minutes)")
        master_key_map = evaluator.extract_key_from_pdf(processor, key_path)
    
    if not master_key_map:
        print("Error: Could not extract questions from Answer Key.")
        return

    # Try to load optional syllabus mapping (question -> module, bloom)
    syllabus_map = {}
    syllabus_path = os.path.join(os.getcwd(), "syllabus.json")
    if os.path.exists(syllabus_path):
        try:
            with open(syllabus_path, 'r', encoding='utf-8') as f:
                syllabus_map = json.load(f)
            print(f"Loaded syllabus mapping from {syllabus_path}")
        except Exception as e:
            print(f"Warning: failed to load syllabus.json: {e}")

    # --- PHASE 2: STUDENT SCRIPT EVALUATION ---
    print(f"\n--- Loading Student Script: {student_path} ---")
    processor.load_pdf(student_path)

    if not processor.pages:
        print("Error: No pages found in student PDF.")
        return

    all_page_results = []

    # Process page by page
    for i, raw_page in enumerate(processor.pages):
        print(f"\n[Page {i+1}] Segmenting and Grading...")
        
        # Step A: Image Cleanup
        binary = processor.clean_page(raw_page)
        lines = processor.extract_lines(binary)
        
        # Step B: Presentation Quality
        neatness = evaluator.calculate_neatness(lines)
        word_count = evaluator.estimate_content(lines)
        
        # Step C: Question-Wise Grading
        page_grading = {}
        if not FAST_MODE and lines:
            # Group lines into Question Blocks
            student_q_map = evaluator.segment_content(lines)
            
            # Compare specifically against the learned Master Key
            page_grading = evaluator.grade_paper(student_q_map, master_key_map)
        
        # Step D: Dashboard Display
        display = cv2.resize(raw_page, (600, 800))
        cv2.putText(display, f"Page {i+1} Neatness: {neatness}%", (40, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        try:
            cv2.imshow("Paper Evaluator - Analysis Feed", display)
        except cv2.error:
            # GUI not available - just log to console
            pass
        
        all_page_results.append({
            "page": i + 1,
            "neatness": neatness,
            "grading": page_grading,
            "words": word_count
        })

        # Press 'q' to stop early, any other key for next page (skip in headless mode)
        try:
            if cv2.waitKey(0) == ord('q'):
                break
        except:
            pass

    # --- PHASE 3: FINAL ACADEMIC SCORECARD ---
    generate_final_report(all_page_results, syllabus_map)
    try:
        cv2.destroyAllWindows()
    except:
        pass

def generate_final_report(results, syllabus_map=None):
    """Aggregates per-page data into a final student summary."""
    print("\n" + "="*60)
    print("                FINAL EVALUATION SUMMARY")
    print("="*60)
    
    total_neatness = 0
    total_questions_found = 0
    grand_score = 0
    
    # We consolidate results from all pages
    consolidated_grading = {}
    for p in results:
        total_neatness += p['neatness']
        consolidated_grading.update(p['grading'])

    print(f"{'QUESTION':<15} | {'SCORE':<10} | {'KEY CONCEPTS IDENTIFIED'}")
    print("-" * 60)
    
    for q, data in consolidated_grading.items():
        score = data.get('score', 0)
        matches = ", ".join(data.get('matches', []))
        print(f"{q:<15} | {score:>6}% | {matches}")
        grand_score += score
        total_questions_found += 1

    # Module-wise and Bloom's taxonomy aggregation
    if not syllabus_map:
        syllabus_map = {}

    module_sums = {}
    bloom_sums = {}
    for q, data in consolidated_grading.items():
        score = data.get('score', 0)
        meta = syllabus_map.get(q, {}) if isinstance(syllabus_map, dict) else {}
        module = meta.get('module', 'Unknown')
        bloom = meta.get('bloom', 'Unspecified')

        module_sums.setdefault(module, []).append(score)
        bloom_sums.setdefault(bloom, []).append(score)

    if module_sums:
        print("\nMODULE-WISE SUMMARY")
        print("{:<25} | {:>6} | {:>6}".format('MODULE', 'AVG%', 'Q_CNT'))
        print('-'*44)
        for m, scores in module_sums.items():
            avg = sum(scores)/len(scores) if scores else 0
            print(f"{m:<25} | {avg:6.2f} | {len(scores):6}")

    if bloom_sums:
        print("\nBLOOM'S TAXONOMY SUMMARY")
        print("{:<20} | {:>6} | {:>6}".format("LEVEL", 'AVG%', 'Q_CNT'))
        print('-'*38)
        for b, scores in bloom_sums.items():
            avg = sum(scores)/len(scores) if scores else 0
            print(f"{b:<20} | {avg:6.2f} | {len(scores):6}")

    avg_neatness = total_neatness / len(results) if results else 0
    final_avg_grade = grand_score / total_questions_found if total_questions_found > 0 else 0

    print("-" * 60)
    print(f"OVERALL PRESENTATION (NEATNESS):  {avg_neatness:.2f}/100")
    print(f"OVERALL KNOWLEDGE (CONTENT):     {final_avg_grade:.2f}/100")
    print(f"TOTAL QUESTIONS EVALUATED:       {total_questions_found}")
    print("="*60)

if __name__ == "__main__":
    # Ensure these filenames match your actual files in the folder
    run_paper_evaluator(STUDENT_PDF, ANSWER_KEY_PDF)