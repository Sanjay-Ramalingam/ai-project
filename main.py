import cv2
import os
import numpy as np
from processor import DocumentProcessor
from evaluator import HandwritingEvaluator

# --- CONFIGURATION ---
STUDENT_PDF = "student_script.pdf"
ANSWER_KEY_PDF = "answer_key.pdf"
FAST_MODE = False  # Set to TRUE to skip OCR for quick vision/neatness testing
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

    master_key_map = evaluator.extract_key_from_pdf(processor, key_path)
    
    if not master_key_map:
        print("Error: Could not extract questions from Answer Key.")
        return

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
        
        cv2.imshow("Paper Evaluator - Analysis Feed", display)
        
        all_page_results.append({
            "page": i + 1,
            "neatness": neatness,
            "grading": page_grading,
            "words": word_count
        })

        # Press 'q' to stop early, any other key for next page
        if cv2.waitKey(0) == ord('q'):
            break

    # --- PHASE 3: FINAL ACADEMIC SCORECARD ---
    generate_final_report(all_page_results)
    cv2.destroyAllWindows()

def generate_final_report(results):
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