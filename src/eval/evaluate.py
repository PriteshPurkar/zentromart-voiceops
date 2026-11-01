# src/eval/evaluate.py
import csv
import time
from pathlib import Path
from src.config import PATHS, GROQ_MODEL
from src.llm import get_llm_client
from src.rag.retriever import get_retriever

def check_facts(answer: str, must_include: str) -> bool:
    """Checks if the answer contains all the required facts."""
    if not must_include:
        return True
        
    answer_lower = answer.lower()
    # The CSV format uses " ; " to separate independent facts
    facts = [f.strip().lower() for f in must_include.split(";") if f.strip()]
    
    for fact in facts:
        # Alternatives are separated by " / "
        alternatives = [a.strip() for a in fact.split("/")]
        if not any(alt in answer_lower for alt in alternatives):
            return False
    return True

def run_evaluation():
    print("Loading practice questions...")
    questions_path = PATHS.practice / "questions.csv"
    if not questions_path.exists():
        print(f"Error: {questions_path} not found.")
        return

    with open(questions_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    # Only evaluate questions that have an answer (not the "unanswerable" ones)
    answerable_qs = [q for q in reader if q.get("answerable", "").lower() == "true"]
    print(f"Loaded {len(answerable_qs)} answerable questions for evaluation.")

    client = get_llm_client()
    retriever = get_retriever()

    step1_correct = 0
    step2_correct = 0

    for i, q in enumerate(answerable_qs):
        question = q["question"]
        must_include = q["must_include"]
        
        print(f"[{i+1}/{len(answerable_qs)}] {question[:50]}...", end="", flush=True)

        # --- STEP 1: Prompt Only ---
        step1_prompt = f"Answer the following customer question as ZentroMart support: {question}"
        res1 = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": step1_prompt}],
            temperature=0.1,
            max_tokens=150
        )
        ans1 = res1.choices[0].message.content
        if check_facts(ans1, must_include):
            step1_correct += 1
            print(" [S1: ✓]", end="")
        else:
            print(" [S1: ✗]", end="")

        # --- STEP 2: RAG ---
        chunks = retriever.search(question, k=3)
        context = "\n".join([c.chunk.text for c in chunks])
        step2_prompt = f"Answer the customer question using ONLY this context:\n{context}\n\nQuestion: {question}"
        
        res2 = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": step2_prompt}],
            temperature=0.1,
            max_tokens=150
        )
        ans2 = res2.choices[0].message.content
        if check_facts(ans2, must_include):
            step2_correct += 1
            print(" [S2: ✓]")
        else:
            print(" [S2: ✗]")
            
        # Small sleep to avoid Groq rate limits
        time.sleep(0.5)

    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    print(f"Total Questions: {len(answerable_qs)}")
    print(f"Step 1 (Prompting Only) Accuracy: {step1_correct}/{len(answerable_qs)} ({(step1_correct/len(answerable_qs))*100:.1f}%)")
    print(f"Step 2 (RAG) Accuracy:            {step2_correct}/{len(answerable_qs)} ({(step2_correct/len(answerable_qs))*100:.1f}%)")
    print("="*40)
    print("Save these numbers for your report!")

if __name__ == "__main__":
    run_evaluation()