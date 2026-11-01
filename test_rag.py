# test_rag.py
from src.rag.retriever import get_retriever
from src.llm import generate_response

def ask(question: str):
    print(f"\nQ: {question}")
    retriever = get_retriever()
    # Get top 3 chunks to keep the context small and fast
    chunks = retriever.search(question, k=3) 
    
    print("--- Retrieved Sections:", [c.chunk.section_id for c in chunks], "---")
    answer = generate_response(question, chunks)
    print(f"A: {answer}")

if __name__ == "__main__":
    # Test 1: A question that IS in the documents
    ask("What is the return window for electronics?")
    
    # Test 2: A question that IS NOT in the documents
    ask("Do you sell live dinosaurs?")