"""
LLM interface for Groq.
Handles system prompts and the RAG context injection.
"""
import os
from typing import List
from groq import Groq

from src.config import GROQ_API_KEY, GROQ_MODEL
from src.rag.retriever import RetrievedChunk

# Initialize the client once
_client = None

def get_llm_client() -> Groq:
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set in .env")
        _client = Groq(api_key=GROQ_API_KEY)
    return _client

def generate_response(
    query: str, 
    retrieved_chunks: List[RetrievedChunk], 
    chat_history: List[dict] = None
) -> str:
    """
    Generates a response using the retrieved chunks as context.
    chat_history is a list of {"role": "user"|"assistant", "content": "..."} for multi-turn.
    """
    client = get_llm_client()
    
    # 1. Format the context from the retrieved chunks
    context_blocks = []
    for i, res in enumerate(retrieved_chunks):
        c = res.chunk
        context_blocks.append(f"--- Document: {c.doc_id} (Section: {c.section_id}) ---\n{c.text}\n")
    context_text = "\n".join(context_blocks)
    
    # 2. Define the strict system prompt
    system_prompt = """You are the ZentroMart customer support assistant. 
You must follow these rules strictly:
1. Answer the customer's question using ONLY the provided document sections.
2. If the answer is not contained in the documents, say: "I don't have that information in my documents. I can create a ticket to connect you with a human agent."
3. Never invent policies, dates, or amounts.
4. Keep your answers clear, direct, and polite.
5. Do not mention that you are reading "document sections" to the user; just answer naturally."""

    # 3. Build the message array
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add chat history (for multi-turn Step 3)
    if chat_history:
        messages.extend(chat_history)
        
    # Add the current user query WITH the context
    user_message = f"Here are the relevant documents for this question:\n\n{context_text}\n\nCustomer Question: {query}"
    messages.append({"role": "user", "content": user_message})

    # 4. Call Groq
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.1, # Low temperature for factual grounding
        max_tokens=300,
    )
    
    return response.choices[0].message.content.strip()