"""
The conversational agent for Step 3.
Handles multi-turn history, order lookups, and ticket creation.
"""
import re
import json
from typing import List, Dict

from src.llm import get_llm_client, GROQ_MODEL
from src.rag.retriever import get_retriever
from src.agent.tools_wrapper import ToolManager

class SupportAgent:
    def __init__(self):
        self.history: List[Dict[str, str]] = []
        self.tools = ToolManager()
        self.retriever = get_retriever()
        
        self.system_prompt = """You are the ZentroMart customer support assistant.
You must follow these rules strictly:
1. You must answer from the provided documents. If the answer is not there, say so.
2. If you need the user's order ID, ask for it.
3. Be polite, concise, and professional. Never output empty responses."""

    def _extract_order_id(self, text: str) -> str | None:
        """Looks for an 8-digit order ID in the text."""
        match = re.search(r'\b\d{8}\b', text.replace(" ", ""))
        return match.group(0) if match else None

    def chat(self, user_message: str) -> str:
        self.history.append({"role": "user", "content": user_message})
        
        # 1. Retrieve RAG context
        chunks = self.retriever.search(user_message, k=3)
        context_text = "\n".join([f"[{c.chunk.section_id}] {c.chunk.text}" for c in chunks])
        
        # 2. Tool Logic
        order_id = self._extract_order_id(user_message)
        if not order_id:
            for msg in reversed(self.history):
                if msg["role"] == "user":
                    order_id = self._extract_order_id(msg["content"])
                    if order_id:
                        break
        
        tool_context = ""
        order_data = None
        gave_verification_this_turn = False
        
        if order_id:
            # Look for 4 or 6 digit code in current message, excluding the order ID
            clean_message = user_message.replace(order_id, "")
            verify_match = re.search(r'\b(\d{4}|\d{6})\b', clean_message)
            verify_value = verify_match.group(0) if verify_match else None
            
            order_data = self.tools.verify_and_lookup(order_id, verification_value=verify_value)
            
            if order_id in self.tools.verified_orders:
                if verify_value:  # They JUST provided a code this turn
                    gave_verification_this_turn = True
                
                status = order_data.get('status', 'unknown')
                total = order_data.get('total_inr', 'unknown')
                items = ", ".join([i['title'] for i in order_data.get('items', [])])
                tool_context = f"[SYSTEM NOTE: The customer is VERIFIED. Order Status: '{status}'. Total: {total} INR. Items: {items}]"
            elif order_data.get("verification_required"):
                tool_context = "[SYSTEM NOTE: Verification required. Ask the customer for their 4-digit phone suffix or 6-digit pincode.]"
            elif order_data.get("verified") is False:
                tool_context = "[SYSTEM NOTE: Verification FAILED. Apologize and ask them to try again.]"

        # 3. Build the prompt
        dynamic_prompt = f"Context from documents:\n{context_text}\n"
        
        # 4. If they just verified, override and skip the LLM entirely
        if gave_verification_this_turn:
            status = order_data.get('status')
            reply = f"Thank you, your identity is verified. Your order is currently {status}. How can I help you with this?"
            self.history.append({"role": "assistant", "content": reply})
            return reply

        # 5. Otherwise, use LLM
        enhanced_system = f"{self.system_prompt}\n\n{tool_context}\n"
        messages = [{"role": "system", "content": enhanced_system}]
        messages.extend(self.history[-4:]) 
        messages[-1]["content"] = f"{dynamic_prompt}\nCustomer message: {user_message}"
        
        client = get_llm_client()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=250,
        )
        
        reply = response.choices[0].message.content
        
        # 6. Fallback
        if not reply or not reply.strip():
            if tool_context and "VERIFIED" in tool_context:
                reply = f"Thank you, your identity is verified. Your order is currently {order_data.get('status')}."
            elif tool_context and "FAILED" in tool_context:
                reply = "I'm sorry, that pincode/phone didn't match our records. Could you please double-check and try again?"
            elif tool_context and "required" in tool_context:
                reply = "Could you please provide the last 4 digits of your phone or your delivery pincode to verify your order?"
            else:
                reply = "I'm sorry, I didn't quite catch that."
        else:
            reply = reply.strip()
            
        self.history.append({"role": "assistant", "content": reply})
        return reply