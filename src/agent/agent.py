"""
The conversational agent for Step 3.
Handles multi-turn history, order lookups, and ticket creation.
"""
import re
import json
from word2number import w2n
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

    def _normalize_stt_numbers(self, text: str) -> str:
        """Converts spoken number words (e.g., 'six nine') to digits."""
        # Replace 'pin code' with 'pincode'
        text = text.lower().replace("pin code", "pincode")
        
        num_map = {
            "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"
        }
        
        words = text.split()
        result = []
        current_number_block = ""
        
        for word in words:
            # Clean punctuation to check if it's a number
            clean_word = word.strip(",.?!")
            
            if clean_word in num_map:
                # Append the digit to our current block
                current_number_block += num_map[clean_word]
            else:
                # If the word is not a number, save the current block (if any)
                if current_number_block:
                    result.append(current_number_block)
                    current_number_block = ""
                # Add the non-number word
                result.append(word)
                
        # Catch any number block at the end of the sentence
        if current_number_block:
            result.append(current_number_block)
            
        return " ".join(result)

    def _extract_order_id(self, text: str) -> str | None:
        """Looks for an 8-digit order ID in the text."""
        match = re.search(r'\b\d{8}\b', text.replace(" ", ""))
        return match.group(0) if match else None

    def chat(self, user_message: str) -> str:
        # Normalize STT numbers into digits
        user_message = self._normalize_stt_numbers(user_message)
        
        self.history.append({"role": "user", "content": user_message})
        
        # 1. Extract numbers directly from the normalized string
        import re
        # Find all sequences of digits in the message
        all_numbers = re.findall(r'\d+', user_message)
        
        order_id = None
        verify_value = None
        
        # A valid order ID is 8 digits, a valid pincode is 6 digits, phone suffix is 4 digits
        for num in all_numbers:
            if len(num) == 8:
                order_id = num
            elif len(num) == 6:
                verify_value = num
            elif len(num) == 4:
                verify_value = num
        
        # If no order ID in this message, check history
        if not order_id:
            for msg in reversed(self.history):
                if msg["role"] == "user":
                    history_nums = re.findall(r'\d+', msg["content"])
                    for num in history_nums:
                        if len(num) == 8:
                            order_id = num
                            break
                    if order_id:
                        break

        tool_context = ""
        order_data = None
        
        # 2. Tool Logic
        if order_id:
            order_data = self.tools.verify_and_lookup(order_id, verification_value=verify_value)
            
            if order_data.get("verified") and verify_value:
                # HARD OVERRIDE: Return immediately
                status = order_data.get('status', 'unknown')
                reply = f"Thank you, your identity is verified. Your order is currently {status}. How can I help you with this?"
                self.history.append({"role": "assistant", "content": reply})
                return reply
                
            elif order_id in self.tools.verified_orders:
                status = order_data.get('status', 'unknown')
                total = order_data.get('total_inr', 'unknown')
                items = ", ".join([i['title'] for i in order_data.get('items', [])])
                tool_context = f"[SYSTEM NOTE: The customer is VERIFIED. Order Status: '{status}'. Total: {total} INR. Items: {items}]"
            elif order_data.get("verification_required"):
                tool_context = "[SYSTEM NOTE: Verification required. Ask the customer for their 4-digit phone suffix or 6-digit pincode.]"
            elif order_data.get("verified") is False:
                tool_context = "[SYSTEM NOTE: Verification FAILED. Apologize and ask them to try again.]"

        # 3. Retrieve RAG context
        chunks = self.retriever.search(user_message, k=3)
        context_text = "\n".join([f"[{c.chunk.section_id}] {c.chunk.text}" for c in chunks])
        
        # 4. Build the prompt
        dynamic_prompt = f"Context from documents:\n{context_text}\n"
        
        enhanced_system = f"{self.system_prompt}\n\n{tool_context}\n"
        messages = [{"role": "system", "content": enhanced_system}]
        messages.extend(self.history[-4:]) 
        messages[-1]["content"] = f"{dynamic_prompt}\nCustomer message: {user_message}"
        
        # 5. Get LLM Response
        client = get_llm_client()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=300,
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