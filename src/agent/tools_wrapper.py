"""
Wraps the mock tools and handles identity verification logic.
"""
import sys
from src.config import PATHS

# Add the tools directory to sys.path so we can import the provided tools.py
sys.path.append(str(PATHS.tools))
from tools import lookup_order, create_ticket

class ToolManager:
    def __init__(self):
        # In-memory store for verified orders during this session
        self.verified_orders = {}

    def verify_and_lookup(self, order_id: str, verification_value: str = None) -> dict:
        """Attempts to look up an order. Requires verification if not already done."""
        # If we already verified this order in this session, return it
        if order_id in self.verified_orders:
            return self.verified_orders[order_id]

        # Otherwise, call the mock tool (which prompts for verification if needed)
        res = lookup_order(order_id, verification_value=verification_value)
        
        # If it succeeded, cache it
        if res.get("verified"):
            self.verified_orders[order_id] = res
            
        return res

    def create_handover_ticket(self, summary: str, order_id: str = None) -> dict:
        return create_ticket(summary=summary, order_id=order_id)