from src.agent.agent import SupportAgent

agent = SupportAgent()

def turn(text):
    print(f"\nCustomer: {text}")
    print(f"Agent: {agent.chat(text)}")

# Turn 1: No order ID yet
turn("Hi, my package is late. Can I cancel it?")

# Turn 2: Provide an order ID (unverified)
# Note: We are using a random order ID from the CSV, e.g., 69520867
turn("The order ID is 69520867.")

# Turn 3: Provide verification (pincode or phone suffix)
# Let's pretend the pincode is 110001
turn("My pincode is 160017.")

# Turn 4: Ask a follow-up based on the order status
turn("If I cancel it, how long will the refund take?")