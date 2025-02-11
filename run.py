#from pythonic.engine import evaluate_row
#result = evaluate_row(row["mock_functions"], completion)
from agent.ollm import ToolCallingAgent
from pythonic.tool import tool


@tool
def find_next_available_slot(user_id: str, event_duration: int) -> str:
    """
    Finds the next available time slot in the user's calendar for rescheduling a meeting.

    :param user_id: The unique identifier of the user (e.g., Samantha Reynolds).
    :param event_duration: Duration of the event in minutes.
    :return: A string representing the next available time slot in the format "HH:MM".
    :raises ValueError: If user_id is invalid or no suitable time slot is found.
    """
    # For demonstration purposes, we return a fixed time.
    return "09:00"


# Create an inference engine with the available tool(s).
inference_engine = ToolCallingAgent(tools=[find_next_available_slot])

# --- Example 1: Using a query string ---
query = "When can I schedule a meeting for a 30-minute duration for Samantha Reynolds?"
final_response = inference_engine.run(query, dry_run=False)
print("Final Response (query string):")
print(final_response)

# --- Example 2: Using a list of messages (simulated conversation) ---
messages = [
    {"role": "user", "content": "I need to reschedule my meeting."},
    {"role": "assistant", "content": "What is the duration of the meeting?"},
    {"role": "user", "content": "30 minutes"}
]
final_response = inference_engine.run(messages, dry_run=True)
print("Final Response (messages list):")
print(final_response)
