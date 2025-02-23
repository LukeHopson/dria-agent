from dria_agent import ToolCallingAgent, tool
import asyncio

@tool
def check_availability(day: str, start_time: str, end_time: str) -> bool:
    """
    Check if a time slot is available on a given day.

    Args:
    - day: The day to check in YYYY-MM-DD format
    - start_time: Start time in HH:MM format
    - end_time: End time in HH:MM format

    Returns:
    - True if slot is available, False otherwise
    """
    if day == "2024-12-11" and start_time == "15:00" and end_time == "16:00":
        return False
    return True


@tool
def make_appointment(day: str, start_time: str, end_time: str) -> dict:
    """
    Make an appointment for a given time slot.

    Args:
    - day: The day to make appointment in YYYY-MM-DD format
    - start_time: Start time in HH:MM format
    - end_time: End time in HH:MM format
    - title: The title of the appointment

    Returns:
    - A dictionary with the appointment details and if it's made or not.
        dict keys:
            - day (str): The day the appointment is on, in YYYY-MM-DD format
            - start_time (str): Start time in HH:MM format
            - end_time (str): End time in HH:MM format
            - appointment_made (bool): Whether the appointment is successfully made or not.
    """
    return {
        "day": day,
        "start_time": start_time,
        "end_time": end_time,
        "appointment_made": True,
    }


@tool
def add_to_reminders(reminder_text: str) -> bool:
    """
    Add a text to reminders.

    Args:
    - reminder_text: The text to add to reminders

    Returns:
    - Whether the reminder was successfully created or not.
    """
    print(f"Reminder added: {reminder_text}")
    return True


agent = ToolCallingAgent(tools=[add_to_reminders, check_availability, make_appointment])

async def main():
    # --- Example 1: Parallel calls ---
    query = "Schedule a meeting with my thesis supervisor today from 15:00 to 16:00 and add it to my reminders."
    execution = await agent.run_feedback(query, num_tools=3, print_results=True)

if __name__ == "__main__":
    asyncio.run(main())
