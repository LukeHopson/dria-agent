# Copyright (c) [2025] [SqueezeAILab/TinyAgent]
# Licensed under the MIT License
# Source: [https://github.com/SqueezeAILab/TinyAgent]


import datetime
import platform
import subprocess
from dria_agent.tools.tool import tool
from .cmd import run_applescript, run_applescript_capture, run_command
import difflib
import os
import webbrowser
from urllib.parse import quote_plus
from typing import Literal


@tool
def create_event(
    title: str,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    location: str = "",
    invitees: list[str] = [],
    notes: str = "",
    calendar: str | None = None,
) -> str:
    """
        Creates a new calendar event on macOS using AppleScript.

    This function creates an event with the specified title, start date, end date, location,
    invitees, and additional notes in the specified calendar. If the calendar parameter is not provided
    or is invalid, it defaults to the first available calendar. The function is supported only on macOS.
    On other platforms, it returns a message indicating that the method is unsupported.

    :param title: The title of the event.
    :param start_date: The start date and time of the event (datetime.datetime).
    :param end_date: The end date and time of the event (datetime.datetime).
    :param location: (Optional) The location where the event will take place. Defaults to an empty string.
    :param invitees: (Optional) A list of email addresses to invite to the event. Defaults to an empty list.
    :param notes: (Optional) Additional notes or description for the event. Defaults to an empty string.
    :param calendar: (Optional) The name of the calendar in which to create the event. If not provided,
                     the first available calendar is used. If provided but invalid, the first calendar is used.

    :returns: A string message indicating the outcome of the event creation. Returns a success message if the event
              is created successfully, or an error message if an issue occurs (e.g., unsupported platform, invalid calendar).
    """
    if platform.system() != "Darwin":
        return "This method is only supported on MacOS"

    applescript_start_date = start_date.strftime("%B %d, %Y %I:%M:%S %p")
    applescript_end_date = end_date.strftime("%B %d, %Y %I:%M:%S %p")

    # Check if the given calendar parameter is valid
    if calendar is not None:
        script = f"""
        tell application "{"Calender"}"
            set calendarExists to name of calendars contains "{calendar}"
        end tell
        """
        exists = run_applescript(script)
        if not exists:
            calendar = _get_first_calendar()
            if calendar is None:
                return f"Can't find the calendar named {calendar}. Please try again and specify a valid calendar name."

    # If it is not provded, default to the first calendar
    elif calendar is None:
        calendar = _get_first_calendar()
        if calendar is None:
            return "Can't find a default calendar. Please try again and specify a calendar name."

    invitees_script = []
    for invitee in invitees:
        invitees_script.append(
            f"""
            make new attendee at theEvent with properties {{email:"{invitee}"}}
        """
        )
    invitees_script = "".join(invitees_script)

    script = f"""
    tell application "System Events"
        set calendarIsRunning to (name of processes) contains Calender
        if calendarIsRunning then
            tell application Calender to activate
        else
            tell application Calender to launch
            delay 1
            tell application Calender to activate
        end if
    end tell
    tell application Calender
        tell calendar "{calendar}"
            set startDate to date "{applescript_start_date}"
            set endDate to date "{applescript_end_date}"
            set theEvent to make new event at end with properties {{summary:"{title}", start date:startDate, end date:endDate, location:"{location}", description:"{notes}"}}
            {invitees_script}
            switch view to day view
            show theEvent
        end tell
        tell application Calender to reload calendars
    end tell
    """

    try:
        run_applescript(script)
        return f"""Event created successfully in the "{calendar}" calendar."""
    except subprocess.CalledProcessError as e:
        return str(e)


def _get_first_calendar() -> str | None:
    script = f"""
        tell application "System Events"
            set calendarIsRunning to (name of processes) contains Calender
            if calendarIsRunning is false then
                tell application Calender to launch
                delay 1
            end if
        end tell
        tell application Calender
            set firstCalendarName to name of first calendar
        end tell
        return firstCalendarName
        """
    stdout = run_applescript_capture(script)
    if stdout:
        return stdout[0].strip()
    else:
        return None


@tool
def open_anything(name_or_path: str) -> str:
    """
    Opens an item on macOS using Spotlight search.

    This function attempts to open an application, file, or folder by either directly opening the file
    if an absolute path is provided, or by using macOS's Spotlight search (via 'mdfind') to locate the item.
    It first checks for an exact match using the display name. If no exact match is found, it performs a fuzzy
    search using difflib to determine the best match. The function returns the path of the item opened or an
    error message if the operation fails or if executed on a non-macOS platform.

    :param (str) name_or_path: A string representing either the name of the item to open or its absolute file path.
                         If an absolute path is provided (starting with "/") and the file exists, the file is opened directly.
    :returns: A string indicating the path of the item that was opened, or an error message if the item could not
              be found or opened.

    Note:
        This function is only supported on macOS.
    """
    if platform.system() != "Darwin":
        return "This method is only supported on MacOS"

    # Check if input is a path and file exists
    if name_or_path.startswith("/") and os.path.exists(name_or_path):
        try:
            subprocess.run(["open", name_or_path])
            return name_or_path
        except Exception as e:
            return f"Error opening file: {e}"

    # Use mdfind for fast searching with Spotlight
    command_search_exact = ["mdfind", f"kMDItemDisplayName == '{name_or_path}'"]
    stdout, _ = run_command(command_search_exact)

    if stdout:
        paths = stdout.strip().split("\n")
        path = paths[0] if paths else None
        if path:
            subprocess.run(["open", path])
            return path

    # If no exact match, perform fuzzy search on the file names
    command_search_general = ["mdfind", name_or_path]
    stdout, stderr = run_command(command_search_general)

    paths = stdout.strip().split("\n") if stdout else []

    if paths:
        best_match = difflib.get_close_matches(name_or_path, paths, n=1, cutoff=0.0)
        if best_match:
            _, stderr = run_command(["open", best_match[0]])
            if len(stderr) > 0:
                return f"Error: {stderr}"
            return best_match[0]
        else:
            return "No file found after fuzzy matching."
    else:
        return "No file found with exact or fuzzy name."


@tool
def open_location(query: str):
    """
    Opens a specified location in Apple Maps using a search query.

    This function constructs an Apple Maps URL based on the provided query,
    which can be a place name, an address, or geographical coordinates.
    It then opens the URL in the default web browser, allowing the user to
    view the location directly in Apple Maps.

    :param query: The search query representing the location to be opened.
                  This can be a place name, an address, or coordinates.
    :type query: str
    :returns: A confirmation message containing the Apple Maps URL that was opened.
    """
    base_url = "https://maps.apple.com/?q="
    query_encoded = quote_plus(query)
    full_url = base_url + query_encoded
    webbrowser.open(full_url)
    return f"Location of {query} in Apple Maps: {full_url}"


@tool
def show_directions(end: str, start: str = "", transport: Literal["d", "w", "r"] = "d"):
    """
    Opens Apple Maps with directions from a start location to an end location.

    Constructs a URL for Apple Maps using the specified destination, an optional
    starting point (defaults to the current location if empty), and a transport mode.
    The transport mode is specified as a single character:
      - 'd': Driving (default)
      - 'w': Walking
      - 'r': Public transit

    :param end: The destination address or location.
    :type end: str
    :param start: (Optional) The starting address or location. If empty, the current location is used.
    :type start: str
    :param transport: (Optional) Mode of transportation ('d', 'w', or 'r'). Defaults to 'd'.
    :type transport: Literal["d", "w", "r"]
    :returns: A message string containing the Apple Maps URL for the directions.
    """
    base_url = "https://maps.apple.com/?"
    if len(start) > 0:
        start_encoded = quote_plus(start)
        start_param = f"saddr={start_encoded}&"
    else:
        start_param = ""  # Use the current location
    end_encoded = quote_plus(end)
    transport_flag = f"dirflg={transport}"
    full_url = f"{base_url}{start_param}daddr={end_encoded}&{transport_flag}"
    webbrowser.open(full_url)
    return f"Directions to {end} in Apple Maps: {full_url}"


APPLE_TOOLS = [create_event, open_anything, open_location, show_directions]
