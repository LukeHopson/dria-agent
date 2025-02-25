from dria_agent.agent.tool import tool
import os
from typing import List, Optional

# Slack
try:
    from slack_sdk import WebClient
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    raise ImportError("Please run pip install 'dria_agent[tools]'")


SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")


@tool
def send_slack_message(
    channel: str, text: str, thread_ts: Optional[str] = None
) -> dict:
    """
    Send message to Slack channel.

    :param token: Slack API token
    :param channel: Channel ID or name
    :param text: Message text
    :param thread_ts: Thread timestamp for replies
    :return: API response
    """
    client = WebClient(token=SLACK_BOT_TOKEN)
    return client.chat_postMessage(channel=channel, text=text, thread_ts=thread_ts)


@tool
def upload_slack_file(
    channels: str, file_path: str, title: Optional[str] = None
) -> dict:
    """
    Upload file to Slack channel.

    :param channels: Channel ID or name
    :param file_path: Path to file
    :param title: File title
    :return: API response
    """
    client = WebClient(token=SLACK_BOT_TOKEN)
    return client.files_upload(channels=channels, file=file_path, title=title)


@tool
def list_slack_channels() -> List[dict]:
    """
    List all Slack channels.
    :return: List of channels
    """
    client = WebClient(token=SLACK_BOT_TOKEN)
    return client.conversations_list()["channels"]


SLACK_TOOLS = [send_slack_message, upload_slack_file, list_slack_channels]
