from dria_agent.tools.tool import tool
import os
from typing import List, Optional

# Gmail
try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    from email.mime.text import MIMEText
    import base64
except ImportError:
    raise ImportError(
        "Please install 'google-auth-oauthlib' and 'google-api-python-client'"
    )

# Slack
try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
except ImportError:
    raise ImportError("Please install 'slack_sdk'")

# Telegram
try:
    import telegram
except ImportError:
    raise ImportError("Please install 'python-telegram-bot'")


# Gmail Tools
@tool
def initialize_gmail_service() -> object:
    """
    Initialize Gmail API service.

    :return: Gmail service object
    """
    SCOPES = [
        "https://www.googleapis.com/auth/gmail.send",
        "https://www.googleapis.com/auth/gmail.readonly",
    ]
    creds = None

    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


@tool
def send_gmail(
    service: object, to: str, subject: str, body: str, from_email: str = "me"
) -> dict:
    """
    Send email using Gmail API.

    :param service: Gmail service object
    :param to: Recipient email address
    :param subject: Email subject
    :param body: Email body
    :param from_email: Sender email address
    :return: Send message details
    """
    message = MIMEText(body)
    message["to"] = to
    message["from"] = from_email
    message["subject"] = subject

    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
    return (
        service.users()
        .messages()
        .send(userId="me", body={"raw": raw_message})
        .execute()
    )


@tool
def list_gmail_messages(
    service: object, max_results: int = 10, query: str = None
) -> List[dict]:
    """
    List Gmail messages.

    :param service: Gmail service object
    :param max_results: Maximum number of messages to return
    :param query: Search query
    :return: List of messages
    """
    results = (
        service.users()
        .messages()
        .list(userId="me", maxResults=max_results, q=query)
        .execute()
    )

    messages = results.get("messages", [])
    return [
        service.users().messages().get(userId="me", id=msg["id"]).execute()
        for msg in messages
    ]


# Slack Tools
@tool
def send_slack_message(
    token: str, channel: str, text: str, thread_ts: Optional[str] = None
) -> dict:
    """
    Send message to Slack channel.

    :param token: Slack API token
    :param channel: Channel ID or name
    :param text: Message text
    :param thread_ts: Thread timestamp for replies
    :return: API response
    """
    client = WebClient(token=token)
    return client.chat_postMessage(channel=channel, text=text, thread_ts=thread_ts)


@tool
def upload_slack_file(
    token: str, channels: str, file_path: str, title: Optional[str] = None
) -> dict:
    """
    Upload file to Slack channel.

    :param token: Slack API token
    :param channels: Channel ID or name
    :param file_path: Path to file
    :param title: File title
    :return: API response
    """
    client = WebClient(token=token)
    return client.files_upload(channels=channels, file=file_path, title=title)


@tool
def list_slack_channels(token: str) -> List[dict]:
    """
    List all Slack channels.

    :param token: Slack API token
    :return: List of channels
    """
    client = WebClient(token=token)
    return client.conversations_list()["channels"]


# Telegram Tools
@tool
def send_telegram_message(
    bot_token: str, chat_id: str, text: str, parse_mode: Optional[str] = None
) -> dict:
    """
    Send Telegram message.

    :param bot_token: Telegram bot token
    :param chat_id: Chat ID
    :param text: Message text
    :param parse_mode: Message parse mode (HTML/Markdown)
    :return: Message info
    """
    bot = telegram.Bot(token=bot_token)
    return bot.send_message(chat_id=chat_id, text=text, parse_mode=parse_mode)


@tool
def send_telegram_file(
    bot_token: str, chat_id: str, file_path: str, caption: Optional[str] = None
) -> dict:
    """
    Send file via Telegram.

    :param bot_token: Telegram bot token
    :param chat_id: Chat ID
    :param file_path: Path to file
    :param caption: File caption
    :return: Message info
    """
    bot = telegram.Bot(token=bot_token)
    return bot.send_document(
        chat_id=chat_id, document=open(file_path, "rb"), caption=caption
    )


@tool
def get_telegram_updates(
    bot_token: str, offset: Optional[int] = None, limit: int = 100
) -> List[dict]:
    """
    Get Telegram bot updates.

    :param bot_token: Telegram bot token
    :param offset: Update ID offset
    :param limit: Maximum number of updates
    :return: List of updates
    """
    bot = telegram.Bot(token=bot_token)
    return bot.get_updates(offset=offset, limit=limit)


GMAIL_TOOLS = [initialize_gmail_service, send_gmail, list_gmail_messages]

SLACK_TOOLS = [send_slack_message, upload_slack_file, list_slack_channels]

TELEGRAM_TOOLS = [send_telegram_message, send_telegram_file, get_telegram_updates]
