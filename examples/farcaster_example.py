from dria_agent.agent import ToolCallingAgent
from dria_agent.tools.tool import tool
import requests
import os


@tool
def post_to_farcaster(text: str) -> str:
    """
    Posts a message (cast) to Farcaster network via Neynar's API.

    Args:
        text: The text content to post (must be <= 320 bytes)

    Returns:
        str: A message indicating success or failure of the post
    """
    api_key = os.environ.get("NEYNAR_API_KEY", None)
    if not api_key:
        raise EnvironmentError("Neynar API key not found.")

    # Neynar's cast endpoint
    url = "https://api.neynar.com/v2/farcaster/cast"

    headers = {
        "accept": "application/json",
        "api_key": api_key,
        "content-type": "application/json",
    }

    # You'll need to get your signer UUID from Neynar dashboard
    signer_uuid = os.environ.get("NEYNAR_SIGNER_UUID")

    payload = {"text": text, "signer_uuid": signer_uuid}

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()

        return f"Successfully posted to Farcaster: {text}"

    except requests.exceptions.RequestException as e:
        return f"Error posting to Farcaster: {str(e)}"


agent = ToolCallingAgent(tools=[post_to_farcaster], backend="mlx")

query = ""
agent.run(query=query, print_results=True)
