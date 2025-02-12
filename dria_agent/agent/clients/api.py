from openai import OpenAI
from dria_agent.agent.settings.providers import PROVIDER_URLS
from typing import List, Dict


# Initialize OpenAI clients for each provider
CLIENTS = {}
for provider, (url, api_key) in PROVIDER_URLS.items():
    CLIENTS[provider] = OpenAI(api_key=api_key, base_url=url)


def get_completion(
    model_name: str, provider: str, messages: List[Dict[str, str]], options=None
) -> str:
    """
    Get a completion from a model for a given provider.

    Args:
        model_name: The name of the model to use
        provider: The provider to use
        system_prompt: The system prompt to use
        user_query: The user query to use

    Returns:
        The completion from the model
    """
    client = CLIENTS.get(provider)
    if not client:
        raise ValueError(f"Provider '{provider}' not recognized.")

    if options:
        response = client.chat.completions.create(
            model=model_name, messages=messages, **options
        )
    else:
        response = client.chat.completions.create(
            model=model_name, messages=messages, temperature=0.0
        )
    return response.choices[0].message.content


def embed(
    model_name: str, provider: str, texts: List[str], options: dict = None
) -> List[List[float]]:
    """
    Get an embedding for the given text using a specified model and provider.
    """
    client = CLIENTS.get(provider)
    if not client:
        raise ValueError(f"Provider '{provider}' not recognized.")

    if options:
        response = client.embeddings.create(model=model_name, input=texts, **options)
    else:
        response = client.embeddings.create(model=model_name, input=texts)
    return [d.embedding for d in response.data]
