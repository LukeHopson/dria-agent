from dria_agent import ToolCallingAgent, tool
import requests
from datetime import datetime, timedelta
from typing import Dict, Tuple
import asyncio
# Cache to store exchange rate responses
# Structure: {base_currency: (data, expiry_timestamp)}
_CACHE: Dict[str, Tuple[dict, datetime]] = {}
# Cache duration - 24 hours since the free API updates once per day
CACHE_DURATION = timedelta(hours=24)


def _get_cached_rates(base_currency: str) -> dict:
    """
    Get exchange rates from cache or fetch from API if needed.

    Args:
    - base_currency: The base currency code

    Returns:
    - Dictionary containing exchange rate data
    """
    base_currency = base_currency.upper()
    now = datetime.now()

    # Check if we have a valid cached response
    if base_currency in _CACHE:
        data, expiry = _CACHE[base_currency]
        if now < expiry:
            return data

    # If no valid cache, fetch from API
    url = f"https://open.er-api.com/v6/latest/{base_currency}"
    response = requests.get(url)
    data = response.json()

    if data["result"] != "success":
        raise Exception(
            f"Error fetching exchange rate: {data.get('error-type', 'Unknown error')}"
        )

    # Cache the response
    _CACHE[base_currency] = (data, now + CACHE_DURATION)
    return data


@tool
def get_exchange_rate(base_currency: str, target_currency: str) -> dict:
    """
    Get the exchange rate between two currencies using the Exchange Rate API.

    Args:
    - base_currency: The base currency code (e.g., 'USD', 'EUR')
    - target_currency: The target currency code (e.g., 'JPY', 'GBP')

    Returns:
    - A dictionary containing the exchange rate information
        dict keys:
            - base_currency (str): The base currency code
            - target_currency (str): The target currency code
            - rate (float): The exchange rate
            - last_update (str): Last update time in UTC
    """
    base_currency = base_currency.upper()
    target_currency = target_currency.upper()

    # Get rates from cache or API
    data = _get_cached_rates(base_currency)

    return {
        "base_currency": base_currency,
        "target_currency": target_currency,
        "rate": data["rates"][target_currency],
        "last_update": data["time_last_update_utc"],
    }


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """
    Convert an amount from one currency to another.

    Args:
    - amount: The amount to convert
    - from_currency: The currency to convert from (e.g., 'USD', 'EUR')
    - to_currency: The currency to convert to (e.g., 'JPY', 'GBP')

    Returns:
    - A dictionary containing the conversion result
        dict keys:
            - from_amount (float): Original amount
            - from_currency (str): Original currency
            - to_amount (float): Converted amount
            - to_currency (str): Target currency
            - rate (float): Exchange rate used
    """
    # Get exchange rate
    exchange_info = get_exchange_rate(from_currency, to_currency)
    rate = exchange_info["rate"]

    converted_amount = amount * rate

    return {
        "from_amount": amount,
        "from_currency": from_currency,
        "to_amount": converted_amount,
        "to_currency": to_currency,
        "rate": rate,
    }

async def main():
    # Create an inference engine with the exchange rate tools
    agent = ToolCallingAgent(tools=[get_exchange_rate, convert_currency], backend="ollama")

    # --- Example 1: Get exchange rate ---
    query = "What's the current exchange rate from USD to EUR?"
    execution = await agent.run_feedback(query, print_results=True)

    # --- Example 2: Convert currency ---
    query = "Convert 100 USD to JPY"
    execution = await agent.run_feedback(query, print_results=True)

    # --- Example 3: Multiple conversions ---
    query = "How much is 50 EUR in USD and GBP?"
    execution = await agent.run_feedback(query, print_results=True)

if __name__ == "__main__":
    asyncio.run(main())
