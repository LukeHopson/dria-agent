# Copyright (c) [2025] [huggingface/smolagents]
# Licensed under the MIT License
# Source: [https://github.com/huggingface/smolagents/blob/main/src/smolagents/default_tools.py]
from dria_agent.agent.tool import tool
from typing import Optional

try:
    from markdownify import markdownify
    import requests
    import re
    from requests.exceptions import RequestException
    from duckduckgo_search import DDGS
    from smolagents.utils import truncate_content
except ImportError:
    raise ImportError("Please run pip install 'dria_agent[tools]'")


@tool
def duckduckgo_search(query: str, max_results: int = 10, **kwargs) -> str:
    """
    Performs a DuckDuckGo web search and returns the top search results in md format.

    This tool uses the DuckDuckGo Search API to perform a web search based on the provided query.
    It retrieves up to `max_results` results (default is 10) and formats each result with its title,
    URL, and snippet in markdown.

    :param query: The search query to perform.
    :type query: str
    :param max_results: The maximum number of search results to return. Defaults to 10.
    :type max_results: int
    :param kwargs: Additional keyword arguments passed to the DuckDuckGo search instance.
    :returns: A markdown-formatted string containing the top search results.
    :rtype: str
    :raises ImportError: If the `duckduckgo_search` package is not installed.
    :raises Exception: If no search results are found.
    """
    ddgs = DDGS(**kwargs)
    results = ddgs.text(query, max_results=max_results)
    if len(results) == 0:
        raise Exception("No results found! Try a less restrictive/shorter query.")
    postprocessed_results = [
        f"[{result['title']}]({result['href']})\n{result['body']}" for result in results
    ]
    return "## Search Results\n\n" + "\n\n".join(postprocessed_results)


@tool
def google_search(query: str, filter_year: Optional[int] = None) -> str:
    """
    Performs a Google web search using SerpAPI and returns the top search results in markdown format.

    This tool queries the Google search engine via SerpAPI with the provided search query.
    Optionally, the results can be filtered by a specific year. The function returns the search results
    formatted as markdown with titles, links, publication dates, sources, and snippets.

    :param query: The search query to perform.
    :type query: str
    :param filter_year: Optionally restricts the search results to a specific year.
    :type filter_year: Optional[int]
    :returns: A markdown-formatted string containing the top search results.
    :rtype: str
    :raises ValueError: If the SerpAPI key is missing in the environment variables.
    :raises Exception: If no search results are found.
    """
    import os
    import requests

    serpapi_key = os.getenv("SERPAPI_API_KEY")
    if serpapi_key is None:
        raise ValueError(
            "Missing SerpAPI key. Make sure you have 'SERPAPI_API_KEY' in your env variables."
        )

    params = {
        "engine": "google",
        "q": query,
        "api_key": serpapi_key,
        "google_domain": "google.com",
    }
    if filter_year is not None:
        params["tbs"] = f"cdr:1,cd_min:01/01/{filter_year},cd_max:12/31/{filter_year}"

    response = requests.get("https://serpapi.com/search.json", params=params)
    if response.status_code == 200:
        results = response.json()
    else:
        raise ValueError(response.json())

    if "organic_results" not in results:
        if filter_year is not None:
            raise Exception(
                f"No results found for query: '{query}' with filtering on year={filter_year}. "
                "Use a less restrictive query or do not filter on year."
            )
        else:
            raise Exception(
                f"No results found for query: '{query}'. Use a less restrictive query."
            )

    if len(results["organic_results"]) == 0:
        year_filter_message = (
            f" with filter year={filter_year}" if filter_year is not None else ""
        )
        return (
            f"No results found for '{query}'{year_filter_message}. "
            "Try with a more general query, or remove the year filter."
        )

    web_snippets = []
    for idx, page in enumerate(results["organic_results"]):
        date_published = "\nDate published: " + page["date"] if "date" in page else ""
        source = "\nSource: " + page["source"] if "source" in page else ""
        snippet = "\n" + page["snippet"] if "snippet" in page else ""
        redacted_version = (
            f"{idx}. [{page['title']}]({page['link']})"
            f"{date_published}{source}\n{snippet}"
        )
        redacted_version = redacted_version.replace(
            "Your browser can't play this video.", ""
        )
        web_snippets.append(redacted_version)
    return "## Search Results\n" + "\n\n".join(web_snippets)


@tool
def visit_webpage(url: str) -> str:
    """
    Visits a webpage at the specified URL, converts its HTML content to Markdown,
    and returns the Markdown-formatted content.

    This tool fetches the content of a webpage using an HTTP GET request with a 20-second timeout.
    The HTML content is then converted to Markdown using the `markdownify` package, cleaned up,
    and truncated to a maximum length if necessary.

    :param url: The URL of the webpage to visit.
    :type url: str
    :returns: The content of the webpage as a Markdown-formatted string.
    :rtype: str
    :raises ImportError: If the required packages (`markdownify` and `requests`) are not installed.
    :raises Exception: If an error occurs during the HTTP request or content processing.
    """

    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        markdown_content = markdownify(response.text).strip()
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)
        return truncate_content(markdown_content, 10000)
    except requests.exceptions.Timeout:
        return "The request timed out. Please try again later or check the URL."
    except RequestException as e:
        return f"Error fetching the webpage: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"


SEARCH_TOOLS = [duckduckgo_search, google_search, visit_webpage]
