"""
Search tools for the ARIA research agent.

Provides web search capabilities using Tavily API.
"""

import os
import time
from dotenv import load_dotenv, find_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import tool

load_dotenv(find_dotenv())


@tool
def web_search(query: str) -> str:
    """
    Search the web using Tavily API.
    
    Returns the top 3 results with title, URL, and content.
    
    Args:
        query (str): The search query.
    
    Returns:
        str: Formatted search results or error message.
    """
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        return {"error": "TAVILY_API_KEY not found in environment"}
    
    tavily = TavilySearchResults(
        max_results=3,
        tavily_api_key=tavily_api_key
    )
    
    # Retry logic: up to 3 attempts with 2-second wait between retries
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            tavily_results = tavily.invoke(query)
            
            results = []
            for r in tavily_results:
                results.append(
                    f"Title: {r.get('title', 'No Title')}\n"
                    f"URL: {r.get('url', 'No URL')}\n"
                    f"Content: {r.get('content', 'No Content')[:600]}\n"
                )
            
            if not results:
                return f"No search results found for query: '{query}'"
            
            return "Search Results:\n\n" + "\n---\n".join(results)
        
        except Exception as e:
            if attempt < max_retries - 1:
                # Not the last attempt, wait and retry
                time.sleep(retry_delay)
            else:
                # All retries exhausted, return error dict
                return {"error": f"Search failed after {max_retries} retries: {str(e)}"}
