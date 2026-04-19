"""
Web search using the `ddgs` package (formerly duckduckgo_search).
Returns top results for a given research query.
"""

from ddgs import DDGS


def search_web(query: str, max_results: int = 6) -> list:
    """
    Searches the web using DuckDuckGo and returns top results.

    Output format:
    [{"title": "...", "body": "...", "link": "..."}]
    """
    results = []
    try:
        raw = DDGS().text(query, max_results=max_results)
        for r in raw:
            results.append({
                "title": r.get("title", ""),
                "body": r.get("body", ""),
                "link": r.get("href", ""),
            })
    except Exception as e:
        print(f"❌ Search Error: {e}")

    return results