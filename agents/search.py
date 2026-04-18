from duckduckgo_search import DDGS


def search_web(query: str) -> list:
    """
    Takes a query string and returns top 5 search results.

    Output format:
    [
        {
            "title": "...",
            "body": "...",
            "link": "..."
        }
    ]
    """

    results = []

    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=5):
                results.append({
                    "title": r.get("title", ""),
                    "body": r.get("body", ""),
                    "link": r.get("href", "")
                })

    except Exception as e:
        print("❌ Search Error:", e)

    return results