def clean_results(results: list) -> list:
    """
    Takes search results and extracts only the bodies for summarization.
    """
    cleaned = []
    for r in results:
        body = r.get("body", "").strip()
        if body:
            cleaned.append(body)
    return cleaned