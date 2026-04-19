def clean_results(results: list) -> list:
<<<<<<< HEAD
    """
    Takes raw search results and returns cleaned, enriched text list.
    Combines title + body for richer context per source.

    Input:  [{"title": "...", "body": "...", "link": "..."}]
    Output: ["clean text 1", "clean text 2", ...]
    """
=======
    """Takes raw search results and returns cleaned, enriched text list."""
>>>>>>> 93e9bc4 (refactor: modularize report generation and improve search result cleaning in agent pipeline)
    cleaned_texts = []
    for r in results:
        title = r.get("title", "").strip()
        body = r.get("body", "").strip()

        if not body or len(body) < 80:
            continue

<<<<<<< HEAD
        # Prepend title for richer LLM context
=======
>>>>>>> 93e9bc4 (refactor: modularize report generation and improve search result cleaning in agent pipeline)
        text = f"{title}. {body}" if title else body
        cleaned_texts.append(text)

    return cleaned_texts