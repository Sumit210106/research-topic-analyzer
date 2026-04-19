def clean_results(results: list) -> list:
    """Takes raw search results and returns cleaned, enriched text list."""
    cleaned_texts = []
    for r in results:
        title = r.get("title", "").strip()
        body = r.get("body", "").strip()

        if not body or len(body) < 80:
            continue

        text = f"{title}. {body}" if title else body
        cleaned_texts.append(text)

    return cleaned_texts