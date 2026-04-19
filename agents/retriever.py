def clean_results(results: list) -> list:
    """
    Takes raw search results and returns cleaned text list.

    Input:
    [
        {"title": "...", "body": "...", "link": "..."}
    ]

    Output:
    [
        "clean text 1",
        "clean text 2"
    ]
    """

    cleaned_texts = []

    for r in results:
        body = r.get("body", "")

        if not body:
            continue

        text = body.strip()

        if len(text) < 100:
            continue

        cleaned_texts.append(text)

    return cleaned_texts