def generate_report(query: str, summaries: list, sources: list) -> dict:
    """
    Generates structured research report.

    Output format:
    {
        "title": str,
        "abstract": str,
        "key_findings": list,
        "sources": list,
        "conclusion": str
    }
    """

    # fallback
    if not summaries:
        return {
            "title": query,
            "abstract": "No sufficient information found for this query.",
            "key_findings": [],
            "sources": sources,
            "conclusion": "Please try a more specific or different query."
        }

    # Abstract (short intro)
    abstract = summaries[0]

    # Key findings (limit to 4–5)
    key_findings = summaries[:5]

    # Simple reasoning for conclusion
    conclusion = (
        "Based on multiple sources, this topic shows significant advancements "
        "and growing adoption across various domains."
    )

    return {
        "title": query,
        "abstract": abstract,
        "key_findings": key_findings,
        "sources": sources,
        "conclusion": conclusion
    }

    