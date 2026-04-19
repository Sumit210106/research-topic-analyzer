from agents.llm import generate_report_llm


def generate_report(query: str, summaries: list, sources: list) -> dict:
    """
    Generates a structured research report using Groq LLM synthesis.
    Falls back to a static template if the LLM is unavailable.

    Output format:
    {
        "title": str,
        "abstract": str,
        "key_findings": list[str],
        "sources": list[str],
        "conclusion": str
    }
    """
    return generate_report_llm(query, summaries, sources)