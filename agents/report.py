from agents.llm import generate_report_llm

<<<<<<< HEAD

=======
>>>>>>> 93e9bc4 (refactor: modularize report generation and improve search result cleaning in agent pipeline)
def generate_report(query: str, summaries: list, sources: list) -> dict:
    """
    Generates a structured research report using Groq LLM synthesis.
    Falls back to a static template if the LLM is unavailable.
<<<<<<< HEAD

    Output format:
    {
        "title": str,
        "abstract": str,
        "key_findings": list[str],
        "sources": list[str],
        "conclusion": str
    }
=======
>>>>>>> 93e9bc4 (refactor: modularize report generation and improve search result cleaning in agent pipeline)
    """
    return generate_report_llm(query, summaries, sources)