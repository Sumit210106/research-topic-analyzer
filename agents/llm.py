import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
_MODEL = "llama-3.1-8b-instant"


def summarize(text: str) -> str:
    """Summarizes a single text in a detailed analytical paragraph using Groq."""
    try:
        text = text.strip()
        if not text:
            return ""

        response = _client.chat.completions.create(
            model=_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an academic researcher. Analyze and summarize the core arguments, "
                        "key data points, and context of the given text in a highly detailed, analytical format. "
                        "Do not just skim it; extract the most valuable insights in 4-6 sentences."
                    )
                },
                {"role": "user", "content": text[:4000]}
            ],
            temperature=0.3,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"❌ Groq summarize error: {e}")
        return text[:400]


def summarize_all(texts: list) -> list:
    """Batch summarize a list of texts."""
    return [summarize(t) for t in texts]


def generate_report_llm(query: str, summaries: list, sources: list) -> dict:
    """
    Uses Groq to synthesize a structured, extensive research report.
    """
    if not summaries:
        return _fallback_report(query, sources)

    combined = "\n\n".join(f"Source {i+1}: {s}" for i, s in enumerate(summaries))

    prompt = f"""You are a Lead Research Data Analyst synthesizing information about '{query}'.
You will write a comprehensive, deeply analytical, academic-grade research report based entirely on the provided Source Summaries.

SOURCE SUMMARIES:
{combined[:6000]}

INSTRUCTIONS:
1. "title": Provide a purely academic, compelling research title.
2. "abstract": Write an executive abstract (5-7 sentences) summarizing the methodology (what was analyzed), core theme, and macroscopic insights.
3. "key_findings": Extract 5 to 8 highly detailed, substantive paragraphs (NOT one-liners). Each finding must synthesize facts, data, and critical analysis. 
4. "conclusion": Write a deep, 4-6 sentence academic conclusion synthesizing implications, limitations, and future outlooks.

Return your response as a valid JSON object with exactly this structure:
{{
  "title": "<str>",
  "abstract": "<str>",
  "key_findings": ["<str>", "<str>"],
  "conclusion": "<str>"
}}
Return ONLY the raw JSON. No backticks, no explanations. Do not acknowledge."""

    try:
        response = _client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": "You are a specialized AI data parser that only outputs pure, valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=2048,
        )

        raw = response.choices[0].message.content.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        report = json.loads(raw.strip())
        report["sources"] = sources
        return report

    except Exception as e:
        print(f"❌ Groq report generation error: {e}")
        return _fallback_report(query, sources, summaries)


def _fallback_report(query: str, sources: list, summaries: list = None) -> dict:
    """Static fallback report when LLM call fails."""
    return {
        "title": f"Research Report: {query}",
        "abstract": summaries[0] if summaries else "No sufficient information found for this query.",
        "key_findings": summaries[:5] if summaries else ["No findings available."],
        "sources": sources,
        "conclusion": (
            "Based on multiple sources, this topic shows significant advancements "
            "and growing adoption across various domains."
        )
    }