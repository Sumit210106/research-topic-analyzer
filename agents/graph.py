from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph

from agents.search import search_web
from agents.retriever import clean_results
from agents.llm import summarize_all
from agents.report import generate_report

class State(TypedDict):
    query: str
    results: List[dict]
    summaries: List[str]
    validated: bool
    report: dict
    error: Optional[str]

def search_node(state: State) -> dict:
    try:
        results = search_web(state["query"])
        return {"results": results, "error": None}
    except Exception as e:
        return {"results": [], "error": f"Search failed: {str(e)}"}

def summarize_node(state: State) -> dict:
    try:
        results = state.get("results", [])
        cleaned = clean_results(results)
        summaries = summarize_all(cleaned)
        return {"summaries": summaries, "error": None}
    except Exception as e:
        return {"summaries": [], "error": f"Summarization failed: {str(e)}"}

def validate_node(state: State) -> dict:
    results = state.get("results", [])
    summaries = state.get("summaries", [])
    if not results:
        return {"validated": False, "error": "No search results found. Try a different query."}
    if not summaries or all(len(s) < 20 for s in summaries):
        return {"validated": False, "error": "Retrieved content was too sparse to summarize."}
    return {"validated": True, "error": None}

def report_node(state: State) -> dict:
    try:
        query = state["query"]
        results = state.get("results", [])
        summaries = state.get("summaries", [])
        sources = [r.get("link", "") for r in results if r.get("link")]
        report = generate_report(query, summaries, sources)
        return {"report": report, "error": None}
    except Exception as e:
        return {
            "report": {
                "title": state["query"],
                "abstract": "Report generation encountered an error.",
                "key_findings": [],
                "sources": [],
                "conclusion": str(e)
            },
            "error": f"Report generation failed: {str(e)}"
        }

def build_graph():
    graph = StateGraph(State)
    graph.add_node("search", search_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("validate", validate_node)
    graph.add_node("report", report_node)
    graph.set_entry_point("search")
    graph.add_edge("search", "summarize")
    graph.add_edge("summarize", "validate")
    graph.add_edge("validate", "report")
    graph.set_finish_point("report")
    return graph.compile()