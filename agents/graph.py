from typing import TypedDict, List
from langgraph.graph import StateGraph

from agents.search import search_web
from agents.retriever import clean_results
from agents.llm import summarize_all
from agents.report import generate_report


class State(TypedDict):
    query: str
    results: List[dict]
    summaries: List[str]
    report: dict



def search_node(state: State):
    query = state["query"]
    results = search_web(query)
    return {"results": results}


def summarize_node(state: State):
    results = state.get("results", [])
    cleaned = clean_results(results)
    summaries = summarize_all(cleaned)
    return {"summaries": summaries}


def report_node(state: State):
    query = state["query"]
    results = state.get("results", [])
    summaries = state.get("summaries", [])

    sources = [r.get("link", "") for r in results]
    report = generate_report(query, summaries, sources)
    return {"report": report}



def build_graph():
    graph = StateGraph(State)


    graph.add_node("search", search_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("report", report_node)
    graph.set_entry_point("search")
    graph.add_edge("search", "summarize")
    graph.add_edge("summarize", "report")

    return graph.compile()