import streamlit as st
import os
import sys

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Intelligent Research Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .report-card {
        background-color: #f8f9fa;
        color: #1e1e1e;
        padding: 24px;
        border-radius: 8px;
        border-left: 4px solid #0056b3;
        margin-bottom: 24px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .report-title {
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 12px;
        color: #111;
    }
    .finding-item {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 16px;
        border-radius: 6px;
        margin-bottom: 12px;
        color: #333;
    }
    div[data-testid="stSidebarNav"] {
        display: none;
    }
    
    @media (prefers-color-scheme: dark) {
        .report-card { background-color: #1e1e1e; color: #f0f0f0; border-left: 4px solid #4da3ff; }
        .report-title { color: #f0f0f0; }
        .finding-item { background-color: #2b2b2b; border: 1px solid #444; color: #eee; }
    }
</style>
""", unsafe_allow_html=True)

from agents.graph import build_graph

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Research Agent")
    st.write("**Milestone 2: Agentic AI**")
    
    st.markdown("---")
    query = st.text_input("Enter a research topic:", placeholder="e.g., AI in drug discovery")
    
    run_research = st.button("Run Research Agent", use_container_width=True, type="primary")
    
    st.markdown("---")
    st.subheader("Agent Pipeline")
    st.write("1. Web Search")
    st.write("2. Content Cleaning")
    st.write("3. LLM Summarization")
    st.write("4. Quality Validation")
    st.write("5. Report Synthesis")

    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.write("### Navigation")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Milestone 1", key="btn_m1", use_container_width=True):
            os.execv(sys.executable, ['python', '-m', 'streamlit', 'run', 'app.py'])
    with col2:
        if st.button("Milestone 2", key="btn_m2", use_container_width=True, type="primary"):
            pass

# ── Main content ───────────────────────────────────────────────────────────────
if run_research and not query.strip():
    st.warning("Please enter a research topic to proceed.")

elif run_research and query.strip():
    st.subheader(f'Researching: "{query}"')
    
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    try:
        status_text.info("Initializing agent and performing web search...")
        
        graph = build_graph()
        initial_state = {
            "query": query,
            "results": [],
            "summaries": [],
            "validated": False,
            "report": {},
            "error": None,
        }

        final_state = graph.invoke(initial_state)

        error_msg = final_state.get("error")
        validated = final_state.get("validated", False)
        report = final_state.get("report", {})
        summaries = final_state.get("summaries", [])
        results = final_state.get("results", [])

        progress_bar.progress(100)
        
        if not validated or error_msg:
            status_text.error(f"Validation Issue: {error_msg or 'Retrieved results were deemed insufficient.'}")
        else:
            status_text.success("Research synthesis completed successfully.")

        st.divider()

        def build_markdown(report_data, results_list):
            lines = [
                f"# {report_data.get('title', query)}", "",
                "## Abstract", report_data.get("abstract", ""), "",
                "## Key Findings"
            ]
            for i, f in enumerate(report_data.get("key_findings", []), 1):
                lines.append(f"{i}. {f}")
            lines.extend(["", "## Conclusion", report_data.get("conclusion", ""), "", "## Sources"])
            for i, r in enumerate(results_list, 1):
                lines.append(f"{i}. [{r.get('title', 'Source')}]({r.get('link', '#')})")
            return "\n".join(lines)

        md_content = build_markdown(report, results)
        st.download_button(
            label="Export Report as Markdown",
            data=md_content,
            file_name=f"research_report.md",
            mime="text/markdown",
        )

        st.write("")

        tab1, tab2 = st.tabs(["Structured Report", f"Sources ({len(results)})"])

        with tab1:
            st.markdown(f"""
            <div class="report-card">
                <div class="report-title">{report.get('title', query)}</div>
                <b>Abstract:</b><br/>{report.get('abstract', 'No abstract available.')}
            </div>
            """, unsafe_allow_html=True)
            
            st.write("#### Key Findings")
            findings = report.get("key_findings", [])
            if findings:
                for item in findings:
                    st.markdown(f'<div class="finding-item">{item}</div>', unsafe_allow_html=True)
            else:
                st.info("No key findings detected.")
                
            st.divider()
            st.write("#### Conclusion")
            st.info(report.get("conclusion", "Analysis complete."))

        with tab2:
            if not results:
                st.warning("No linked sources available.")
            else:
                for i, r in enumerate(results, 1):
                    with st.expander(f"Source {i}: {r.get('title', 'Untitled')}", expanded=True):
                        if i-1 < len(summaries):
                            st.write(f"**Summary:** {summaries[i-1]}")
                        st.write(f"[Read Original Article]({r.get('link', '#')})")

    except Exception as e:
        progress_bar.progress(0)
        status_text.error(f"An error occurred during execution: {str(e)}")

else:
    st.header("Intelligent AI Research Assistant")
    st.write(
        "Welcome to the **Milestone 2 Agentic Analysis System**. "
        "This tool automates the process of researching any given topic by deploying "
        "an autonomous agent that searches the web, summarizes literature, "
        "verifies data quality, and organizes the findings into a clear, structured format."
    )
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("**Capabilities**\n- Autonomous Web Extraction\n- Groq LLaMA 3 Summarization\n- Automated Validation\n- Actionable Insight extraction")
    with col2:
        st.success("**Architecture**\n- Orchestrated via **LangGraph**\n- Stateless Python Agents\n- Highly Responsive UI \n- Exportable Markdown Reports")
    
    st.divider()
    st.write("**Enter a query in the sidebar to begin analyzing.**")
