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

st.title("🤖 Intelligent AI Research Assistant")
st.markdown("### Milestone 2: Agentic Workflow & Report Generation")
st.write("Leveraging LangGraph and BART-Large-CNN to perform autonomous research.")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Research Agent")
    st.write("**Milestone 2: Agentic AI**")
    
    st.markdown("---")
    query = st.text_input("Enter a research topic:", placeholder="e.g., AI in drug discovery")
    
    run_research = st.button("Run Research Agent", use_container_width=True, type="primary")
    
    st.markdown("""
    **Agentic Workflow:**
    1. 🔍 **Web Search**: Crawls top research findings.
    2. 🧹 **Clean & Extract**: Processes raw HTML/Text data.
    3. 📝 **AI Summarization**: Generates concise insights using LLMs.
    4. 📊 **Report Synthesis**: Compiles a structured research report.
    """)
    
    run_research = st.button("🚀 Execute Agent")
    
    st.divider()
    st.info("Milestone 2 uses an automated graph-based approach, connecting multiple specialized agents.")

if run_research and query:
    with st.spinner("Agentic Workflow in progress... Initializing LangGraph..."):
        try:
            # Initialize graph
            graph = build_graph()
            
            # Initial state
            initial_state = {"query": query}
            
            # Run graph
            final_state = graph.invoke(initial_state)
            
            st.success(f"Successfully synthesized research for: **{query}**")
            
            # Create tabs for Report and Sources
            tab1, tab2 = st.tabs(["📑 Research Report", "🔗 Sources & Summaries"])
            
            with tab1:
                report = final_state.get("report", {})
                
                st.markdown(f"""
                <div class="report-card">
                    <h2>{report.get('title', 'Research Report')}</h2>
                    <hr style="border: 0.5px solid #3d4156;">
                    <h3>Abstract</h3>
                    <p style="font-style: italic; color: #b0b3b8;">{report.get('abstract', 'No abstract available.')}</p>
                    <br>
                    <h3>Key Insights</h3>
                </div>
                """, unsafe_allow_html=True)
                
                for finding in report.get("key_findings", []):
                    st.markdown(f"""
                    <div class="finding-item">
                        {finding}
                    </div>
                    """, unsafe_allow_html=True)
                
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
