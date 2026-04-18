import streamlit as st
import pandas as pd
from agents.graph import build_graph

# Page config for a premium feel
st.set_page_config(
    page_title="AI Research Agent | Milestone 2",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3.5em;
        background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%);
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #45a049 0%, #388E3C 100%);
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.4);
        transform: translateY(-2px);
    }
    .report-card {
        padding: 30px;
        border-radius: 15px;
        background-color: #1e2130;
        border-left: 8px solid #4CAF50;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    .source-card {
        padding: 15px;
        border-radius: 10px;
        background-color: #25293d;
        margin-bottom: 10px;
        border: 1px solid #3d4156;
    }
    .source-link {
        color: #4CAF50;
        text-decoration: none;
        font-weight: bold;
    }
    .source-link:hover {
        text-decoration: underline;
    }
    .finding-item {
        margin-bottom: 10px;
        padding-left: 10px;
        border-left: 3px solid #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 Intelligent AI Research Assistant")
st.markdown("### Milestone 2: Agentic Workflow & Report Generation")
st.write("Leveraging LangGraph and BART-Large-CNN to perform autonomous research.")

with st.sidebar:
    st.header("Search Parameters")
    query = st.text_input("Enter Research Topic", placeholder="e.g., Generative AI in Drug Discovery")
    
    st.divider()
    
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
                
                st.markdown("---")
                st.subheader("Final Conclusion")
                st.info(report.get("conclusion", "Analysis complete."))
            
            with tab2:
                summaries = final_state.get("summaries", [])
                results = final_state.get("results", [])
                
                if not results:
                    st.warning("No external sources were found for this query.")
                else:
                    for i, (res, summ) in enumerate(zip(results, summaries)):
                        with st.container():
                            st.markdown(f"""
                            <div class="source-card">
                                <strong>Source {i+1}: {res.get('title', 'Untitled')}</strong><br>
                                <p style="font-size: 0.9em; color: #b0b3b8;">{summ}</p>
                                <a href="{res.get('link', '#')}" target="_blank" class="source-link">View Original Source ↗</a>
                            </div>
                            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Execution Error: {str(e)}")
            with st.expander("Technical Details"):
                st.exception(e)

elif run_research and not query:
    st.warning("⚠️ Please provide a research topic before starting the agent.")

else:
    # Beautiful landing visual
    st.markdown("""
    <div style="text-align: center; padding: 50px;">
        <h2 style="color: #4CAF50;">Welcome to Milestone 2</h2>
        <p>Start your research by entering a topic in the sidebar.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Placeholder for a professional look
    st.image("https://images.unsplash.com/photo-1518770660439-4636190af475?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80", use_column_width=True)
