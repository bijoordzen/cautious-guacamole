"""
Streamlit UI Chatbot for Techpaathshala Terms & Conditions RAG.
Demonstrates:
1. Parametric Modeling (direct LLM pre-trained memory).
2. Document Retrieval (FAISS vector store RAG).
3. Web Search (live DuckDuckGo web results).
LangGraph evaluates all 3 approaches and selects the best one.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure local directory is in Python path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# Load .env file
load_dotenv(dotenv_path=CURRENT_DIR / ".env")

import streamlit as st
from retriever import build_or_load_vectorstore, DEFAULT_FILE_PATH, DEFAULT_INDEX_DIR
from graph import run_pipeline

# Configure Streamlit page
st.set_page_config(
    page_title="Techpaathshala T&C Chatbot (RAG vs Parametric vs Web)",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .winner-badge {
        background-color: #ECFDF5;
        border-left: 5px solid #10B981;
        padding: 10px 14px;
        border-radius: 6px;
        font-weight: 600;
        color: #065F46;
        margin: 8px 0;
    }
    .rationale-box {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 0.95rem;
        color: #1E40AF;
        margin-bottom: 12px;
    }
    .score-card {
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# --- Sidebar Setup ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key Input
    saved_api_key = os.getenv("GEMINI_API_KEY", "")
    api_key = st.text_input(
        "Google Gemini API Key",
        value=saved_api_key,
        type="password",
        help="Enter your Gemini API Key. Can also be set in rag/.env"
    )
    
    # Model Selector
    model_choice = st.selectbox(
        "LLM Model",
        options=["gemini-flash-lite-latest", "gemini-3.8-flash", "gemini-3.5-flash-lite", "gpt-4o-mini"],
        index=0
    )

    st.markdown("---")
    st.header("📚 FAISS Vector Database")
    doc_exists = DEFAULT_FILE_PATH.exists()
    st.write(f"**Document**: `{DEFAULT_FILE_PATH.name}`")
    st.write(f"**Status**: {'✅ Found' if doc_exists else '❌ Missing'}")
    
    index_exists = (DEFAULT_INDEX_DIR / "index.faiss").exists()
    st.write(f"**FAISS Index**: {'✅ Built on Disk' if index_exists else '⚠️ Not built yet'}")
    
    if st.button("🔄 Rebuild FAISS Vector Index"):
        with st.spinner("Rebuilding FAISS index from documents/t&c.txt..."):
            try:
                vs = build_or_load_vectorstore(force_rebuild=True, api_key=api_key)
                st.success("FAISS index successfully rebuilt and saved!")
            except Exception as e:
                st.error(f"Error building FAISS index: {e}")

    st.markdown("---")
    st.header("💡 Sample Test Questions")
    sample_queries = [
        "What is Techpaathshala's refund policy?",
        "Where is the head office located?",
        "Can I use the course materials commercially?",
        "What happens if Techpaathshala changes its course fees?",
        "What is the policy regarding limitation of liability?",
        "What are the main programs offered by Techpaathshala?"
    ]
    
    selected_sample = None
    for q in sample_queries:
        if st.button(q, key=f"btn_{q}"):
            selected_sample = q


# --- Main Chat Page ---
st.markdown('<div class="main-title">🤖 Techpaathshala T&C Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Powered by <b>LangGraph & FAISS</b>: Comparing <b>Parametric Modeling</b> vs '
    '<b>Document Retrieval (RAG)</b> vs <b>Web Search</b> and arbitrating the best response!</div>',
    unsafe_allow_html=True
)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Message History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            # Assistant response card
            best_approach = msg.get("best_approach", "Document Retrieval (FAISS RAG)")
            rationale = msg.get("rationale", "")
            final_ans = msg.get("content", "")
            
            st.markdown(f'<div class="winner-badge">🏆 Best Approach Selected: <b>{best_approach}</b></div>', unsafe_allow_html=True)
            if rationale:
                st.markdown(f'<div class="rationale-box"><b>Arbiter Rationale:</b> {rationale}</div>', unsafe_allow_html=True)
            
            st.markdown(final_ans)

            # Interactive Comparison Accordion
            with st.expander("🔍 Inspect All 3 Approaches & Scoring Comparison", expanded=False):
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📄 Document Retrieval (FAISS)",
                    "🧠 Parametric Modeling",
                    "🌐 Web Search",
                    "📊 Evaluation Breakdown"
                ])
                
                with tab1:
                    st.subheader("Document Retrieval (FAISS RAG)")
                    st.info(msg.get("rag_response", "No RAG response generated."))
                    st.write("**Retrieved Chunks from `t&c.txt`:**")
                    docs = msg.get("retrieved_docs", [])
                    if docs:
                        for idx, d in enumerate(docs, 1):
                            with st.container():
                                st.caption(f"Chunk {idx} (Source: {d.metadata.get('source', 't&c.txt')}):")
                                st.code(d.page_content, language="text")
                    else:
                        st.caption("No chunks retrieved.")

                with tab2:
                    st.subheader("Parametric Modeling (Internal Weights Only)")
                    st.warning(msg.get("parametric_response", "No parametric response generated."))
                    st.caption("ℹ️ Note how parametric models either produce generic disclaimers or hallucinate specific internal policy clauses.")

                with tab3:
                    st.subheader("Web Search (Live Engine Results)")
                    st.info(msg.get("web_response", "No web search response generated."))
                    st.write("**Web Snippets Found:**")
                    web_snippets = msg.get("web_results", [])
                    if web_snippets:
                        for idx, item in enumerate(web_snippets, 1):
                            st.markdown(f"**[{idx}] [{item.get('title')}]({item.get('link')})**")
                            st.caption(item.get("snippet"))
                    else:
                        st.caption("No web snippets found.")

                with tab4:
                    st.subheader("Score Comparison Matrix")
                    scores = msg.get("scores", {})
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        p_score = scores.get("parametric", {}).get("score", "N/A")
                        p_notes = scores.get("parametric", {}).get("notes", "")
                        st.markdown(f"**🧠 Parametric Score**: `{p_score}/10`")
                        st.caption(p_notes)
                    with col2:
                        r_score = scores.get("rag", {}).get("score", "N/A")
                        r_notes = scores.get("rag", {}).get("notes", "")
                        st.markdown(f"**📄 FAISS RAG Score**: `{r_score}/10`")
                        st.caption(r_notes)
                    with col3:
                        w_score = scores.get("web_search", {}).get("score", "N/A")
                        w_notes = scores.get("web_search", {}).get("notes", "")
                        st.markdown(f"**🌐 Web Search Score**: `{w_score}/10`")
                        st.caption(w_notes)

# Query Execution Logic
user_prompt = st.chat_input("Ask any question regarding Techpaathshala's Terms and Conditions...")
active_query = selected_sample or user_prompt

if active_query:
    # Append user question
    st.session_state.messages.append({"role": "user", "content": active_query})
    with st.chat_message("user"):
        st.markdown(active_query)

    # Process with LangGraph Pipeline
    with st.chat_message("assistant"):
        with st.spinner("Executing LangGraph multi-approach arbitration pipeline..."):
            pipeline_result = run_pipeline(
                query=active_query,
                api_key=api_key,
                model_name=model_choice
            )

        best_approach = pipeline_result.get("best_approach", "Document Retrieval (FAISS RAG)")
        rationale = pipeline_result.get("rationale", "")
        final_answer = pipeline_result.get("final_answer", "")
        scores = pipeline_result.get("scores", {})
        rag_resp = pipeline_result.get("rag_response", "")
        parametric_resp = pipeline_result.get("parametric_response", "")
        web_resp = pipeline_result.get("web_response", "")
        docs = pipeline_result.get("retrieved_docs", [])
        web_res = pipeline_result.get("web_results", [])

        # Display Result
        st.markdown(f'<div class="winner-badge">🏆 Best Approach Selected: <b>{best_approach}</b></div>', unsafe_allow_html=True)
        if rationale:
            st.markdown(f'<div class="rationale-box"><b>Arbiter Rationale:</b> {rationale}</div>', unsafe_allow_html=True)
        st.markdown(final_answer)

        # Store in session state
        st.session_state.messages.append({
            "role": "assistant",
            "content": final_answer,
            "best_approach": best_approach,
            "rationale": rationale,
            "scores": scores,
            "rag_response": rag_resp,
            "parametric_response": parametric_resp,
            "web_response": web_resp,
            "retrieved_docs": docs,
            "web_results": web_res
        })

        # Render expander tabs for the current response
        with st.expander("🔍 Inspect All 3 Approaches & Scoring Comparison", expanded=True):
            tab1, tab2, tab3, tab4 = st.tabs([
                "📄 Document Retrieval (FAISS)",
                "🧠 Parametric Modeling",
                "🌐 Web Search",
                "📊 Evaluation Breakdown"
            ])
            with tab1:
                st.subheader("Document Retrieval (FAISS RAG)")
                st.info(rag_resp)
                st.write("**Retrieved Chunks from `t&c.txt`:**")
                for idx, d in enumerate(docs, 1):
                    st.caption(f"Chunk {idx} (Source: {d.metadata.get('source', 't&c.txt')}):")
                    st.code(d.page_content, language="text")
            with tab2:
                st.subheader("Parametric Modeling (Internal Weights Only)")
                st.warning(parametric_resp)
                st.caption("ℹ️ Notice the contrast between zero-shot internal memory vs document retrieval.")
            with tab3:
                st.subheader("Web Search (Live Engine Results)")
                st.info(web_resp)
                st.write("**Web Snippets Found:**")
                for idx, item in enumerate(web_res, 1):
                    st.markdown(f"**[{idx}] [{item.get('title')}]({item.get('link')})**")
                    st.caption(item.get("snippet"))
            with tab4:
                st.subheader("Score Comparison Matrix")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**🧠 Parametric Score**: `{scores.get('parametric', {}).get('score', 'N/A')}/10`")
                    st.caption(scores.get('parametric', {}).get('notes', ''))
                with col2:
                    st.markdown(f"**📄 FAISS RAG Score**: `{scores.get('rag', {}).get('score', 'N/A')}/10`")
                    st.caption(scores.get('rag', {}).get('notes', ''))
                with col3:
                    st.markdown(f"**🌐 Web Search Score**: `{scores.get('web_search', {}).get('score', 'N/A')}/10`")
                    st.caption(scores.get('web_search', {}).get('notes', ''))
