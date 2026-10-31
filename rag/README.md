# 📑 Techpaathshala Terms & Conditions RAG System (LangGraph + FAISS + Streamlit)

A specialized testing project demonstrating the power and accuracy of Retrieval-Augmented Generation (RAG) over Parametric Modeling and Web Search for company-specific documents.

## 🎯 Key Objectives Demonstrated
When asked questions regarding Techpaathshala's Terms and Conditions (`t&c.txt`), the system generates 3 distinct answers simultaneously and evaluates them:

1. **🧠 Parametric Modeling**: Direct LLM response using only pre-trained internal weights (demonstrating hallucinations or lack of internal policy specifics).
2. **📄 Document Retrieval (FAISS RAG)**: Exact, grounded retrieval from `t&c.txt` indexed in a FAISS vector database.
3. **🌐 Web Search**: Live internet search synthesis using DuckDuckGo.
4. **⚖️ LangGraph Arbiter / Evaluator**: Analyzes and scores all 3 candidate answers on Groundedness, Accuracy, and Hallucination risk, and dynamically selects the **Best Approach**.

---

## 🏗️ Architecture & File Structure

```
rag/
├── documents/
│   └── t&c.txt             # Source terms and conditions document
├── faiss_index/            # Persistent FAISS vector database
├── retriever.py            # Chunker, FAISS vector index builder & retriever
├── generator.py            # Handlers for Parametric, RAG, Web Search & Evaluator
├── graph.py                # LangGraph StateGraph orchestration pipeline
├── app.py                  # Streamlit UI chatbot with 3-way inspection tabs
├── requirements.txt        # Python dependencies
└── .env                    # Environment keys (GEMINI_API_KEY)
```

---

## 🚀 How to Run

1. Open your terminal in the workspace and activate the environment:
   ```powershell
   cd c:\Users\Srinivas\Documents\projects\cautious-guacamole
   .\.venv\Scripts\Activate.ps1
   ```

2. Run the Streamlit application:
   ```powershell
   streamlit run rag/app.py
   ```

3. Open the displayed local URL (typically `http://localhost:8501`) in your browser.
