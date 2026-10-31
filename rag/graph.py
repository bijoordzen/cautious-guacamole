"""
LangGraph Workflow Orchestration for Multi-Approach RAG Comparison.
Orchestrates:
1. Retrieval Branch: FAISS Vectorstore query & RAG generation.
2. Web Search Branch: Live DuckDuckGo search & synthesis generation.
3. Parametric Branch: Direct zero-shot LLM memory generation.
4. Arbiter / Evaluator Node: Compares all three candidates, selects the best approach,
   and outputs the authoritative answer with scoring rationale.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, TypedDict
from dotenv import load_dotenv

# Load local environment if present
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document

from retriever import retrieve_relevant_documents, build_or_load_vectorstore
from generator import (
    get_llm,
    generate_parametric_response,
    generate_rag_response,
    perform_web_search,
    generate_web_response,
    evaluate_and_select_best,
)


class RagGraphState(TypedDict, total=False):
    query: str
    api_key: Optional[str]
    provider: str
    model_name: Optional[str]
    # Branch 1: Parametric
    parametric_response: str
    # Branch 2: Document Retrieval (FAISS)
    retrieved_docs: List[Document]
    rag_response: str
    # Branch 3: Web Search
    web_results: List[Dict[str, str]]
    web_response: str
    # Synthesis & Evaluation
    best_approach: str
    scores: Dict[str, Any]
    rationale: str
    final_answer: str


# --- Node Definitions ---

def retrieve_docs_node(state: RagGraphState) -> Dict[str, Any]:
    """Node: Retrieves relevant clauses from FAISS vector database."""
    query = state["query"]
    api_key = state.get("api_key")
    docs = retrieve_relevant_documents(query=query, k=3, api_key=api_key)
    return {"retrieved_docs": docs}


def web_search_node(state: RagGraphState) -> Dict[str, Any]:
    """Node: Searches the web for relevant context."""
    query = state["query"]
    results = perform_web_search(query=query, max_results=3)
    return {"web_results": results}


def parametric_generator_node(state: RagGraphState) -> Dict[str, Any]:
    """Node: Generates response using purely internal parametric knowledge."""
    query = state["query"]
    api_key = state.get("api_key")
    model_name = state.get("model_name")
    llm = get_llm(model_name=model_name, api_key=api_key)
    resp = generate_parametric_response(query=query, llm=llm, api_key=api_key)
    return {"parametric_response": resp}


def rag_generator_node(state: RagGraphState) -> Dict[str, Any]:
    """Node: Generates response grounded in retrieved FAISS document context."""
    query = state["query"]
    docs = state.get("retrieved_docs", [])
    api_key = state.get("api_key")
    model_name = state.get("model_name")
    llm = get_llm(model_name=model_name, api_key=api_key)
    resp = generate_rag_response(query=query, retrieved_docs=docs, llm=llm, api_key=api_key)
    return {"rag_response": resp}


def web_generator_node(state: RagGraphState) -> Dict[str, Any]:
    """Node: Generates response synthesized from live web search."""
    query = state["query"]
    web_res = state.get("web_results", [])
    api_key = state.get("api_key")
    model_name = state.get("model_name")
    llm = get_llm(model_name=model_name, api_key=api_key)
    resp = generate_web_response(query=query, web_results=web_res, llm=llm, api_key=api_key)
    return {"web_response": resp}


def evaluator_node(state: RagGraphState) -> Dict[str, Any]:
    """Node: Evaluates all 3 candidate answers and determines the best approach."""
    query = state["query"]
    parametric_ans = state.get("parametric_response", "")
    rag_ans = state.get("rag_response", "")
    web_ans = state.get("web_response", "")
    retrieved_docs = state.get("retrieved_docs", [])
    web_results = state.get("web_results", [])
    api_key = state.get("api_key")
    model_name = state.get("model_name")
    llm = get_llm(model_name=model_name, api_key=api_key)

    evaluation = evaluate_and_select_best(
        query=query,
        parametric_ans=parametric_ans,
        rag_ans=rag_ans,
        web_ans=web_ans,
        retrieved_docs=retrieved_docs,
        web_results=web_results,
        llm=llm,
        api_key=api_key
    )

    return {
        "best_approach": evaluation.get("best_approach", "Document Retrieval (FAISS RAG)"),
        "scores": evaluation.get("scores", {}),
        "rationale": evaluation.get("rationale", ""),
        "final_answer": evaluation.get("final_answer", rag_ans)
    }


# --- Graph Construction ---

def create_rag_graph():
    """Builds and compiles the LangGraph pipeline."""
    workflow = StateGraph(RagGraphState)

    # Register nodes
    workflow.add_node("retrieve_docs", retrieve_docs_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate_parametric", parametric_generator_node)
    workflow.add_node("generate_rag", rag_generator_node)
    workflow.add_node("generate_web", web_generator_node)
    workflow.add_node("evaluator", evaluator_node)

    # Fan-out from START to all three branches
    workflow.add_edge(START, "retrieve_docs")
    workflow.add_edge(START, "web_search")
    workflow.add_edge(START, "generate_parametric")

    # Sequential dependent steps
    workflow.add_edge("retrieve_docs", "generate_rag")
    workflow.add_edge("web_search", "generate_web")

    # Fan-in to evaluator node
    workflow.add_edge("generate_parametric", "evaluator")
    workflow.add_edge("generate_rag", "evaluator")
    workflow.add_edge("generate_web", "evaluator")

    # Termination
    workflow.add_edge("evaluator", END)

    return workflow.compile()


# Global compiled graph instance
rag_graph = create_rag_graph()


def run_pipeline(
    query: str,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    provider: str = "auto"
) -> Dict[str, Any]:
    """
    Executes the full LangGraph pipeline for a user query.
    Returns the comprehensive state dictionary.
    """
    initial_state: RagGraphState = {
        "query": query,
        "api_key": api_key,
        "model_name": model_name,
        "provider": provider
    }
    result = rag_graph.invoke(initial_state)
    return result


if __name__ == "__main__":
    print("[Testing LangGraph Pipeline] Executing sample run...")
    sample_query = "What is Techpaathshala's refund policy?"
    out = run_pipeline(sample_query)
    print("\n--- Pipeline Result ---")
    print(f"Best Approach: {out.get('best_approach')}")
    print(f"Rationale: {out.get('rationale')}")
    print(f"\nFinal Answer:\n{out.get('final_answer')}")
