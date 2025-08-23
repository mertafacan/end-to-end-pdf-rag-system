import os, uuid, time
from typing import List, Dict, Literal, TypedDict
from functools import lru_cache
from pathlib import Path
import numpy as np

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Qdrant
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import END, StateGraph, START
from langchain_core.documents import Document
from langchain_litellm import ChatLiteLLM
from sentence_transformers import CrossEncoder
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker

load_dotenv()

# Optional logging - graceful fallback if not available
try:
    from loki_logger import log_llm_call
except ImportError:
    def log_llm_call(**kwargs): pass

def _parse_bool(v: str, default: bool = True) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

# Environment configuration with defaults
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME")
RERANK_TOPK = int(os.getenv("RERANK_TOPK"))
RERANK_CHOOSE = int(os.getenv("RERANK_CHOOSE"))
RERANK_THRESHOLD_RAW = float(os.getenv("RERANK_THRESHOLD_RAW"))
LLM_MODEL = os.getenv("LLM_MODEL")
RERANK_ENABLED = _parse_bool(os.getenv("RERANK_ENABLED"))
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ---------- State ----------
class AgentState(TypedDict):
    question: str
    messages: List[str]
    generation: str
    retriever_docs: List[Document]
    reflection_round: int
    max_reflections: int
    force_skip_reflection: bool
    relevance_score: float

# ---------- Init ----------
def init_models():
    """Initialize LLM and embedding models"""
    llm = ChatLiteLLM(model=LLM_MODEL, temperature=0.2)
    emb = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", api_key=GOOGLE_API_KEY)
    return llm, emb

@lru_cache(maxsize=1)
def get_cross_encoder(name: str = RERANK_MODEL_NAME) -> CrossEncoder:
    """Load cross-encoder model with caching"""
    return CrossEncoder(name)

def preload_startup():
    """Preload models during startup for faster first requests"""
    if RERANK_ENABLED:
        _ = get_cross_encoder()

def _ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def save_pdf_bytes(files: List[tuple[str, bytes]], target_dir: str) -> List[tuple[str, str]]:
    """Save uploaded PDF bytes to disk with unique filenames"""
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    paths: List[tuple[str, str]] = []
    for name, data in files:
        stem = Path(name).stem
        out = Path(target_dir) / f"{stem}_{uuid.uuid4().hex}.pdf"
        out.write_bytes(data)
        paths.append((str(out), name))
    return paths

def _load_docs_with_meta(saved_files: List[tuple[str, str]], session_id: str) -> List[Document]:
    """Load PDF documents and add session metadata"""
    docs = []
    for file_path, original_name in saved_files:
        chunks = PyPDFLoader(file_path).load_and_split()
        for d in chunks:
            d.metadata.update({
                "session_id": session_id,
                "original_filename": original_name,
                "page": d.metadata.get("page", 0)
            })
            docs.append(d)
    return docs

def upsert_to_qdrant(docs: List[Document], emb) -> Qdrant:
    return Qdrant.from_documents(
        documents=docs,
        embedding=emb,
        url=QDRANT_URL,
        collection_name=QDRANT_COLLECTION
    )

def build_session_retriever(session_id: str, vectorstore: Qdrant):
    """Create retriever filtered by session ID"""
    flt = Filter(must=[FieldCondition(key="metadata.session_id", match=MatchValue(value=session_id))])
    return vectorstore.as_retriever(search_kwargs={"k": RERANK_TOPK, "filter": flt})

def _format_context(docs: List[Document]) -> str:
    """Format documents into context string with metadata"""
    parts = []
    for d in docs:
        if d and d.page_content.strip():
            filename = d.metadata.get("original_filename", "document.pdf")
            page = d.metadata.get("page", 0) + 1
            parts.append(f"[{Path(filename).name} p{page}]\n{d.page_content.strip()}")
    return "\n\n---\n\n".join(parts)

# ---------- Rerank ----------
def rerank_with_crossencoder(query: str, docs: List[Document]) -> List[Document]:
    """Rerank documents using cross-encoder model (optional)"""
    if not docs:
        return []
    if not RERANK_ENABLED:
        # Rerank false
        return docs[:min(RERANK_CHOOSE, len(docs))]
    ranker = HuggingFaceCrossEncoder(model_name=RERANK_MODEL_NAME)
    compressor = CrossEncoderReranker(model=ranker, top_n=min(RERANK_CHOOSE, len(docs)))
    ranked = list(compressor.compress_documents(docs, query))
    # Add raw scores to metadata
    try:
        scores = ranker.score([(query, d.page_content) for d in ranked])
        for d, s in zip(ranked, scores):
            md = dict(d.metadata or {})
            md["rerank_score_raw"] = float(s)
            d.metadata = md
    except Exception:
        pass
    return ranked

# ---------- Graph ----------
def build_graph(retriever, llm, max_reflections: int = 3):
    """Build LangGraph workflow for RAG with reflection"""
    def retrieve_docs(s: AgentState):
        """Retrieve and rerank relevant documents"""
        topk = retriever.invoke(s["question"])
        topn = rerank_with_crossencoder(s["question"], topk)
        avg_raw = float(np.mean([d.metadata.get("rerank_score_raw", 0.0) for d in topn])) if topn else 0.0
        # Skip reflection if relevance is high enough
        force_skip = avg_raw >= RERANK_THRESHOLD_RAW
        return {"retriever_docs": topn, "relevance_score": avg_raw, "force_skip_reflection": force_skip}

    def decide_llm(s: AgentState):
        """Check if documents can answer the question"""
        sys = SystemMessage(content="You are a document analysis expert. Your job is to check if the given document can answer the given question. Your answers should always be either 'yes' or 'no'.")
        context = _format_context(s["retriever_docs"])
        hum = HumanMessage(content=f"Check if the given document can answer the question... Only answer 'yes' or 'no'.\n\nDOCUMENT:\n{context}\n\nQUESTION:\n{s['question']}")
        r = llm.invoke([sys, hum])
        return {"messages": [(r.content or "").strip()]}

    def decision_node(s: AgentState) -> Literal["generate_llm","irrelevant_llm"]:
        return "generate_llm" if s["messages"] and s["messages"][-1].strip().lower()=="yes" else "irrelevant_llm"

    def irrelevant_llm(s: AgentState):
        return {"messages": ["The given document is not relevant to your question."]}

    def generate_llm(s: AgentState):
        """Generate answer based on retrieved documents"""
        start_time = time.time()
        
        sys = SystemMessage(content="You are an expert at answering questions related to documents. You should only use the information from the document provided to answer the question.")
        context = _format_context(s["retriever_docs"])
        hum = HumanMessage(content=f"Examine the given document and question, and provide an appropriate answer. Question: {s['question']}. \n\n Document: \n{context}\n\n Your answer should be concise and to the point.")
        
        try:
            r = llm.invoke([sys, hum])
            duration = time.time() - start_time
            
            # Log LLM call for monitoring
            log_llm_call(
                model=LLM_MODEL,
                prompt_tokens=len(sys.content) + len(hum.content),
                completion_tokens=len(r.content) if r.content else 0,
                duration=duration,
                status="success",
                prompt=f"{sys.content}\n\n{hum.content}",
                response=r.content,
                operation_type="generation"
            )
            
            return {"messages": [r.content], "generation": r.content}
        except Exception as e:
            duration = time.time() - start_time
            log_llm_call(model=LLM_MODEL, duration=duration, status="error", error=str(e), operation_type="generation")
            raise

    def reflect_decide_llm(s: AgentState):
        """Evaluate if answer needs improvement"""
        start_time = time.time()
        
        draft = s.get("generation", s["messages"][-1])
        sys = SystemMessage(content=("You are a quality control expert. Evaluate the following draft response based only on the given document in terms of accuracy, correctness, and fully answering the question. If the response is unclear/incomplete or contains inferences outside the document, write only 'yes'. Otherwise, write only 'no'."))
        context = _format_context(s["retriever_docs"])
        hum = HumanMessage(content=f"QUESTION: {s['question']}\n\nDOCUMENT:\n{context}\n\nDRAFT: {draft}")
        
        try:
            v = llm.invoke([sys, hum]).content.strip().lower()
            duration = time.time() - start_time
            
            log_llm_call(
                model=LLM_MODEL,
                prompt_tokens=len(sys.content) + len(hum.content),
                completion_tokens=len(v),
                duration=duration,
                status="success",
                prompt=f"{sys.content}\n\n{hum.content}",
                response=v,
                operation_type="quality_check"
            )
            
            return {"messages": [v]}
        except Exception as e:
            duration = time.time() - start_time
            log_llm_call(model=LLM_MODEL, duration=duration, status="error", error=str(e), operation_type="quality_check")
            raise

    def reflection_node(s: AgentState) -> Literal["reflect_llm","skip_reflection"]:
        """Decide whether to improve the answer or skip reflection"""
        if s.get("force_skip_reflection", False): return "skip_reflection"
        verdict = s["messages"][-1].strip().lower()
        return "reflect_llm" if verdict=="yes" and s.get("reflection_round",0)<s.get("max_reflections",max_reflections) else "skip_reflection"

    def reflect_llm(s: AgentState):
        """Improve the draft answer based on documents"""
        draft = s.get("generation", s["messages"][-1])
        sys = SystemMessage(content=("Improve the following draft response based ONLY on the given documents. Don't write unnecessary things."))
        context = _format_context(s["retriever_docs"])
        hum = HumanMessage(content=f"QUESTION: {s['question']}\n\nDOCUMENT:\n{context}\n\nDRAFT: {draft}")
        v = (llm.invoke([sys, hum]).content or "").strip()
        return {"generation": v or draft, "reflection_round": int(s.get("reflection_round",0))+1}

    def skip_reflection(s: AgentState):
        return {}

    g = StateGraph(AgentState)

    g.add_node("retrieve_docs", retrieve_docs)
    g.add_node("decide_llm", decide_llm)
    g.add_node("generate_llm", generate_llm)
    g.add_node("irrelevant_llm", irrelevant_llm)
    g.add_node("reflect_decide_llm", reflect_decide_llm)
    g.add_node("reflect_llm", reflect_llm)
    g.add_node("skip_reflection", skip_reflection)

    g.add_edge(START, "retrieve_docs")
    g.add_edge("retrieve_docs", "decide_llm")
    g.add_conditional_edges("decide_llm", decision_node, {"generate_llm": "generate_llm", "irrelevant_llm": "irrelevant_llm"})
    g.add_edge("generate_llm", "reflect_decide_llm")

    g.add_conditional_edges(
        "reflect_decide_llm", reflection_node,
        {"reflect_llm": "reflect_llm", "skip_reflection": "skip_reflection"}
    )
    g.add_edge("reflect_llm", "reflect_decide_llm")

    g.add_edge("skip_reflection", END)
    g.add_edge("irrelevant_llm", END)

    return g.compile()

# ---------- Public ----------
def index_pdfs_from_bytes(files: List[tuple[str, bytes]], session_id: str):
    """Main function to index PDF files and create RAG graph"""
    work_dir = os.path.join("uploaded_docs", session_id); _ensure_dir(work_dir)
    saved = save_pdf_bytes(files, work_dir)
    llm, emb = init_models()

    # Load and chunk documents
    chunks = _load_docs_with_meta(saved, session_id)

    # Create vector store and retriever
    vs = upsert_to_qdrant(chunks, emb)
    retriever = build_session_retriever(session_id, vs)
    return build_graph(retriever, llm)

def run_agent(graph, question: str):
    """Execute RAG workflow and return answer with sources"""
    state = {"question": question, "messages": [], "generation": "", "retriever_docs": [],
             "reflection_round": 0, "max_reflections": 3, "force_skip_reflection": False,
             "relevance_score": 0.0}
    
    result = graph.invoke(state)
    answer = result.get("generation") or (result.get("messages", [])[-1] if result.get("messages") else "Could not generate answer.")

    # Format sources for response
    sources = [
        {
            "label": Path(doc.metadata.get("original_filename", "document.pdf")).name,
            "text": doc.page_content,
            "score_raw": doc.metadata.get("rerank_score_raw", 0.0),
            "page": doc.metadata.get("page", 0),
        }
        for doc in result.get("retriever_docs", [])
    ]

    # Add metadata about the retrieval process
    meta = {
        "relevance_avg": result.get("relevance_score", 0.0),
        "threshold_raw": RERANK_THRESHOLD_RAW,
        "reflection_skipped": result.get("force_skip_reflection", False),
    }

    return answer, sources, meta