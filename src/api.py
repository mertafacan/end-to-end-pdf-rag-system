from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel
import uuid, time

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from helper_func import index_pdfs_from_bytes, run_agent, preload_startup

# Optional logging - graceful fallback if not available
try:
    from loki_logger import log_rag_operation, set_trace_id, clear_trace_id
except ImportError:
    def log_rag_operation(**kwargs): pass
    def set_trace_id(*a, **k): pass
    def clear_trace_id(): pass

app = FastAPI(title="RAG API")

# Prometheus metrics for monitoring
REQUEST_COUNT = Counter("rag_requests_total", "Total requests", ["endpoint"])
REQUEST_LATENCY = Histogram("rag_request_duration_seconds", "Request duration (s)", ["endpoint"])

Instrumentator().instrument(app)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Store active sessions in memory
SESSIONS = {}

class AskPayload(BaseModel):
    session_id: str
    question: str

@app.on_event("startup")
async def _startup():
    # Preload models for faster first requests
    preload_startup()

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
def health():
    return JSONResponse({"status": "ok"})

@app.post("/index")
async def index(files: list[UploadFile] = File(...)):
    # Generate unique trace ID for this request
    trace_id = uuid.uuid4().hex
    set_trace_id(trace_id)
    start = time.time()
    REQUEST_COUNT.labels(endpoint="/index").inc()
    try:
        # Create new session and process uploaded PDFs
        session_id = uuid.uuid4().hex
        data = [(f.filename, await f.read()) for f in files]
        graph = index_pdfs_from_bytes(data, session_id)
        SESSIONS[session_id] = graph

        dur = time.time() - start
        REQUEST_LATENCY.labels(endpoint="/index").observe(dur)
        log_rag_operation(operation="indexing", documents_count=len(files), duration=dur, status="success")

        return {"session_id": session_id, "count": len(files), "trace_id": trace_id}
    except Exception as e:
        dur = time.time() - start
        REQUEST_LATENCY.labels(endpoint="/index").observe(dur)
        log_rag_operation(operation="indexing", duration=dur, status="error", error=str(e))
        return JSONResponse({"error": str(e), "trace_id": trace_id}, status_code=500)
    finally:
        clear_trace_id()

@app.post("/ask")
async def ask(payload: AskPayload):
    trace_id = uuid.uuid4().hex
    set_trace_id(trace_id)
    start = time.time()
    REQUEST_COUNT.labels(endpoint="/ask").inc()
    try:
        # Validate session exists
        if payload.session_id not in SESSIONS:
            dur = time.time() - start
            REQUEST_LATENCY.labels(endpoint="/ask").observe(dur)
            log_rag_operation(operation="generation", duration=dur, status="error", error="invalid_session")
            return JSONResponse({"error": "Invalid session_id. Call /index first.", "trace_id": trace_id}, status_code=400)

        # Run RAG pipeline to generate answer
        answer, sources, meta = run_agent(SESSIONS[payload.session_id], payload.question.strip())

        dur = time.time() - start
        REQUEST_LATENCY.labels(endpoint="/ask").observe(dur)
        log_rag_operation(operation="generation", documents_count=len(sources), duration=dur, status="success")

        return {"answer": answer, "sources": sources, "meta": meta, "trace_id": trace_id}
    except Exception as e:
        dur = time.time() - start
        REQUEST_LATENCY.labels(endpoint="/ask").observe(dur)
        log_rag_operation(operation="generation", duration=dur, status="error", error=str(e))
        return JSONResponse({"error": str(e), "trace_id": trace_id}, status_code=500)
    finally:
        clear_trace_id()