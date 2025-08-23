import logging, time, json
from typing import Dict, Any, Optional
from datetime import datetime
import requests
import contextvars

# Thread-safe trace ID storage
_trace_id_var = contextvars.ContextVar("trace_id", default=None)

def set_trace_id(value: Optional[str]) -> None:
    _trace_id_var.set(value)

def get_trace_id() -> Optional[str]:
    return _trace_id_var.get()

def clear_trace_id() -> None:
    _trace_id_var.set(None)

class LLMLogger:
    """Logger for LLM operations with Loki integration"""
    def __init__(self, loki_url: str = "http://localhost:3100", service: str = "pdf_rag"):
        self.loki_push_url = f"{loki_url}/loki/api/v1/push"
        self.service = service

        # Setup console logger
        self.logger = logging.getLogger("llm_operations")
        if not self.logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
            self.logger.addHandler(h)
        self.logger.setLevel(logging.INFO)

    def _send_to_loki(self, message: str, labels: Dict[str, str]) -> None:
        """Send log message to Loki with trace ID if available"""
        tid = get_trace_id()
        if tid and "trace_id" not in labels:
            labels = {**labels, "trace_id": tid}

        ts_ns = int(time.time() * 1e9)
        payload = {"streams": [{"stream": labels, "values": [[str(ts_ns), message]]}]}
        try:
            requests.post(
                self.loki_push_url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=2,
            )
        except Exception:
            # Silently ignore Loki failures
            pass

    def log_llm_call(
        self,
        *,
        model: str,
        duration: float = 0.0,
        prompt: Optional[str] = None,
        response: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        operation_type: str = "generation",
        status: str = "success",
        error: Optional[str] = None,
        **extra: Any,
    ) -> None:
        """Log LLM API calls with metrics and optional Loki integration"""
        # Truncate long texts for logging
        clip = lambda s, n=2000: (s[:n] + "…") if s and len(s) > n else s
        data: Dict[str, Any] = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "model": model,
            "operation": operation_type,
            "duration_s": round(float(duration), 3),
            "prompt": clip(prompt),
            "response": clip(response),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "status": status,
            **extra,
        }
        tid = get_trace_id()
        if tid:
            data["trace_id"] = tid

        message = json.dumps({k: v for k, v in data.items() if v is not None})
        labels = {
            "service": self.service,
            "component": "llm",
            "event_type": "llm_call",
            "status": status,
        }
        self.logger.info(message)
        self._send_to_loki(message, labels)

    def log_rag_operation(
        self,
        *,
        operation: str,
        duration: float = 0.0,
        documents_count: Optional[int] = None,
        status: str = "success",
        error: Optional[str] = None,
        **extra: Any,
    ) -> None:
        """Log RAG operations like indexing and generation"""
        data: Dict[str, Any] = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "operation": operation,
            "duration_s": round(float(duration), 3),
            "documents_count": documents_count,
            "status": status,
            **extra,
        }
        tid = get_trace_id()
        if tid:
            data["trace_id"] = tid

        message = json.dumps({k: v for k, v in data.items() if v is not None})
        labels = {
            "service": self.service,
            "component": "rag",
            "event_type": "rag_op",
            "status": status,
        }
        self.logger.info(message)
        self._send_to_loki(message, labels)

# Global instance + convenience functions
llm_logger = LLMLogger()

def log_llm_call(**kwargs): llm_logger.log_llm_call(**kwargs)
def log_rag_operation(**kwargs): llm_logger.log_rag_operation(**kwargs)

__all__ = ["log_llm_call", "log_rag_operation", "set_trace_id", "get_trace_id", "clear_trace_id"]