"""ARC v2 — single app serving both the UI (/) and the API (/query)."""
import os, time, threading
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Optional

from arc_v2 import answer_question

MAX_QUESTION_CHARS = 500
MAX_CONTEXT_CHARS = 20000
RATE_LIMIT = 30          # requests per IP per hour (protects the free Groq quota)
WINDOW_SEC = 3600

app = FastAPI(title="ARC v2 API", version="1.0.0")
_hits = defaultdict(deque)
# arc_v2 keeps the current document in a module-level variable, so requests
# must run one at a time or two users' documents could get mixed up.
_lock = threading.Lock()


class QueryRequest(BaseModel):
    question: str = Field(..., description="The question to answer")
    context: Optional[str] = Field("", description="Optional document text")


def _check_rate_limit(request: Request):
    fwd = request.headers.get("x-forwarded-for", "")
    ip = fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "unknown")
    now = time.time()
    q = _hits[ip]
    while q and now - q[0] > WINDOW_SEC:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        raise HTTPException(429, "Rate limit reached (30 questions/hour). Please try again later.")
    q.append(now)


@app.get("/", response_class=HTMLResponse)
def home():
    return (Path(__file__).parent / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {"status": "ok", "groq_key_set": bool(os.environ.get("GROQ_API_KEY"))}


@app.post("/query")
def query(req: QueryRequest, request: Request):
    q = (req.question or "").strip()
    ctx = req.context or ""
    if not q:
        raise HTTPException(400, "question cannot be empty")
    if len(q) > MAX_QUESTION_CHARS:
        raise HTTPException(400, f"question too long (max {MAX_QUESTION_CHARS} characters)")
    if len(ctx) > MAX_CONTEXT_CHARS:
        raise HTTPException(400, f"context too long (max {MAX_CONTEXT_CHARS} characters)")
    _check_rate_limit(request)

    start = time.time()
    try:
        with _lock:
            r = answer_question(q, context=ctx)
    except Exception as e:
        raise HTTPException(500, f"ARC v2 failed: {str(e)[:200]}")

    return {
        "answer": str(r.get("answer", "")),
        "confidence": str(r.get("confidence", "")),
        "votes": str(r.get("votes", "")),
        "agent_1_answer": str(r.get("a1", "")),
        "agent_2_answer": str(r.get("a2", "")),
        "agent_3_answer": str(r.get("a3", "")),
        "latency_sec": round(time.time() - start, 2),
        "llm_calls": r.get("steps", 0),
        "tool_calls": r.get("tool_calls", 0),
    }
