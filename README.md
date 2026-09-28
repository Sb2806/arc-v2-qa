# ARC v2 — Triple-Agent Majority-Vote Question Answering

Live demo: _add your Render link here_

Three independent agents answer the same question (a context reader, a ReWOO tool-planner, and a
full-context reader). A voter picks the majority answer and reports confidence
(HIGH = 3/3 agree, MEDIUM = 2/3, LOW = 1/3, with an LLM judge breaking three-way splits).

ARC v2 was designed and benchmarked in my MSc thesis on privacy-preserving agentic QA
(HotpotQA and TriviaQA, local Llama 3.1 8B via Ollama — 61% / 82% accuracy).
Thesis repo: https://github.com/Sb2806/Agentic-LLM-Benchmarking

**Note:** the live demo runs the same framework on a hosted open-weight model through Groq, so the
thesis accuracy numbers do not directly apply to it.

## Run locally
```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key      # Windows: set GROQ_API_KEY=your_key
uvicorn app:app --port 8000
```
Open http://localhost:8000

## API
`POST /query` with `{"question": "...", "context": "optional document text"}`.
Leave `context` empty and ARC v2 falls back to web + Wikipedia search.

## Deployment
Free Render web service. Build: `pip install -r requirements.txt`.
Start: `uvicorn app:app --host 0.0.0.0 --port $PORT`. Secret: `GROQ_API_KEY`.

## Limits
30 questions per IP per hour, question up to 500 chars, context up to 20,000 chars.
