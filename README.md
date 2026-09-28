# ARC v2 — Triple-Agent Majority-Vote Question Answering

Live demo: https://arc-v2-qa.onrender.com

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

## Cost comparison: local/free inference vs. paid hosted APIs

ARC v2 answers each question with **3 independent agents**, then a voter picks the majority
answer. This gives higher reliability than a single-call approach, but it also means every
question costs roughly 3x the tokens of a normal chatbot call — a real tradeoff worth showing
explicitly rather than hiding.

**Method:** token counts below come directly from this project's own test runs (see the `tokens`
field returned by the API), not estimates. Two scenarios are shown because document-based
questions use far more tokens per call than short factual ones.

| Scenario | Tokens / question (3 agents) | Groq free tier | GPT-4o-mini-class API | GPT-4o-class API |
|---|---|---|---|---|
| Short factual Q&A, no document | ~780 | $0 | ~$0.29 / 1,000 q | ~$3.90 / 1,000 q |
| Q&A against an uploaded document | ~5,400 | $0 | ~$2.03 / 1,000 q | ~$27.00 / 1,000 q |

At 10,000 questions/month (a small internal tool's realistic volume):

| Scenario | Groq free tier | GPT-4o-mini-class API | GPT-4o-class API |
|---|---|---|---|
| Short factual Q&A | $0 | ~$2.93 | ~$39.00 |
| Document-based Q&A | $0 | ~$20.25 | ~$270.00 |

**Takeaway:** running ARC v2 on Groq's free tier costs nothing at this scale, but the multi-agent
design would cost noticeably more than a single-call system on a metered API — roughly 3x the
tokens of an equivalent single-shot pipeline. That's the real price of the accuracy gain measured
in the dissertation (ARC v2: 61% HotpotQA / 82% TriviaQA vs. weaker single-pass baselines). Whether
that tradeoff is worth it depends on how much the use case values reliability over raw cost — for
a customer-facing support tool answering thousands of questions a day, the frontier-tier column
could get expensive fast; for an internal tool where wrong answers are costly, the 3x cost is cheap
insurance.

*Note: hosted API prices are approximate blended rates as of late 2026 and change frequently —
check current provider pricing pages before relying on these figures for a real deployment decision.*
