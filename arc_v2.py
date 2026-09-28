"""
ARC v2 — Triple Agent Majority Vote Framework
Extracted from Sramana's dissertation notebook (my_dissertation.ipynb, cell 22),
with all supporting functions pulled from cells 1, 3, 5, 6, 10, 18.

Usage:
    from arc_v2 import answer_question
    result = answer_question("Who wrote Romeo and Juliet?", context="")
    print(result["answer"])
"""

import os, time, re, math, json, requests

# ============================================================
# LLM SETUP (from cell 3)
# ============================================================

class GroqLLM:
    """
    Drop-in replacement for the old Ollama-based LocalLLM.
    Same .invoke(prompt, system, max_tokens, temperature) interface,
    so nothing else in arc_v2.py needs to change.

    Requires GROQ_API_KEY to be set as an environment variable
    (never hardcode it in this file).
    """
    def __init__(self, model="openai/gpt-oss-20b"):
        self.model = model
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.api_key = os.environ.get("GROQ_API_KEY", "")
        if not self.api_key:
            print("   WARNING: GROQ_API_KEY environment variable is not set.")

    def invoke(self, prompt, system="", max_tokens=800, temperature=0):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            # gpt-oss is a reasoning model: thinking tokens count toward the limit,
            # so tiny limits (e.g. 15) can return empty text. Keep a safe floor.
            "max_tokens": max(max_tokens, 300),
            "temperature": temperature,
            "reasoning_effort": "low",
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        for attempt in range(4):
            try:
                r = requests.post(self.api_url, json=payload, headers=headers, timeout=60)
                r.raise_for_status()
                result = r.json()["choices"][0]["message"]["content"]
                if result:
                    return result
                time.sleep(3)
            except Exception as e:
                print(f"   LLM retry {attempt+1}/4: {str(e)[:80]}")
                time.sleep(3)
        return "ERROR: LLM failed"


llm = GroqLLM(model="openai/gpt-oss-20b")

# ============================================================
# METRICS + SCORING (from cell 5)
# ============================================================

def estimate_tokens(text):
    # Lightweight fallback — original used tiktoken; not needed for API use
    return len(str(text).split())


class MetricsTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.steps = 0
        self.tool_calls = 0
        self.tokens = 0
        self.start_time = time.time()
        self.tool_usage = {}

    def record_tool_call(self, name, output=""):
        self.tool_calls += 1
        self.tokens += estimate_tokens(output)
        self.tool_usage[name] = self.tool_usage.get(name, 0) + 1

    def record_llm_call(self, text):
        self.steps += 1
        self.tokens += estimate_tokens(text)

    def elapsed(self):
        return round(time.time() - self.start_time, 2)

    def to_dict(self):
        return {"steps": self.steps, "tool_calls": self.tool_calls,
                "tokens": self.tokens, "time_sec": self.elapsed(),
                "tool_usage": str(self.tool_usage)}


def normalise(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    for w in ["the", "a", "an", "is", "was", "are", "were"]:
        text = re.sub(rf"\b{w}\b", "", text)
    return text.strip()


def score_answer(predicted, ground_truth):
    """Fuzzy matcher — used internally by the voter to compare agent answers."""
    if not predicted or not ground_truth:
        return 0.0
    p = normalise(str(predicted))
    g = normalise(str(ground_truth))
    if not p or not g:
        return 0.0
    if p == g:
        return 1.0
    if g in p or p in g:
        return 1.0
    if _fuzz_ratio(p, g) >= 0.82:
        return 1.0
    return 0.0


def _fuzz_ratio(a, b):
    """Simple token-sort ratio fallback (avoids requiring rapidfuzz)."""
    try:
        from rapidfuzz import fuzz
        return fuzz.token_sort_ratio(a, b) / 100
    except ImportError:
        # crude fallback using difflib
        import difflib
        return difflib.SequenceMatcher(None, a, b).ratio()


class _Fuzz:
    """Shim so existing code calling fuzz.token_sort_ratio(...) still works."""
    @staticmethod
    def token_sort_ratio(a, b):
        return _fuzz_ratio(a, b) * 100


fuzz = _Fuzz()


def extract_final_answer(text):
    text = str(text).strip()
    patterns = [r'Final Answer:\s*(.+?)(?:\n|$)', r'FINAL ANSWER:\s*(.+?)(?:\n|$)',
                r'Answer:\s*(.+?)(?:\n|$)', r'CONCLUSION:\s*(.+?)(?:\n|$)']
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            ans = m.group(1).strip()
            if ans:
                return ans
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    result = lines[-1] if lines else text.strip()
    for prefix in ["confident:", "uncertain:", "therefore,", "so,"]:
        if result.lower().startswith(prefix):
            result = result[len(prefix):].strip()
    return result

# ============================================================
# TOOLS (from cell 6) — web search / wikipedia / calculator
# ============================================================

CURRENT_CONTEXT = ""  # set by answer_question() before each call


def _sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def context_search(query, top_k=8):
    if not CURRENT_CONTEXT:
        return ""
    if len(CURRENT_CONTEXT) <= 4000:
        return CURRENT_CONTEXT
    qw = set(re.findall(r'\w+', query.lower()))
    scored = []
    for s in _sentences(CURRENT_CONTEXT):
        overlap = len(qw & set(re.findall(r'\w+', s.lower())))
        if overlap:
            scored.append((overlap, s))
    scored.sort(reverse=True)
    return "  ".join(s for _, s in scored[:top_k])


def _real_web_search(query):
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return "No search results found"
        return "\n\n".join([f"{r['title']}: {r['body'][:200]}" for r in results[:3]])[:2000]
    except Exception as e:
        return f"Search error: {str(e)[:100]}"


def _real_wiki(topic):
    try:
        import wikipediaapi
        wiki = wikipediaapi.Wikipedia(language="en", user_agent="ARCv2App/1.0")
        page = wiki.page(topic)
        if page.exists():
            return page.summary[:1500]
        page2 = wiki.page(" ".join(topic.split()[:3]))
        if page2.exists():
            return page2.summary[:1500]
        return f"Page not found for '{topic}'"
    except Exception as e:
        return f"Wikipedia error: {str(e)[:100]}"


def web_search_func(query):
    hit = context_search(query)
    return hit if len(hit) > 40 else _real_web_search(query)


def wikipedia_func(topic):
    hit = context_search(topic)
    return hit if len(hit) > 40 else _real_wiki(topic)


def calculator_func(expression):
    try:
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        allowed.update({"abs": abs, "round": round, "int": int, "float": float})
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(round(float(result), 6))
    except Exception as e:
        return f"Calculator error: {str(e)[:100]}"

# ============================================================
# AGENT 2 — ReWOO (from cell 10)
# ============================================================

REWOO_PLANNER = """You are a ReWOO planner.
Plan ALL tool calls upfront in ONE pass.
Use variable substitution to chain results.

Available tools: web_search, wikipedia, calculator

Output EXACTLY this format:
Plan: [one sentence description of approach]
#E1 = web_search["specific query here"]
#E2 = wikipedia["specific topic here"]
#E3 = calculator["math expression or #E1 value"]

Rules:
- Use #E1, #E2, #E3 as variable names
- Reference earlier results: web_search["#E1 result"]
- Minimum 2 tool calls, maximum 4
- Use specific queries in quotes
- No extra text after the plan"""

REWOO_SOLVER = """Answer using ONLY the evidence below.

Question: {question}

Evidence collected:
{evidence}

Final Answer: [give a direct short answer]"""


def run_rewoo(task, tracker=None, max_tokens=800, temperature=0):
    if tracker is None:
        tracker = MetricsTracker()
    tracker.reset()

    plan_response = llm.invoke(f"Question: {task}\n\nCreate a ReWOO plan:",
                                system=REWOO_PLANNER, max_tokens=250, temperature=temperature)
    tracker.record_llm_call(plan_response)

    evidence = {}
    tool_calls = re.findall(r'(#E\d+)\s*=\s*(\w+)\s*\["?([^"\]]+)"?\]', plan_response)

    if not tool_calls:
        result = web_search_func(task[:100])
        tracker.record_tool_call("web_search", result)
        evidence["#E1"] = result
    else:
        for var, tool_name, query in tool_calls[:4]:
            for ev, ev_val in evidence.items():
                query = query.replace(ev, ev_val[:100])
            tn = tool_name.lower()
            if "calculator" in tn:
                result, tool_used = calculator_func(query), "calculator"
            elif "wikipedia" in tn:
                result, tool_used = wikipedia_func(query), "wikipedia"
            else:
                result, tool_used = web_search_func(query), "web_search"
            tracker.record_tool_call(tool_used, result)
            evidence[var] = result[:500]

    evidence_text = "\n".join([f"{v}: {val[:300]}" for v, val in evidence.items()])
    solver_response = llm.invoke(
        REWOO_SOLVER.format(question=task, evidence=evidence_text[:1000]),
        max_tokens=150, temperature=temperature)
    tracker.record_llm_call(solver_response)

    return {"answer": extract_final_answer(solver_response), "trace": evidence_text[:500], **tracker.to_dict()}

# ============================================================
# AGENT 1 — Context Reader (from cell 18)
# ============================================================

STEP2_SYSTEM = """You are a precise information extractor.
Read the question and context carefully.
Extract ONLY the specific piece of information asked.
Give a short direct answer — maximum 8 words.
If the answer is not in the context say: NOT IN CONTEXT"""

STEP3_SYSTEM_V2 = """You are a precise answer extractor.
Extract ONLY the specific answer to the question.
Rules:
- Year question: reply with ONLY the year e.g. 1755
- Person question: reply with ONLY the name
- Place question: reply with ONLY the place name
- Yes/no question: reply with ONLY yes or no
- Number question: reply with ONLY the number
- Maximum 6 words. No sentences. No explanations."""


def select_relevant_paragraphs(question, context, tracker=None):
    paragraphs = context.split('\n\n')
    if not paragraphs:
        return context[:1500], ''
    stop_words = {'the', 'a', 'an', 'is', 'was', 'are', 'were', 'in',
                  'of', 'and', 'or', 'to', 'at', 'by', 'for', 'with',
                  'what', 'which', 'who', 'where', 'when', 'how', 'did',
                  'does', 'do', 'both', 'between', 'from', 'that', 'this',
                  'have', 'had', 'has', 'be', 'been', 'first', 'last',
                  'also', 'only', 'just'}
    q_words = set(re.findall(r'\w+', question.lower())) - stop_words
    scored = []
    for para in paragraphs:
        if not para.strip():
            continue
        para_lower = para.lower()
        para_words = set(re.findall(r'\w+', para_lower))
        overlap = len(q_words & para_words)
        specific = sum(1 for w in q_words if w in para_lower and len(w) > 5)
        scored.append((overlap + specific * 2, para.strip()))
    scored.sort(reverse=True)
    para1 = scored[0][1][:800] if len(scored) > 0 else ''
    para2 = scored[1][1][:800] if len(scored) > 1 else ''
    return para1, para2


def extract_hop1(question, para1, para2, tracker):
    combined = f"{para1}\n\n{para2}"
    q = question.lower()
    if any(w in q for w in ['where did', 'which university', 'which school',
                             'which college', 'which company', 'which team']):
        hint = 'Find the specific institution or organization name.'
    elif any(w in q for w in ['who was', 'who is', 'who did', 'who directed',
                               'who wrote', 'who starred', 'which actor', 'which actress']):
        hint = 'Find the specific person name.'
    elif any(w in q for w in ['what year', 'when was', 'when did']):
        hint = 'Find the specific year or time period.'
    elif any(w in q for w in ['what country', 'which country', 'what city',
                               'which city', 'where is', 'where was']):
        hint = 'Find the specific country or city name.'
    else:
        hint = 'Find the specific fact or entity needed.'
    hop1 = llm.invoke(
        f"Question: {question}\n\nContext:\n{combined[:1200]}\n\n"
        f"Task: {hint}\nExtract in 1-5 words only.\n"
        f"If not found say: NOT IN CONTEXT",
        system=STEP2_SYSTEM, max_tokens=20, temperature=0)
    tracker.record_llm_call(hop1)
    tracker.record_tool_call('context', combined)
    return hop1.strip()


def extract_final(question, hop1_answer, para1, para2, tracker):
    combined = f"{para1}\n\n{para2}"
    q = question.lower()
    if q.startswith(('did ', 'does ', 'do ', 'were both', 'is ', 'are both',
                      'are ', 'have ', 'has ', 'was ')):
        instruction = 'Reply only yes or no.'
    elif any(w in q for w in ['what year', 'in what year', 'when was', 'when did']):
        instruction = 'Extract only the year as a number.'
    elif any(w in q for w in ['who was', 'who is', 'who did', 'who directed',
                               'who wrote', 'which actor', 'which actress',
                               'who created', 'who hosted']):
        instruction = 'Extract only the person name.'
    elif any(w in q for w in ['what country', 'which country', 'what city',
                               'which city', 'where is', 'where was']):
        instruction = 'Extract only the place name.'
    elif any(w in q for w in ['how many', 'how much', 'how far', 'what number', 'what channel']):
        instruction = 'Extract only the number.'
    elif any(w in q for w in ['what type', 'what kind', 'what class', 'what genre',
                               'what heritage', 'what nationality', 'what language']):
        instruction = 'Extract only the type or category.'
    else:
        instruction = 'Extract only the specific answer. Maximum 6 words.'
    final = llm.invoke(
        f"Question: {question}\n\nKey finding: {hop1_answer}\n\n"
        f"Context: {combined[:800]}\n\n{instruction}\nAnswer:",
        system=STEP3_SYSTEM_V2, max_tokens=15, temperature=0)
    tracker.record_llm_call(final)
    return final.strip()


def run_context_reader(question, context, tracker=None):
    if tracker is None:
        tracker = MetricsTracker()
    tracker.reset()
    if not context or len(context) < 50:
        return {'answer': 'NOT FOUND', 'hop1': '', 'source': 'NONE', **tracker.to_dict()}
    para1, para2 = select_relevant_paragraphs(question, context, tracker)
    hop1 = extract_hop1(question, para1, para2, tracker)
    if 'NOT IN CONTEXT' in hop1.upper():
        direct = llm.invoke(
            f'Question: {question}\n\nContext:\n{context[:2000]}\n\n'
            f'Answer the question directly from the context above.\n'
            f'Give only the specific answer in 1-6 words.\n'
            f'If not found say: NOT FOUND',
            max_tokens=15, temperature=0)
        tracker.record_llm_call(direct)
        ans = direct.strip()
        if 'NOT FOUND' in ans.upper() or len(ans) < 2:
            alt = llm.invoke(
                f'Question: {question}\n\nContext:\n{context[2000:4000]}\n\nAnswer in 1-6 words only:',
                max_tokens=15, temperature=0)
            tracker.record_llm_call(alt)
            ans = alt.strip()
        return {'answer': ans, 'hop1': 'DIRECT', 'source': 'CONTEXT_DIRECT', **tracker.to_dict()}
    final = extract_final(question, hop1, para1, para2, tracker)
    return {'answer': final, 'hop1': hop1, 'source': 'CONTEXT', **tracker.to_dict()}


def run_context_reader_full(question, context, tracker):
    if not context or len(context) < 50:
        return {'answer': 'NOT FOUND', **tracker.to_dict()}
    q = question.lower()
    if q.startswith(('did ', 'does ', 'do ', 'were both', 'is ', 'are both',
                      'are ', 'have ', 'has ', 'was ')):
        instruction = 'Reply only yes or no.'
    elif any(w in q for w in ['what year', 'in what year', 'when was', 'when did']):
        instruction = 'Extract only the year as a number.'
    elif any(w in q for w in ['who was', 'who is', 'who did', 'who directed',
                               'who wrote', 'which actor', 'which actress',
                               'who created', 'who hosted']):
        instruction = 'Extract only the person name.'
    elif any(w in q for w in ['what country', 'which country', 'what city',
                               'which city', 'where is', 'where was']):
        instruction = 'Extract only the place name.'
    elif any(w in q for w in ['how many', 'how much', 'how far', 'what number', 'what channel']):
        instruction = 'Extract only the number.'
    else:
        instruction = 'Extract only the specific answer. Maximum 6 words.'
    ans = llm.invoke(
        f'Question: {question}\n\nContext:\n{context[:3000]}\n\n{instruction}\nAnswer:',
        max_tokens=15, temperature=0)
    tracker.record_llm_call(ans)
    tracker.record_tool_call('context_full', context[:3000])
    return {'answer': ans.strip(), **tracker.to_dict()}


def quality_check(answer, question, tracker):
    a = str(answer).strip()
    if not a or a in ('NOT FOUND', 'ERROR: LLM failed', ''):
        return None, 'EMPTY'
    if len(a.split()) > 12:
        short = llm.invoke(
            f'Question: {question}\nAnswer: {a}\nExtract the specific answer in 1-5 words only:',
            max_tokens=10, temperature=0)
        tracker.record_llm_call(short)
        a = short.strip()
    uncertain = ['i don', 'unfortunately', 'i cannot', 'there is no', 'no evidence',
                 'not provided', 'i do not have', 'no information']
    if any(u in a.lower() for u in uncertain):
        return None, 'UNCERTAIN'
    q = question.lower()
    if not q.startswith(('did ', 'does ', 'do ', 'is ', 'are ', 'were ', 'was ', 'have ', 'has ')):
        negations = ['is not', 'was not', 'are not', 'were not', 'did not', 'does not', 'not in']
        for neg in negations:
            if neg in a.lower():
                return None, 'NEGATED'
    return a, 'GOOD'

# ============================================================
# ARC v2 — Triple Agent Majority Vote (from cell 22 — final version)
# ============================================================

def run_arc_v2(question, context, tracker=None):
    if tracker is None:
        tracker = MetricsTracker()
    tracker.reset()

    has_context = context and len(context) > 50

    if has_context:
        # ── document-provided mode ──
        result1 = run_context_reader(question, context, tracker)
        a1 = result1.get('answer', '')
        v1, _ = quality_check(a1, question, tracker)

        rewoo_result = run_rewoo(question, tracker)
        a2_raw = rewoo_result['answer'] if isinstance(rewoo_result, dict) else rewoo_result[0]
        v2, _ = quality_check(a2_raw, question, tracker)

        result3 = run_context_reader_full(question, context, tracker)
        a3 = result3.get('answer', '')
        v3, _ = quality_check(a3, question, tracker)

    else:
        # ── no-context mode — 3 independent web/wiki searches ──
        web1 = web_search_func(question)
        tracker.record_tool_call('web_search', web1)
        ans1 = llm.invoke(
            f'Question: {question}\n\nEvidence:\n{web1[:800]}\n\nAnswer in 1-6 words only:',
            max_tokens=15, temperature=0)
        tracker.record_llm_call(ans1)
        a1 = ans1.strip()
        v1, _ = quality_check(a1, question, tracker)

        short_q = ' '.join(question.split()[:8])
        web2 = web_search_func(short_q)
        tracker.record_tool_call('web_search', web2)
        ans2 = llm.invoke(
            f'Question: {question}\n\nEvidence:\n{web2[:800]}\n\nAnswer in 1-6 words only:',
            max_tokens=15, temperature=0)
        tracker.record_llm_call(ans2)
        a2_raw = ans2.strip()
        v2, _ = quality_check(a2_raw, question, tracker)

        key_words = [w for w in question.split() if len(w) > 4 and w[0].isupper()]
        wiki_q = ' '.join(key_words[:3]) if key_words else short_q
        web3 = wikipedia_func(wiki_q)
        tracker.record_tool_call('wikipedia', web3)
        ans3 = llm.invoke(
            f'Question: {question}\n\nEvidence:\n{web3[:800]}\n\nAnswer in 1-6 words only:',
            max_tokens=15, temperature=0)
        tracker.record_llm_call(ans3)
        a3 = ans3.strip()
        v3, _ = quality_check(a3, question, tracker)

    # ── Agent 4 — Majority Vote ──
    valid = [(v1, 'A1'), (v2, 'A2'), (v3, 'A3')]
    valid = [(a, s) for a, s in valid if a]

    if not valid:
        return {
            'answer': 'NOT FOUND', 'confidence': 'NONE', 'votes': '0/3',
            'a1': a1, 'a2': a2_raw, 'a3': a3, **tracker.to_dict()
        }

    best_answer = None
    best_count = 0

    for i, (a_i, s_i) in enumerate(valid):
        count = 0
        for j, (a_j, s_j) in enumerate(valid):
            if (score_answer(a_i, a_j) >= 1.0 or
                    fuzz.token_sort_ratio(normalise(a_i), normalise(a_j)) >= 80):
                count += 1
        if count > best_count:
            best_count = count
            best_answer = a_i

    if best_count >= 3:
        confidence = 'HIGH'
        votes = '3/3'
    elif best_count == 2:
        confidence = 'MEDIUM'
        votes = '2/3'
    else:
        confidence = 'LOW'
        votes = '1/3'
        candidates = [a for a in [v1, v2, v3] if a]
        if len(candidates) >= 2:
            judge = llm.invoke(
                f'Question: {question}\n\n'
                f'Answer A: {candidates[0]}\n'
                f'Answer B: {candidates[1]}\n'
                + (f'Answer C: {candidates[2]}\n' if len(candidates) > 2 else '')
                + '\nWhich answer is most likely correct?\nReply with only: A or B or C',
                max_tokens=3, temperature=0)
            if 'C' in judge.upper() and len(candidates) > 2:
                best_answer = candidates[2]
            elif 'B' in judge.upper():
                best_answer = candidates[1]
            else:
                best_answer = candidates[0]
        else:
            best_answer = v1 or v2 or v3

    return {
        'answer': best_answer, 'confidence': confidence, 'votes': votes,
        'a1': v1 or a1, 'a2': v2 or a2_raw, 'a3': v3 or a3, **tracker.to_dict()
    }

# ============================================================
# PUBLIC ENTRY POINT — this is what your FastAPI endpoint calls
# ============================================================

def answer_question(question: str, context: str = "") -> dict:
    """
    Run ARC v2 on a question, optionally with a document/context string.
    If context is empty, ARC v2 falls back to live web search + Wikipedia.
    """
    global CURRENT_CONTEXT
    CURRENT_CONTEXT = context or ""

    tracker = MetricsTracker()
    result = run_arc_v2(question, context, tracker)
    return result


if __name__ == "__main__":
    # Quick manual test — requires Ollama running locally with llama3.1:8b pulled
    r = answer_question("Who wrote Romeo and Juliet?", context="")
    print(json.dumps(r, indent=2))
