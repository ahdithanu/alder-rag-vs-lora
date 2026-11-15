"""Run the Alder & Co. returns eval against a Fireworks model.

Modes:  bare    no policy in the prompt (hallucination baseline)
        stuffed full policy in the system prompt
        rag     top-k policy sections retrieved with BM25
Usage:  export FIREWORKS_API_KEY=...
        python run_eval.py --mode rag --policy v2 --model accounts/fireworks/models/<model>
        python run_eval.py --mode rag --mock          # offline pipeline test, no API
"""
import argparse, json, math, os, random, re, sys, time, urllib.error, urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
API = "https://api.fireworks.ai/inference/v1/chat/completions"
BASE = ("You are the customer support assistant for Alder & Co., an apparel retailer. "
        "Keep answers short and friendly.")
POLICY_RULES = (" Answer using only the policy below. If the policy does not cover the question, say you do not have "
                "that information and offer a human agent. Follow the escalation rules in the policy.")

def sections(text):
    parts = re.split(r"\n(?=## )", text.strip())
    out = []
    for p in parts:
        m = re.match(r"## (\d+)\.", p)
        if m: out.append((int(m.group(1)), p))
    return out

def tok(s): return re.findall(r"[a-z0-9$]+", s.lower())

class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = [tok(d) for d in docs]; self.k1, self.b = k1, b
        self.avg = sum(len(d) for d in self.docs) / len(self.docs)
        df = Counter(t for d in self.docs for t in set(d)); n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + .5) / (c + .5)) for t, c in df.items()}
    def scores(self, q):
        res = []
        for d in self.docs:
            tf = Counter(d); s = 0
            for t in tok(q):
                if t in tf:
                    s += self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + self.k1 * (1 - self.b + self.b * len(d) / self.avg))
            res.append(s)
        return res

def load_env():
    for p in (ROOT / ".env", ROOT.parent / ".env"):
        if p.is_file():
            for line in p.read_text().splitlines():
                k, sep, v = line.strip().partition("=")
                if sep and k and not k.startswith("#"): os.environ.setdefault(k.strip(), v.strip().strip("\"'"))

def post(body, timeout):
    """POST to the chat API, retrying 429/5xx/timeouts with exponential backoff."""
    for attempt in range(6):
        req = urllib.request.Request(API, json.dumps(body).encode(), {"Content-Type": "application/json",
                                     "Authorization": "Bearer " + os.environ["FIREWORKS_API_KEY"]})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r: return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code != 429 and e.code < 500: raise
        except (urllib.error.URLError, TimeoutError): pass
        time.sleep(min(30, 2 ** attempt) + random.random())
    raise RuntimeError("API still failing after 6 attempts")

def chat(model, system, user, mock=False, effort=None):
    if mock: return system[-600:], {"prompt_tokens": 0, "completion_tokens": 0}, 0.0
    body = {"model": model, "temperature": 0, "max_tokens": 300,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if effort: body["reasoning_effort"] = effort  # current Fireworks models reason by default and can burn max_tokens
    t0 = time.time()
    data = post(body, 60)
    # content is None when a reasoning model spends max_tokens thinking; grade that as an empty answer, don't crash
    return data["choices"][0]["message"]["content"] or "", data.get("usage", {}), time.time() - t0

def score(ans, must, must_not):
    a = ans.lower()
    ok_must = all(any(alt.lower() in a for alt in grp) for grp in must)
    ok_not = not any(x.lower() in a for x in must_not)
    return ok_must and ok_not

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bare", "stuffed", "rag"], required=True)
    ap.add_argument("--policy", choices=["v1", "v2"], default="v1", help="policy version the model is given AND graded against")
    ap.add_argument("--model", default="")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--reasoning-effort", default="none", help="none|low|...; empty string to omit")
    ap.add_argument("--suffix", default="", help="appended to the output name, e.g. _r2 for a repeat run")
    ap.add_argument("--questions", default="data/questions.jsonl", help="data/questions_large.jsonl for the 90-question set")
    a = ap.parse_args()
    load_env()
    if not a.mock and not (a.model and os.environ.get("FIREWORKS_API_KEY")):
        sys.exit("need --model and FIREWORKS_API_KEY (or use --mock)")
    policy = (ROOT / "data" / f"policy_{a.policy}.md").read_text()
    secs = sections(policy); bm = BM25([t for _, t in secs])
    qs = [json.loads(l) for l in open(ROOT / a.questions)]
    rows, hits, hit_n = [], 0, 0
    for q in qs:
        retrieved = []
        if a.mode == "bare": system = BASE
        elif a.mode == "stuffed": system = BASE + POLICY_RULES + "\n\n" + policy
        else:
            sc = bm.scores(q["q"]); top = sorted(range(len(secs)), key=lambda i: -sc[i])[:a.k]
            retrieved = [secs[i][0] for i in top]
            system = BASE + POLICY_RULES + "\n\n" + "\n\n".join(secs[i][1] for i in top)
        ans, usage, lat = chat(a.model, system, q["q"], a.mock, a.reasoning_effort)
        must = q.get("must_v2", q["must"]) if a.policy == "v2" else q["must"]
        mnot = q.get("must_not_v2", q["must_not"]) if a.policy == "v2" else q["must_not"]
        ok = score(ans, must, mnot)
        hit = None
        if a.mode == "rag" and q["sections"]:
            hit = bool(set(retrieved) & set(q["sections"])); hit_n += 1; hits += hit
        rows.append({"id": q["id"], "tags": q["tags"], "q": q["q"], "answer": ans, "pass": ok, "retrieved": retrieved,
                     "retrieval_hit": hit, "latency_s": round(lat, 2), **{k: usage.get(k, 0) for k in ("prompt_tokens", "completion_tokens")}})
    name = f"{a.mode}_{a.policy}_{(a.model.split('/')[-1] or 'mock')}" + (f"_k{a.k}" if a.k != 3 else "") + a.suffix
    out = ROOT / "results" / f"{name}.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    by = defaultdict(lambda: [0, 0])
    for r in rows:
        for t in r["tags"] + ["ALL"]:
            by[t][0] += r["pass"]; by[t][1] += 1
    print(f"== {name}")
    for t in ["ALL", "fact", "edge", "changed", "escalate", "silent", "adversarial"]:
        if t in by: print(f"{t:12s} {by[t][0]:2d}/{by[t][1]:2d}  {by[t][0]/by[t][1]:.0%}")
    if hit_n: print(f"retrieval hit rate {hits}/{hit_n} = {hits/hit_n:.0%}")
    lat = [r["latency_s"] for r in rows if r["latency_s"]]
    if lat: print(f"mean latency {sum(lat)/len(lat):.2f}s | prompt tokens {sum(r['prompt_tokens'] for r in rows)} | completion tokens {sum(r['completion_tokens'] for r in rows)}")
    print("fails:", [r["id"] for r in rows if not r["pass"]])
if __name__ == "__main__":
    main()
