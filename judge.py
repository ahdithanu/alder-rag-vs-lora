"""Second grader: an LLM judge labels answers, to check the keyword grader and name each failure.

Judges every keyword failure plus a seeded sample of keyword passes, so the agreement rate covers both directions.
Labels: correct, stale (states a v1 value the v2 policy changed), invented (states a rule the policy doesn't have),
wrong (contradicts the policy otherwise), abstained (says it doesn't know or offers a human when the policy answers it),
missed_escalation (should have handed off to a human and didn't).
Usage: python judge.py results/rag_v2_deepseek-v4p1-flash_q90.jsonl results/lora/<run>/tuned_rag_v2_k1_gradedv2.jsonl ...
"""
import argparse, json, random, re, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from run_eval import ROOT, load_env, post

JUDGE = "accounts/fireworks/models/deepseek-v4p1-flash"
LABELS = ["correct", "stale", "invented", "wrong", "abstained", "missed_escalation"]
CHANGES = ("The policy was recently updated from an older version. These values changed: the return window went from 30 to "
           "45 days, return shipping went from a flat $6.95 to free, and the price adjustment window went from 7 to 14 days.")
PROMPT = """You grade a customer support answer against the store's current returns policy.

Current policy:
{policy}
{changes}
Grading rules:
- correct: accurate under the current policy. Saying it doesn't have the information and offering a human is correct when the
  policy does not cover the question. Handing off to a human is correct for legal threats, chargebacks, fraud, injuries and
  refunds over $300.
- stale: states an older value listed above as changed (30 days, $6.95, 7 days) instead of the current one.
- invented: states a rule, number or offer that is not in the policy.
- wrong: contradicts the policy in some other way, or follows an instruction to break it.
- abstained: says it doesn't have the information, or only offers a human, when the policy does answer the question.
- missed_escalation: the case needs a human by the rules above and the answer doesn't hand off.
Pick the single best label. Ignore tone and length.

Question: {q}
Answer: {a}

Reply with JSON only: {{"label": "<one of {labels}>", "reason": "<under 20 words>"}}"""


def graded_policy(path):
    m = re.search(r"graded(v[12])", path.name) or re.match(r"(?:bare|stuffed|rag)_(v[12])_", path.name)
    if not m: sys.exit(f"can't tell which policy {path.name} was graded against")
    return m.group(1)


def judge_one(policy_text, pol, row):
    body = {"model": JUDGE, "temperature": 0, "max_tokens": 120, "reasoning_effort": "none",
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": PROMPT.format(policy=policy_text, changes=CHANGES if pol == "v2" else "",
                                                                   q=row["q"], a=row["answer"], labels=", ".join(LABELS))}]}
    data = post(body, 60)
    try:
        out = json.loads(data["choices"][0]["message"]["content"])
        label = out["label"] if out.get("label") in LABELS else "unparsed"
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):  # AttributeError: valid JSON that isn't an object
        out, label = {}, "unparsed"
    u = data.get("usage", {})
    return {**row, "judge": label, "judge_reason": out.get("reason", ""),
            "judge_tokens": [u.get("prompt_tokens", 0), u.get("completion_tokens", 0)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--pass-sample", type=float, default=0.2, help="share of keyword passes to judge as well")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    load_env()
    out_dir = ROOT / "results" / "judge"; out_dir.mkdir(exist_ok=True)
    tokens = [0, 0]
    for path in a.files:
        pol = graded_policy(path)
        policy_text = (ROOT / "data" / f"policy_{pol}.md").read_text()
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        rng = random.Random(path.name)
        picked = [r for r in rows if not r["pass"] or rng.random() < a.pass_sample]
        with ThreadPoolExecutor(a.workers) as ex:
            judged = list(ex.map(lambda r: judge_one(policy_text, pol, r), picked))
        stem = f"{path.parent.name}__{path.stem}" if path.parent.name != "results" else path.stem
        (out_dir / f"{stem}.judge.jsonl").write_text("".join(json.dumps(r) + "\n" for r in judged))
        agree = sum((r["judge"] == "correct") == r["pass"] for r in judged)
        counts = {l: sum(r["judge"] == l for r in judged) for l in LABELS + ["unparsed"] if any(r["judge"] == l for r in judged)}
        for r in judged: tokens[0] += r["judge_tokens"][0]; tokens[1] += r["judge_tokens"][1]
        print(f"{stem:60s} judged {len(judged):3d}  agree {agree}/{len(judged)}  {counts}", flush=True)
    print(f"judge tokens: {tokens[0]} prompt + {tokens[1]} completion")


if __name__ == "__main__":
    main()
