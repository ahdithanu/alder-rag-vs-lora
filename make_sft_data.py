"""Build a synthetic SFT set for the LoRA step. A teacher model writes customer questions per policy section, then
answers them in the target style. Training prompts have NO policy in context, so facts get baked into the weights.
With --context, each prompt instead carries BM25-retrieved sections (the same format as run_eval.py's rag mode), about 1 in 5
is a forced retrieval miss, and the teacher answers from only the sections shown. That teaches reading and abstaining, not facts.
Usage: export FIREWORKS_API_KEY=...; python make_sft_data.py --teacher accounts/fireworks/models/<bigger-model> --per-section 12
Check Fireworks' docs for the exact dataset format before uploading (this writes chat-style {"messages": [...]})."""
import argparse, json, random, re

from run_eval import BASE, BM25, POLICY_RULES, ROOT, load_env, post, sections

STYLE = ("Answer in 1 to 3 sentences, friendly, state the rule and the number if there is one. If the situation involves a legal "
         "threat, chargeback, fraud, injury, or a refund over $300, say a human agent will take over and do not promise an outcome. "
         "If the policy does not cover it, say you do not have that information and offer a human agent.")
CONTEXT_STYLE = (" Use only the policy sections shown. If they do not contain the answer, say you do not have that information and "
                 "offer a human agent, even if you think you know the answer. Never mention section numbers.")

EFFORT = "none"

def call(model, prompt):
    body = {"model": model, "temperature": 0.7, "max_tokens": 900, "messages": [{"role": "user", "content": prompt}]}
    if EFFORT: body["reasoning_effort"] = EFFORT
    return post(body, 90)["choices"][0]["message"]["content"] or ""

def words(s): return set(re.findall(r"[a-z0-9]+", s.lower()))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--teacher", required=True); ap.add_argument("--per-section", type=int, default=12)
    ap.add_argument("--policy", default="v1"); ap.add_argument("--reasoning-effort", default="none")
    ap.add_argument("--context", action="store_true", help="retrieved sections in each prompt; answers use only those")
    ap.add_argument("--miss-rate", type=float, default=0.2, help="with --context, share of prompts given only wrong sections")
    ap.add_argument("--questions", default="data/questions_large.jsonl", help="eval questions to keep out of training")
    a = ap.parse_args()
    global EFFORT; EFFORT = a.reasoning_effort; load_env()
    policy = (ROOT / "data" / f"policy_{a.policy}.md").read_text()
    evalq = [json.loads(l)["q"] for l in open(ROOT / a.questions)]
    secs = sections(policy); bm = BM25([t for _, t in secs]); rng = random.Random(0); misses = 0
    topics = [f"section {i} of the policy" for i in range(1, 14)] + ["something the policy does NOT cover (price matching, discounts, delivery speed, anything else)"]
    out = []
    for n, t in enumerate(topics, 1):
        raw = call(a.teacher, f"Policy:\n{policy}\n\nWrite {a.per_section} different realistic customer messages about {t}. "
                              f"Vary tone and detail. One per line, no numbering, no quotes.")
        for q in [x.strip("-• ").strip() for x in raw.splitlines() if len(x.strip()) > 15][:a.per_section]:
            if any(len(words(q) & words(e)) / max(1, len(words(q) | words(e))) > 0.6 for e in evalq): continue  # keep eval clean
            if not a.context:
                ans = call(a.teacher, f"Policy:\n{policy}\n\n{STYLE}\n\nCustomer: {q}\nAssistant:").strip()
                out.append({"messages": [{"role": "system", "content": BASE}, {"role": "user", "content": q}, {"role": "assistant", "content": ans}]})
                continue
            k = rng.choice([1, 3])
            if n <= len(secs) and rng.random() < a.miss_rate:  # forced miss: k sections, none of them the one asked about
                shown = rng.sample([i for i in range(len(secs)) if secs[i][0] != n], k); misses += 1
            else:
                sc = bm.scores(q); shown = sorted(range(len(secs)), key=lambda i: -sc[i])[:k]
            context = "\n\n".join(secs[i][1] for i in shown)
            ans = call(a.teacher, f"Policy sections:\n{context}\n\n{STYLE}{CONTEXT_STYLE}\n\nCustomer: {q}\nAssistant:").strip()
            out.append({"messages": [{"role": "system", "content": BASE + POLICY_RULES + "\n\n" + context},
                                     {"role": "user", "content": q}, {"role": "assistant", "content": ans}]})
    random.Random(0).shuffle(out)
    name = f"sft_train_{a.policy}" + ("_context" if a.context else "")
    (ROOT / "data" / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
    print(len(out), "examples written to", name, *(["with", misses, "forced misses"] if a.context else []))

if __name__ == "__main__":
    main()
