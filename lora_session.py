"""Base vs LoRA-tuned Muse Glimmer 30B on Fireworks Serverless Training, graded on the Alder & Co. questions
(31 by default, 90 with --questions data/questions_large.jsonl).

One session, nothing deployed. Muse Glimmer isn't on serverless inference, so every comparison is sampled inside the
training session (base adapter first, then the tuned one) with the same retrieval, prompts and grader as run_eval.py.
LoRA SFT on Muse Glimmer is free until 11/30 (docs.fireworks.ai/fine-tuning/models); in-session sampling is billed per token.

Run from the Fireworks cookbook's venv (github.com/fw-ai/cookbook, training/):
    <cookbook>/training/.venv/bin/python lora_session.py --run-id alder-v1-$(date +%Y%m%d-%H%M%S)
"""
import argparse, json, os, random, sys, time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

from run_eval import BASE, BM25, POLICY_RULES, ROOT, load_env, score, sections

BASE_MODEL = "accounts/fireworks/models/muse-glimmer-30b"
TOKENIZER_MODEL = "meta-models/Muse-Glimmer-30B"
PROMOTION_DEADLINE = datetime(2026, 12, 1, tzinfo=timezone(timedelta(hours=-8)))
LORA_ALPHA = 32
LORA_TARGETS = {"train_mlp": True, "train_attn": True, "train_unembed": True}
TAGS = ["ALL", "fact", "edge", "changed", "escalate", "silent", "adversarial"]

# (condition, policy in the prompt). "bare" has no policy, so its answers are graded against both v1 and v2.
BASE_CONDITIONS = [("bare", None), ("stuffed", "v1"), ("rag", "v1"), ("stuffed", "v2"), ("rag", "v2")]
TUNED_CONDITIONS = [("bare", None), ("rag", "v1"), ("rag", "v2"), ("stuffed", "v2")]


def parse_conditions(spec):
    """'rag:v2:1,bare' -> [("rag", "v2", 1), ("bare", None, None)]; a rag condition without k uses --k."""
    out = []
    for c in filter(None, spec.split(",")):
        parts = c.split(":")
        out.append((parts[0], parts[1] if len(parts) > 1 else None, int(parts[2]) if len(parts) > 2 else None))
    return out


def build_prompts(questions, ks):
    """System prompt and retrieved section ids per (condition, policy, k, question id), same as run_eval.py."""
    out = {}
    for pol in ("v1", "v2"):
        text = (ROOT / "data" / f"policy_{pol}.md").read_text()
        secs = sections(text); bm = BM25([t for _, t in secs])
        for q in questions:
            out[("stuffed", pol, None, q["id"])] = (BASE + POLICY_RULES + "\n\n" + text, [])
            sc = bm.scores(q["q"]); order = sorted(range(len(secs)), key=lambda i: -sc[i])
            for k in ks:
                out[("rag", pol, k, q["id"])] = (BASE + POLICY_RULES + "\n\n" + "\n\n".join(secs[i][1] for i in order[:k]),
                                                 [secs[i][0] for i in order[:k]])
    for q in questions:
        out[("bare", None, None, q["id"])] = (BASE, [])
    return out


def grade(q, answer, policy):
    must = q.get("must_v2", q["must"]) if policy == "v2" else q["must"]
    mnot = q.get("must_not_v2", q["must_not"]) if policy == "v2" else q["must_not"]
    return score(answer, must, mnot)


def tag_rates(rows):
    by = defaultdict(lambda: [0, 0])
    for r in rows:
        for t in r["tags"] + ["ALL"]:
            by[t][0] += r["pass"]; by[t][1] += 1
    return {t: f"{by[t][0]}/{by[t][1]}" for t in TAGS if t in by}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--train", default="data/sft_train_v1.clean.jsonl")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--reasoning-strength", default="low", help="Muse Glimmer system-prompt reasoning level, train and eval")
    ap.add_argument("--no-promote", action="store_true", help="Keep the adapter session-only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conditions", default=None, help="Sample only these for base and tuned, e.g. 'rag:v2:1,rag:v2:3,bare'")
    ap.add_argument("--questions", default="data/questions.jsonl", help="data/questions_large.jsonl for the 90-question set")
    ap.add_argument("--skip-base", action="store_true", help="Sample only the tuned adapter (base already sampled in another run)")
    a = ap.parse_args()
    if datetime.now(timezone.utc) >= PROMOTION_DEADLINE:
        sys.exit("The free Muse Glimmer SFT promotion has ended; training would be billed. Not launching.")
    load_env()

    from fireworks.training import sdk
    from fireworks.training.sdk import FireworksClient, FiretitanServiceClient
    from training.renderer import TrainOnWhat, get_text_content
    from training.renderer.muse_glimmer import MuseGlimmerOptions, MuseGlimmerRenderer
    from training.renderer.tokenizer import get_tokenizer
    from training.utils.serverless import enable_serverless_supervised_409_retry, serverless_base_url

    run_dir = ROOT / "results" / "lora" / a.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({**vars(a), "base_model": BASE_MODEL, "lora_alpha": LORA_ALPHA,
                                                     **LORA_TARGETS}, indent=2))
    questions = [json.loads(l) for l in open(ROOT / a.questions)]
    train = [json.loads(l) for l in open(ROOT / a.train) if l.strip()]
    base_conds = parse_conditions(a.conditions) if a.conditions else [(c, p, None) for c, p in BASE_CONDITIONS]
    tuned_conds = parse_conditions(a.conditions) if a.conditions else [(c, p, None) for c, p in TUNED_CONDITIONS]
    base_conds = [(c, p, (k or a.k) if c == "rag" else None) for c, p, k in base_conds]
    tuned_conds = [(c, p, (k or a.k) if c == "rag" else None) for c, p, k in tuned_conds]
    prompts = build_prompts(questions, {k for _, _, k in base_conds + tuned_conds if k})

    tokenizer = get_tokenizer(TOKENIZER_MODEL)
    # Pin the date and reasoning level so training and eval prompts render identically.
    renderer = MuseGlimmerRenderer(tokenizer, options=MuseGlimmerOptions(
        reasoning_strength=a.reasoning_strength, current_date=date.today().isoformat()))

    datums = []
    for ex in train:
        model_input, weights = renderer.build_supervised_example(ex["messages"], train_on_what=TrainOnWhat.LAST_ASSISTANT_TURN)
        tokens = [int(t) for t in model_input.to_ints()]; w = [float(x) for x in weights.tolist()]
        if len(tokens) > a.max_seq_len:
            sys.exit(f"a train row has {len(tokens)} tokens > --max-seq-len {a.max_seq_len}")
        datums.append(sdk.Datum(model_input=sdk.ModelInput.from_ints(tokens[:-1]),
                                loss_fn_inputs={"target_tokens": tokens[1:], "weights": w[1:]}))
    print(f"{len(datums)} train rows, {len(questions)} eval questions, run dir {run_dir}", flush=True)

    service = FiretitanServiceClient(api_key=os.environ["FIREWORKS_API_KEY"],
                                     base_url=serverless_base_url(os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")))
    enable_serverless_supervised_409_retry(service)
    log = (run_dir / "metrics.jsonl").open("a")
    usage = {"prompt_tokens": 0, "sampled_tokens": 0}
    params = sdk.SamplingParams(max_tokens=a.max_tokens, temperature=0.0, stop=renderer.get_stop_sequences())
    summary = {"run_id": a.run_id, "results": {}}
    t0 = time.time()
    try:
        tc = service.create_lora_training_client(base_model=BASE_MODEL, rank=a.lora_rank, alpha=LORA_ALPHA, **LORA_TARGETS)
        summary["session"] = getattr(service, "training_session_id", None)
        print(f"session={summary['session']}", flush=True)

        def evaluate(label, conditions):
            snapshot = tc.save_weights_for_sampler(f"alder-{label}").result().path
            sampler = service.create_sampling_client(model_path=snapshot, tokenizer=tokenizer)
            try:
                jobs = []
                for cond, pol, k in conditions:
                    for q in questions:
                        system, retrieved = prompts[(cond, pol, k, q["id"])]
                        pr = renderer.build_generation_prompt([{"role": "system", "content": system}, {"role": "user", "content": q["q"]}])
                        jobs.append((cond, pol, k, q, retrieved, pr, sampler.sample(prompt=pr, num_samples=1, sampling_params=params)))
                rows = defaultdict(list)
                for cond, pol, k, q, retrieved, pr, fut in jobs:
                    out = list((fut.result() if hasattr(fut, "result") else fut).sequences[0].tokens)
                    usage["prompt_tokens"] += len(pr.to_ints()); usage["sampled_tokens"] += len(out)
                    msg, term = renderer.parse_response(out)
                    answer = get_text_content(msg)
                    for gpol in ([pol] if pol else ["v1", "v2"]):
                        hit = bool(set(retrieved) & set(q["sections"])) if cond == "rag" and q["sections"] else None
                        ktag = f"_k{k}" if k and k != a.k else ""  # the original runs' names have no k
                        rows[f"{label}_{cond}_{pol or 'none'}{ktag}_graded{gpol}"].append({
                            "id": q["id"], "tags": q["tags"], "q": q["q"], "answer": answer, "pass": grade(q, answer, gpol),
                            "retrieved": retrieved, "retrieval_hit": hit, "truncated": not term.is_clean,
                            "completion_tokens": len(out)})
            finally:
                close = getattr(sampler, "close", None)
                if close: close()
            for name, rs in rows.items():
                (run_dir / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs))
                hits = [r["retrieval_hit"] for r in rs if r["retrieval_hit"] is not None]
                res = {"pass": tag_rates(rs), "fails": [r["id"] for r in rs if not r["pass"]],
                       "truncated": sum(r["truncated"] for r in rs),
                       **({"retrieval_hit": f"{sum(hits)}/{len(hits)}"} if hits else {})}
                summary["results"][name] = res
                log.write(json.dumps({"eval": name, **res}) + "\n"); log.flush()
                print(f"{name:34s} {res['pass']['ALL']:>6s}  changed {res['pass'].get('changed', '-')}  "
                      f"escalate {res['pass'].get('escalate', '-')}  silent {res['pass'].get('silent', '-')}  "
                      f"truncated {res['truncated']}", flush=True)

        if not a.skip_base: evaluate("base", base_conds)
        adam = sdk.AdamParams(learning_rate=a.learning_rate, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0, grad_clip_norm=1.0)
        rng, step = random.Random(a.seed), 0
        for epoch in range(a.epochs):
            order = list(range(len(datums))); rng.shuffle(order)
            for start in range(0, len(order), a.batch_size):
                fb = tc.forward_backward([datums[i] for i in order[start:start + a.batch_size]], "cross_entropy").result()
                opt = tc.optim_step(adam, emit_grad_norm_metrics=True).result()
                step += 1
                m = dict(getattr(fb, "metrics", None) or {})
                loss = m.get("loss:sum", 0.0) / max(float(m.get("response_tokens") or m.get("num_loss_tokens") or 1.0), 1.0)
                row = {"step": step, "epoch": epoch + 1, "loss": round(loss, 4),
                       **{k: v for k, v in (getattr(opt, "metrics", None) or {}).items() if "grad" in k}}
                log.write(json.dumps(row) + "\n"); log.flush(); print(json.dumps(row), flush=True)
        summary["steps"] = step
        evaluate("tuned", tuned_conds)

        if not a.no_promote:
            fw = FireworksClient(api_key=os.environ["FIREWORKS_API_KEY"], base_url=os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai"))
            ckpts = [r for r in fw.list_training_session_checkpoints(service.training_session_name)
                     if r.get("promotable") and "alder-tuned" in (r.get("checkpointName") or r.get("name") or "")]
            if ckpts:
                model_id = f"alder-lora-{a.run_id}"[:63].rstrip("-")
                fw.promote_session_checkpoint(name=max(ckpts, key=lambda r: r.get("createTime", ""))["name"],
                                              output_model_id=model_id, base_model=BASE_MODEL)
                summary["promoted_model_id"] = model_id
                print(f"promoted tuned adapter as model {model_id} (serving it needs a dedicated deployment)", flush=True)
    finally:
        summary["sampling_usage"] = usage
        summary["total_minutes"] = round((time.time() - t0) / 60, 1)
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        log.close(); service.close()
        print(f"session closed, nothing deployed. sampling used {usage['prompt_tokens']} prompt + "
              f"{usage['sampled_tokens']} sampled tokens. summary: {run_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
