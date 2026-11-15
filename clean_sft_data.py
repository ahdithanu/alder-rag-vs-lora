"""Strip policy-section citations ("(Policy #7)", "per Section 5", "Rule 12") from the teacher's answers.
The tuned model never sees the policy, so citations would teach it to quote sections the customer can't see.
Usage: python clean_sft_data.py data/sft_train_v1.jsonl  ->  data/sft_train_v1.clean.jsonl"""
import json, re, sys
from pathlib import Path

REF = r"(?:our\s+)?(?:returns\s+)?(?:policy|section|rule)\s*#?\s*\d+(?:\s+of\s+our\s+(?:returns\s+)?policy)?"
PATTERNS = [
    re.compile(r"\s*\(\s*(?:per|see|under)?\s*" + REF + r"\s*\)", re.I),                  # " (Policy #7)"
    re.compile(r"\b(?:per|under|according to|as per)\s+" + REF + r"\s*,?\s*", re.I),       # "Per Policy #12, "
    re.compile(r",?\s*(?:per|under|according to)\s+" + REF + r"(?=[.!?])", re.I),          # "..., per Section 5."
    re.compile(r"\s*\([^)]*\b(?:policy|policies|section|sections|rule|rules)\b[^)]*\d[^)]*\)", re.I),  # "(Policy #1 and #8)"
    re.compile(r"\b" + REF + r"\s+(?:notes|says|states|explains)\s+(?:that\s+)?", re.I),          # "Policy 2 notes refunds"
]

def clean(text):
    for p in PATTERNS: text = p.sub(" " if p is PATTERNS[1] else "", text)
    text = re.sub(r"  +", " ", text).replace(" ,", ",").replace(" .", ".").strip()
    return re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)

src = Path(sys.argv[1]); rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
changed = 0
for r in rows:
    a = r["messages"][-1]; new = clean(a["content"]); changed += new != a["content"]; a["content"] = new
cites = lambda r: re.search(r"\b(policy|section|rule)\s*#?\s*\d+", r["messages"][-1]["content"], re.I)
keep = [r for r in rows if not cites(r)]  # the few the patterns miss are dropped, not hand-edited
# Context rows (made with --context) must answer from the sections shown. A number that is in neither the context nor the
# question came from the teacher's memory of the full policy, which is the habit this data is meant to train out.
nums = lambda s: set(re.findall(r"\d+(?:\.\d+)?", s))
from_memory = lambda r: r["messages"][0]["content"] != r["messages"][0]["content"].split("\n\n")[0] and \
    nums(r["messages"][-1]["content"]) - nums(r["messages"][0]["content"] + " " + r["messages"][1]["content"])
memory = [r for r in keep if from_memory(r)]
keep = [r for r in keep if not from_memory(r)]
out = src.with_suffix(".clean.jsonl"); out.write_text("".join(json.dumps(r) + "\n" for r in keep))
print(f"{len(rows)} rows, {changed} answers changed, {len(rows) - len(keep) - len(memory)} still cited a section and "
      f"{len(memory)} used a number not in their context; both dropped -> {len(keep)} rows in {out}")
