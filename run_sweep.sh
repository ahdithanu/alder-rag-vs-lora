#!/usr/bin/env bash
# Old recipe (facts in the weights) vs the fix (trained with retrieved context), 3 shuffle seeds each, on the 90 questions.
# Serverless Training allows one run per base model at a time, so sessions run back to back. Base is sampled once.
# Usage: PY=<cookbook>/training/.venv/bin/python ./run_sweep.sh
set -u
cd "$(dirname "$0")"
PY=${PY:?set PY to the cookbook venv python}
COMMON=(--questions data/questions_large.jsonl --conditions rag:v2:1,rag:v2:3,bare --k 3 --no-promote)
STAMP=$(date +%Y%m%d-%H%M%S)

run() {  # run <run-id> <args...>
  local id=$1; shift
  local log=results/lora/$id.log
  local waits=0 fails=0  # counted separately: waiting for another run's slot is not a failure
  while :; do
    "$PY" lora_session.py --run-id "$id" "$@" "${COMMON[@]}" > "$log" 2>&1 && { echo "$id ok"; return 0; }
    if grep -q "maximum concurrent" "$log"; then
      waits=$((waits + 1))
      if [ "$waits" -ge 15 ]; then echo "$id: still blocked by another run after $waits waits, see $log"; return 1; fi
      echo "$id: another run is active, waiting"; rm -rf "results/lora/$id"; sleep 120
    else
      fails=$((fails + 1))
      if [ "$fails" -ge 2 ]; then echo "$id failed twice, see $log"; return 1; fi
      echo "$id failed, retrying once"; mv "$log" "$log.failed$fails"; rm -rf "results/lora/$id"; sleep 60
    fi
  done
}

run "q90-old-s0-$STAMP" --train data/sft_train_v1.clean.jsonl --seed 0
for s in 1 2; do run "q90-old-s$s-$STAMP" --train data/sft_train_v1.clean.jsonl --seed $s --skip-base; done
for s in 0 1 2; do run "q90-fix-s$s-$STAMP" --train data/sft_train_v1_context.clean.jsonl --seed $s --skip-base; done
