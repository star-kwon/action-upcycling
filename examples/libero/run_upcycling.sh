#!/bin/bash
# Evaluate pi0.5 on the four LIBERO suites (4 x 10 tasks x 50 trials = 2,000 episodes) against a running policy server.
# Every episode is logged to results/<run_name>/<suite>.jsonl; finished suites are skipped, so the script can be re-run.
#
#   bash examples/libero/run_upcycling.sh <run_name> [client args ...]
#
#   bash examples/libero/run_upcycling.sh baseline                                  # default setting (h = 5)
#   bash examples/libero/run_upcycling.sh upcycling --args.upcycle                  # Action Upcycling, r = 1.5
#   bash examples/libero/run_upcycling.sh online --args.upcycle-online-ratio 1.5    # online pool
#   bash examples/libero/run_upcycling.sh fixed7 --args.replan-steps 7              # fixed-length execution
#
# Environment variables: PORT (default 8000), SUITES, TRIALS (default 50), RESULTS_DIR (default results).
set -eu
cd "$(dirname "$0")/../.."

RUN_NAME=${1:?usage: run_upcycling.sh <run_name> [client args ...]}
shift
PORT=${PORT:-8000}
SUITES=${SUITES:-"libero_spatial libero_object libero_goal libero_10"}
TRIALS=${TRIALS:-50}
OUT=${RESULTS_DIR:-results}/$RUN_NAME
mkdir -p "$OUT"
export PYTHONPATH=${PYTHONPATH:-}:$PWD/third_party/libero
export MUJOCO_GL=${MUJOCO_GL:-egl}

for SUITE in $SUITES; do
  LOG=$OUT/$SUITE.jsonl
  if [ -f "$LOG" ] && [ "$(wc -l < "$LOG")" -ge $((10 * TRIALS)) ]; then
    echo "skip $RUN_NAME/$SUITE (done)"
    continue
  fi
  rm -f "$LOG"
  echo "=== $RUN_NAME/$SUITE ==="
  python examples/libero/main.py \
    --args.task-suite-name "$SUITE" \
    --args.num-trials-per-task "$TRIALS" \
    --args.port "$PORT" \
    --args.log-path "$LOG" \
    --args.video-out-path "$OUT/videos/$SUITE" \
    "$@"
done

python examples/libero/summarize.py "$OUT"
