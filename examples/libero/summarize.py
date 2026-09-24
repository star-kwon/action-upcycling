"""Summarize LIBERO runs logged by main.py (--args.log-path).

Reports the success rate, policy calls per episode, latency per policy call (ms, measured on the client and
including communication; the first call of each log file is excluded as warm-up), inference time per episode
(s, calls per episode x latency per call), and the mean execution length h_bar. The first run is the reference
for the call reduction.

    python examples/libero/summarize.py results/baseline results/upcycling
"""

import json
import pathlib
import sys

import numpy as np

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def load(run_dir):
    episodes, latencies = [], []
    for path in sorted(pathlib.Path(run_dir).glob("*.jsonl")):
        with open(path) as f:
            records = [json.loads(line) for line in f]
        episodes += records
        latencies += [c["ms"] for r in records for c in r["calls"]][1:]
    return episodes, latencies


def stats(episodes, latencies):
    calls = sum(len(e["calls"]) for e in episodes)
    executed = sum(c["h_exec"] for e in episodes for c in e["calls"])
    ms = float(np.mean(latencies)) if latencies else float("nan")
    calls_per_ep = calls / len(episodes)
    return {
        "episodes": len(episodes),
        "success": 100.0 * sum(e["success"] for e in episodes) / len(episodes),
        "calls_per_ep": calls_per_ep,
        "ms": ms,
        "s_per_ep": calls_per_ep * ms / 1000.0,
        "h_bar": executed / calls,
    }


def main(run_dirs):
    runs = {d: load(d) for d in run_dirs}
    ref = None
    print(f"{'run':<28} {'suite':<15} {'eps':>5} {'succ.':>6} {'calls/ep':>15} {'ms':>6} {'s/ep':>6} {'h_bar':>6}")
    for d, (episodes, latencies) in runs.items():
        if not episodes:
            print(f"{d:<28} (no episodes)")
            continue
        total = stats(episodes, latencies)
        ref = ref or total
        for suite in SUITES + ["all"]:
            eps = episodes if suite == "all" else [e for e in episodes if e["suite"] == suite]
            if not eps:
                continue
            s = stats(eps, latencies)
            reduction = f"({ref['calls_per_ep'] / s['calls_per_ep']:.2f}x)" if suite == "all" else ""
            print(
                f"{pathlib.Path(d).name:<28} {suite:<15} {s['episodes']:>5} {s['success']:>6.2f} "
                f"{s['calls_per_ep']:>7.1f} {reduction:<7} {s['ms']:>6.0f} {s['s_per_ep']:>6.2f} {s['h_bar']:>6.2f}"
            )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
