"""Select the fluctuation threshold tau for a target upcycling ratio r from an offline pool.

The pool C holds the velocity fluctuation signals c_{h+1}, ..., c_H of chunks predicted by the policy itself,
e.g. from the logs of a run under the default setting (no demonstrations or success labels are needed).
tau is the smallest element of C whose mean execution length over the pool satisfies h_bar(tau) >= r * h.

    python examples/libero/calibrate_threshold.py results/baseline --ratio 1.5
    python examples/libero/calibrate_threshold.py results/baseline --ratio 1.1 1.25 1.5 --fraction 0.02
"""

import argparse
import pathlib

import numpy as np

import action_upcycling as _upcycling


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log_dir", help="directory with the *.jsonl logs written by main.py (--args.log-path)")
    parser.add_argument("--ratio", type=float, nargs="+", default=[1.5], help="target upcycling ratio(s) r")
    parser.add_argument("--replan-steps", type=int, default=5, help="execution horizon h of the logged run")
    parser.add_argument("--fraction", type=float, default=1.0, help="use a random fraction of the chunks (e.g. 0.02)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    paths = sorted(pathlib.Path(args.log_dir).glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No *.jsonl logs in {args.log_dir}")
    pool = _upcycling.load_signals(str(p) for p in paths)
    if args.fraction < 1.0:
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(len(pool), max(1, int(round(args.fraction * len(pool)))), replace=False)
        pool = [pool[i] for i in keep]

    h = args.replan_steps
    print(f"pool: {len(pool)} chunks from {len(paths)} log file(s), h = {h}, H = {h + len(pool[0])}")
    print(f"{'ratio r':>8} {'tau':>10} {'h_bar(tau)':>11}")
    for r in args.ratio:
        tau = _upcycling.threshold_from_pool(pool, h, r)
        print(f"{r:>8.2f} {tau:>10.4f} {_upcycling.mean_execution_length(pool, h, tau):>11.2f}")


if __name__ == "__main__":
    main()
