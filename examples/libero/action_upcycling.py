"""Action Upcycling: reuse the tail of an action chunk while its velocity stays smooth.

A chunked policy predicts H actions per call, executes the first h, and discards the rest (the *tail*).
Action Upcycling keeps executing tail actions as long as the accumulated velocity fluctuation

    c_k = sum_{j=h+1}^{k} || v_j - v_{j-1} ||_2 ,    h < k <= H

stays below a threshold tau, i.e. it executes h_exec = max{k : c_k <= tau} actions (with c_h = 0).
tau is set from a pool C of signals so that the mean execution length reaches r * h, where r is the
upcycling ratio (r = 1 recovers the default setting of the policy).

Only numpy is required, so this file runs in the LIBERO client environment (Python 3.8).
"""

import math
from typing import Iterable, List, Sequence

import numpy as np

# LIBERO actions are relative end-effector displacements (3 translation + 3 rotation) followed by the
# gripper command. The velocity is the action itself, and the gripper dimension is excluded.
LIBERO_VELOCITY_DIMS = slice(0, 6)


def velocity(chunk: np.ndarray, *, relative: bool = True, dims: slice = LIBERO_VELOCITY_DIMS) -> np.ndarray:
    """Velocity v_k induced by each action of a chunk [H, D].

    relative=True:  v_k = a_k            (relative displacements, e.g. LIBERO)
    relative=False: v_k = a_k - a_{k-1}  (absolute joint positions, e.g. RoboTwin 2.0)
    """
    a = np.asarray(chunk, dtype=np.float64)[:, dims]
    return a if relative else np.diff(a, axis=0, prepend=a[:1])


def velocity_fluctuation(chunk: np.ndarray, h: int, **velocity_kwargs) -> np.ndarray:
    """Signal c_{h+1}, ..., c_H of one chunk (Eq. 3). Non-decreasing, length H - h."""
    v = velocity(chunk, **velocity_kwargs)
    # v is 0-indexed, so ||v_j - v_{j-1}|| for j = h+1..H (1-indexed) is diff[h-1:].
    return np.cumsum(np.linalg.norm(np.diff(v, axis=0), axis=1)[h - 1 :])


def execution_length(signal: np.ndarray, h: int, tau: float) -> int:
    """Adaptive horizon selection (Eq. 4): h_exec = max{k in [h, H] : c_k <= tau}.

    tau <= 0 disables upcycling and returns h (the default setting of the policy).
    Since c_k is non-decreasing, the number of reused tail actions is #{k : c_k <= tau}.
    """
    if tau <= 0:
        return h
    return h + int(np.searchsorted(signal, tau, side="right"))


def mean_execution_length(pool: Sequence[np.ndarray], h: int, tau: float) -> float:
    """Mean execution length h_bar(tau) over the chunks of a pool."""
    return float(np.mean([execution_length(c, h, tau) for c in pool]))


def threshold_from_pool(pool: Sequence[np.ndarray], h: int, ratio: float) -> float:
    """Smallest element tau of the pool C with h_bar(tau) >= ratio * h.

    Because every c_k is non-decreasing along its chunk, h_bar(tau) = h + #{c in C : c <= tau} / N
    for a pool of N chunks, so tau is the ceil((ratio - 1) * h * N)-th smallest element of C.
    """
    if ratio <= 1.0 or len(pool) == 0:
        return 0.0
    flat = np.sort(np.concatenate([np.asarray(c, dtype=np.float64) for c in pool]))
    need = math.ceil((ratio - 1.0) * h * len(pool) - 1e-9)
    if need > flat.size:  # the target exceeds H / h: execute every chunk as a whole
        return float("inf")
    return float(flat[need - 1])


class OnlinePool:
    """Online pool: signals of the chunks predicted during deployment, with tau updated at every call."""

    def __init__(self, h: int, ratio: float):
        self.h = h
        self.ratio = ratio
        self.pool: List[np.ndarray] = []

    def update(self, signal: np.ndarray) -> float:
        self.pool.append(np.asarray(signal, dtype=np.float64))
        return threshold_from_pool(self.pool, self.h, self.ratio)


def load_signals(paths: Iterable[str]) -> List[np.ndarray]:
    """Collect the signals logged by examples/libero/main.py (one JSON line per episode)."""
    import json

    pool = []
    for path in paths:
        with open(path) as f:
            for line in f:
                for call in json.loads(line)["calls"]:
                    pool.append(np.asarray(call["signal"], dtype=np.float64))
    return pool
