# Don't Throw Away the Tail: Action Upcycling for Policy Acceleration

This repository is the official implementation of "Don't Throw Away the Tail: Action Upcycling for Policy Acceleration", built upon the official [openpi](https://github.com/Physical-Intelligence/openpi) repository.

## Action Upcycling

A chunked policy predicts a chunk of $H$ actions from one observation, executes only the first $h$ of them
(the *execution horizon*), and discards the rest (the *tail*) before replanning.
Discarded actions stay close to their replanned versions as long as the action velocity stays smooth.
*Action Upcycling* extends the execution horizon up to the point where the velocity begins to fluctuate:

1. **Velocity fluctuation as a signal.** For each predicted chunk, accumulate the change between consecutive velocities along the tail:
   $c_k = \sum_{j=h+1}^{k} \lVert v_j - v_{j-1} \rVert_2$ for $h < k \le H$. For LIBERO, $v_k = a_k$ (relative end-effector displacement, gripper excluded).
2. **Adaptive horizon selection.** Execute $h_{\text{exec}} = \max_{c_k \le \tau} k$ actions ($h \le k \le H$), then replan.
3. **Threshold from a pool.** Collect the signals $c_{h+1}, \dots, c_H$ of chunks predicted by the policy into a pool $\mathcal{C}$,
   then set $\tau$ to the smallest element of $\mathcal{C}$ for which the mean execution length satisfies $\bar{h}(\tau) \ge r\,h$,
   where $r$ is the *upcycling ratio*. $r = 1$ (or $\tau = 0$) is the default setting of the policy.

The method is training-free and reads only the chunk that has already been sampled, so it needs no access to model internals and no extra samples.
The entire method is in [`examples/libero/action_upcycling.py`](examples/libero/action_upcycling.py) and uses only numpy.

## Results (π0.5, LIBERO)

π0.5 predicts $H = 10$ actions per call and executes $h = 5$ under its default setting. We use $N = 10$ denoising steps and $r = 1.5$.

| | Spatial | Object | Goal | Long | Avg. succ. (%) | Calls / ep | ms / call | s / ep |
|---|---|---|---|---|---|---|---|---|
| π0.5 | 97.6 | 98.6 | 97.8 | 93.6 | 96.9 | 32.4 (1×) | 136 | 4.41 |
| + Action Upcycling | 98.6 | 99.4 | 97.6 | 96.0 | **97.9** | **21.9 (1.48×)** | 137 | **3.00** |

We measured latency on a single RTX 4090.

## Installation

Clone the repository and fetch LIBERO at the commit pinned by openpi:

```bash
git clone https://github.com/star-kwon/action-upcycling.git
cd action-upcycling
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/libero
git -C third_party/libero checkout f78abd68ee283de9f9be3c8f7e2a9ad60246e95c
```

We use [uv](https://docs.astral.sh/uv/) to manage Python dependencies, as in openpi:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
# openpi installs both opencv-python and opencv-python-headless; make sure cv2 comes from the headless build
uv pip install --reinstall --no-deps opencv-python-headless==4.11.0.86
```

Create the LIBERO client environment, as in the official [LIBERO example](examples/libero/README.md):

```bash
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
```

## Running Inference

Terminal window 1: run the π0.5 LIBERO policy server. For the PyTorch backend, see [PyTorch Support](README_openpi.md#pytorch-support).

```bash
uv run scripts/serve_policy.py --env LIBERO
```

Terminal window 2: evaluate the baseline and Action Upcycling on the four LIBERO suites (2,000 episodes each), then compare them:

```bash
source examples/libero/.venv/bin/activate

bash examples/libero/run_upcycling.sh baseline                                  # default setting (h = 5)
bash examples/libero/run_upcycling.sh upcycling --config pi05_libero_r1.5       # Action Upcycling (r = 1.5)

python examples/libero/summarize.py results/baseline results/upcycling
```
