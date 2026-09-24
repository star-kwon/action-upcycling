# Don't Throw Away the Tail: Action Upcycling for Policy Acceleration

[Project page](https://actionupcycling.github.io/)

This repository reproduces the **π0.5 LIBERO** results of *Action Upcycling*. It is built on the official
[openpi](https://github.com/Physical-Intelligence/openpi) repository with a few small changes (listed [below](#changes-to-openpi)).
The original openpi README is kept in [README_openpi.md](README_openpi.md).

## Action Upcycling in a nutshell

A chunked policy predicts a chunk of $H$ actions from one observation, executes only the first $h$ of them
(the *execution horizon*), and discards the rest (the *tail*) before replanning.
Discarded actions stay close to their replanned versions as long as the action velocity stays smooth.
Action Upcycling therefore extends the execution horizon up to the point where the velocity begins to fluctuate:

1. **Velocity fluctuation as a signal.** For each predicted chunk, accumulate the change between consecutive velocities along the tail:
   $c_k = \sum_{j=h+1}^{k} \lVert v_j - v_{j-1} \rVert_2$ for $h < k \le H$. For LIBERO, $v_k = a_k$ (relative end-effector displacement, gripper excluded).
2. **Adaptive horizon selection.** Execute $h_{\text{exec}} = \max_{c_k \le \tau} k$ actions ($h \le k \le H$), then replan.
3. **Threshold from a pool.** Collect the signals $c_{h+1}, \dots, c_H$ of chunks predicted by the policy into a pool $\mathcal{C}$,
   then set $\tau$ to the smallest element of $\mathcal{C}$ for which the mean execution length satisfies $\bar{h}(\tau) \ge r\,h$,
   where $r$ is the *upcycling ratio*. $r = 1$ (or $\tau = 0$) is the default setting of the policy.

The method is training-free and reads only the chunk that has already been sampled, so it needs no access to model internals and no extra samples.
The entire method is in [`examples/libero/action_upcycling.py`](examples/libero/action_upcycling.py) and uses only numpy.

## Results (π0.5, LIBERO, 4 suites × 10 tasks × 50 trials = 2,000 episodes)

π0.5 predicts $H = 10$ actions per call and executes $h = 5$ under its default setting. We use $N = 10$ denoising steps and $r = 1.5$ ($\tau = 0.1454$).

| | Spatial | Object | Goal | Long | Avg. succ. (%) | Calls / ep | ms / call | s / ep |
|---|---|---|---|---|---|---|---|---|
| π0.5 | 97.6 | 98.6 | 97.8 | 93.6 | 96.9 | 32.4 (1×) | 136 | 4.41 |
| + Action Upcycling | 98.6 | 99.4 | 97.6 | 96.0 | **97.9** | **21.9 (1.48×)** | 137 | **3.00** |

We measured latency on a single RTX 4090. Absolute latency depends on the GPU and the backend. The number of policy calls does not.
Closed-loop simulation is not bit-exact across machines, so expect small differences in the success rate.

## Setup

Unzip the repository, then fetch LIBERO at the commit pinned by openpi:

```bash
cd action-upcycling-openpi
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/libero
git -C third_party/libero checkout f78abd68ee283de9f9be3c8f7e2a9ad60246e95c
```

**Policy server** (same as openpi; requires [uv](https://docs.astral.sh/uv/)):

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

**LIBERO client** (same as the official [LIBERO example](examples/libero/README.md)):

```bash
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
```

## Reproducing the main result

**1. Start the π0.5 LIBERO policy server** (terminal 1). The simplest option is the JAX checkpoint, which openpi downloads automatically:

```bash
uv run scripts/serve_policy.py --env LIBERO
```

<details>
<summary>PyTorch backend (the setting used for the numbers above)</summary>

Follow the openpi [PyTorch instructions](README_openpi.md#pytorch-support) to patch `transformers`, then convert the released checkpoint and serve it:

```bash
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
uv run python -c "from openpi.shared import download; print(download.maybe_download('gs://openpi-assets/checkpoints/pi05_libero'))"
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero/params \
    --config_name pi05_libero \
    --output_path checkpoints/pi05_libero_pytorch
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=checkpoints/pi05_libero_pytorch
```
</details>

**2. Run the baseline and Action Upcycling** (terminal 2, inside the client environment). Each run evaluates all four suites,
writes one log per suite to `results/<run_name>/`, and prints a summary at the end:

```bash
source examples/libero/.venv/bin/activate

# Baseline: default setting of pi0.5 (h = 5)
bash examples/libero/run_upcycling.sh baseline

# Action Upcycling with the threshold used in the paper (r = 1.5)
bash examples/libero/run_upcycling.sh upcycling --args.upcycle-tau 0.1454

# Compare the two runs (success rate, calls / ep, ms / call, s / ep, mean execution length)
python examples/libero/summarize.py results/baseline results/upcycling
```

**3. (Optional) Select the threshold yourself.** Every call of the baseline run logs its velocity fluctuation signal,
so the pool $\mathcal{C}$ comes from that run without additional rollouts:

```bash
python examples/libero/calibrate_threshold.py results/baseline --ratio 1.5
#  ratio r        tau  h_bar(tau)
#     1.50     0.1451        7.50

# A small pool suffices: a random 2 % of the chunks gives nearly the same threshold
python examples/libero/calibrate_threshold.py results/baseline --ratio 1.5 --fraction 0.02
```

Then pass the printed value to `--args.upcycle-tau`. To trade off policy calls against reactivity, pass other ratios, e.g. `--ratio 1.1 1.25 1.5 2.0`.

## Other experiments in the paper

| Experiment | Command | Paper result |
|---|---|---|
| Online pool (update $\tau$ from the signals seen during deployment) | `bash examples/libero/run_upcycling.sh online --args.upcycle-online-ratio 1.5` | 97.7 %, 21.0 calls / ep |
| Fixed-length execution ($h = 7$, close to $\bar{h}$ of Action Upcycling) | `bash examples/libero/run_upcycling.sh fixed7 --args.replan-steps 7` | 97.05 %, 23.0 calls / ep |
| Few-step sampling ($N = 5$ or $2$) | server: add `--num-steps 5` (or `2`); then run a baseline, select $\tau$ from it, and run Action Upcycling | $N{=}5$: 96.9 → 97.7 %<br>$N{=}2$: 96.8 → 97.8 % |

The thresholds selected in the paper for few-step sampling were $\tau = 0.1426$ ($N = 5$) and $\tau = 0.1412$ ($N = 2$).

Useful environment variables for `run_upcycling.sh`: `PORT` (default 8000), `SUITES` (e.g. `SUITES=libero_spatial`), `TRIALS` (default 50), and `RESULTS_DIR`.
Finished suites are skipped, so an interrupted run can be resumed with the same command.
Add `--args.no-save-videos` to skip writing rollout videos.

## Changes to openpi

| File | Change |
|---|---|
| `examples/libero/action_upcycling.py` | **New.** Velocity fluctuation signal, adaptive horizon selection, threshold selection from a pool, online pool |
| `examples/libero/main.py` | After each policy call, execute `h_exec` actions instead of `replan_steps`; per-call logging (`--args.log-path`). With the default arguments it behaves exactly like the official script |
| `examples/libero/calibrate_threshold.py` | **New.** Selects $\tau$ for a target upcycling ratio from the logged pool |
| `examples/libero/summarize.py` | **New.** Success rate, policy calls, latency, and inference time per episode |
| `examples/libero/run_upcycling.sh` | **New.** Runs the four LIBERO suites for one setting |
| `scripts/serve_policy.py` | Optional `--num-steps` (number of denoising steps) |

The policy server is unchanged apart from `--num-steps`: Action Upcycling runs entirely on the client, on the chunk returned by the policy.

### Using Action Upcycling with another policy

The method needs only the predicted action chunk. In your control loop, replace `actions[:h]` with:

```python
import action_upcycling as au

signal = au.velocity_fluctuation(chunk, h, relative=True)  # relative=False for absolute joint positions
h_exec = au.execution_length(signal, h, tau)
execute(chunk[:h_exec])
```

To select `tau`, collect `signal` over a few rollouts and call `au.threshold_from_pool(signals, h, ratio)`.
If your action layout is not LIBERO's (6 motion dimensions followed by the gripper), pass `dims=` to select the motion dimensions.

## License

This repository follows the license of openpi ([Apache 2.0](LICENSE)). The Gemma model weights are covered by [LICENSE_GEMMA.txt](LICENSE_GEMMA.txt).
