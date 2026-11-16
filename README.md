# FuzRL Lab

[![CPU regression tests](https://github.com/DIVYANSH-675/FuzRL-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/DIVYANSH-675/FuzRL-Lab/actions/workflows/tests.yml)

A reproducible experimentation toolkit built on [waunx/FuzRL](https://github.com/waunx/FuzRL), the implementation of **Fuz-RL: A Fuzzy-Guided Robust Framework for Safe Reinforcement Learning under Uncertainty**, NeurIPS 2025.

Maintained extensions by **Divyansh Gupta**. The original research, attribution, MIT notices, and Git history are preserved. See [upstream documentation](UPSTREAM_README.md) and [change details](CHANGELOG.md).

## What improved

- **Correct constraint updates:** exceeding the cost budget increases the Lagrange multiplier. Epochs without completed episodes leave it unchanged.
- **Reliable learning targets:** true terminal states do not bootstrap; time limits do. Fuzzy Bellman targets also mask terminal next-state values.
- **Stable advantage normalization:** constant reward or cost advantages no longer divide by zero.
- **Environment compatibility:** supports Gym's four-value, Gymnasium's five-value, and Safety-Gymnasium's six-value step APIs, without hard-coded environment names.
- **Reproducible disturbances:** independent seeded noise generator; observation, continuous-action, and MuJoCo velocity disturbances; white noise, impulse, and periodic profiles. No hidden simulator steps during construction.
- **Practical installation:** one root package, optional Safety-Gymnasium/MPI/tracking dependencies, and CPU support without MPI.
- **Useful artifacts:** post-update policy weight checkpoints, training configuration, runtime/version manifest, and JSON evaluations across noise levels and seeds.
- **Regression checks:** automated tests and CPU CI for Python 3.10 and 3.12.

These are implementation and experiment-workflow improvements. They are **not a new algorithm or a claim of better benchmark performance**.

## Install

Python 3.10–3.12 is the tested target. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
# CPU-only installation; install an appropriate PyTorch build separately for CUDA.
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
```

Optional integrations:

```bash
python -m pip install -e '.[safety]'   # Safety-Gymnasium / MuJoCo tasks
python -m pip install -e '.[mpi]'      # Also requires a system MPI runtime
python -m pip install -e '.[tracking]' # Opt-in SwanLab logging
```

Safety-Gymnasium has its own Python and simulator dependency requirements; use a separate compatible environment if its constraints conflict. Ordinary Gymnasium tasks work with the base installation.

## Train a small CPU experiment

```bash
python scripts/run_training.py \
  --algorithm fuzppolag --env Pendulum-v1 --cpu \
  --epochs 2 --steps-per-epoch 64 --max-ep-len 32 \
  --train-pi-iters 2 --train-v-iters 2 \
  --level 3 --train-eps 0.1 --disturb-part observation \
  --exp-name pendulum-smoke
```

This is a functional smoke test, not enough training to produce a capable controller. Pendulum has no safety cost: zero reported cost is **not evidence of safety**.

Algorithms: `ppolag`, `fuzppolag`, `cvarppolag`. Observations must be flat numeric vectors; actions may be discrete or a flat continuous Box. Action disturbance requires continuous actions. Standard environments can supply `info['cost']` or `info['constraint_violation']`; otherwise cost defaults to zero.

The launcher accepts both hyphenated and underscore option names. It works from other working directories, supports `--dry-run`, and defaults to automatic CPU/CUDA selection. Explicit `--gpu 3` selects physical GPU 3 and uses logical `cuda:0` in the child process. Existing `CUDA_VISIBLE_DEVICES` settings are respected when no GPU override is supplied.

For a Safety-Gymnasium experiment after installing the optional dependency:

```bash
python scripts/run_training.py \
  --algorithm fuzppolag --env SafetyPointGoal1-v0 \
  --epochs 100 --seed 42 --cost-limit 25 \
  --train-eps 0.1 --disturb-part observation
```

Use `--num-cpu 2` for MPI or `--use-swanlab` for single-process tracking. Set tracking credentials using the `SWANLAB_API_KEY` environment variable. Logs are local by default.

## Evaluate a checkpoint under noise

```bash
python scripts/evaluate.py logs/pendulum-smoke \
  --noise 0 0.1 0.3 --seeds 100 101 102 \
  --max-steps 200 --output logs/pendulum-evaluation.json
```

The report contains every episode's reward, cumulative cost, length, budget violation, and whether the evaluation horizon cut it short. It also summarizes mean reward, mean/worst cost, upper-tail cost CVaR at 10%, and violation rate. All noise settings use the same seeds and a deterministic policy. `--checkpoint last` selects the latest numbered policy-weight file; `best` is available for fuzzy training and is selected by training return, **not by safety**.

Evaluation loads plain `state_dict` weights with `weights_only=True`, reconstructing the policy from the run configuration. It does not deserialize the legacy environment pickle. Reports with cutoffs describe partial episodes; choose a long enough evaluation horizon for complete-episode comparisons.

## Experiment outputs

```text
logs/<experiment>/
  config.json          Algorithm parameters
  run.json             CLI settings, dependency versions, source revision
  progress.txt         Per-epoch tab-separated metrics
  pyt_save/weightsN.pt  Policy and value-network state dictionaries
  pyt_save/modelN.pt   Upstream-compatible full-model checkpoint
  varsN.pkl            Upstream-compatible environment snapshot, when serializable
```

Weights support evaluation; they do not constitute resumable training state. Optimizers, fuzzy-network state, and RNG state are not all included. `NaN` episode statistics mean no episode completed in that epoch; `Episodes` records the count. This is separate from numerical failures in training losses.

## Validation and research limitations

```bash
python -m pytest -q
python -m build
```

The regression suite exercises all three algorithms on a seeded cost-bearing test environment, including epochs that end mid-episode, terminal/time-limit targets, multiplier direction, finite losses, checkpoint contents, and repeated evaluation. Short CPU training/evaluation on Gymnasium Pendulum is also checked locally.

Full paper benchmarks, real Safety-Gymnasium/MuJoCo dynamics, CUDA, and multi-process MPI have not been validated in this revision. Existing general non-additive Sugeno-lambda solver behavior has not been exhaustively audited. The CVaR variant retains the upstream batch-tail critic formulation; fixing cost-tail direction does not establish a fully validated distributional or risk-constrained RL method.

Disturbance semantics changed intentionally: the default is no noise (`train_eps=0`), strength is in the target's native units, and MuJoCo dynamics noise modifies velocity before a step. It does not reproduce the upstream friction-drift schedule. Compare methods under the same explicit configuration; these runs are not drop-in reproductions of the published results.

## Attribution and license

Upstream Fuz-RL: Xu Wan, Chao Yang, Cheng Yang, Jie Song, and Mingyang Sun. [Paper](https://arxiv.org/abs/2602.20729). Original notices: [MIT license](LICENSE), [OpenAI Spinning Up MIT license](src/LICENSE). New contributions are also MIT licensed; see [additional contribution license](ADDITIONS_LICENSE).

```bibtex
@inproceedings{wan2025fuzrl,
  title={Fuz-RL: A Fuzzy-Guided Robust Framework for Safe Reinforcement Learning under Uncertainty},
  author={Xu Wan and Chao Yang and Cheng Yang and Jie Song and Mingyang Sun},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025}
}
```
