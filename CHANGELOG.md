# Changes from upstream

## 0.2.0 — 2026-11-17

### Learning correctness

- Reverse the dual-loss sign and optimize the positive softplus multiplier: costs above the limit now strengthen the penalty. Skip the optimizer step entirely when no episode cost is available, including optimizer momentum.
- Handle terminal versus truncated transitions consistently in all three algorithms. Mask the fuzzy next-state targets on true termination.
- Retain partial fuzzy-training episodes across epoch boundaries instead of silently restarting them.
- Avoid division by zero in advantage normalization, and keep episode metric columns stable when no episode finishes.
- Use the high-cost tail for the CVaR cost critic, with fractional boundary weighting; use the low-return tail for rewards. This changes the upstream baseline's results.
- Normalize softmax fuzzy singleton densities to sum to one and follow the module's actual device after `.to(...)`.
- Save numbered policies after the optimizer update so the final checkpoint includes the last update.

### Experiment tooling

- Root-installable package; optional simulator, MPI, plotting, and tracking dependencies.
- CPU operation without mpi4py; explicit error for requested MPI without a runtime; device-aware copies for MPI parameter/gradient synchronization.
- Shared reset/step adapters with environment seeding, action clipping, cost extraction, and finite-value checks.
- Replace the noise wrapper's simulator probes, unseeded global RNG, undefined periodic-noise variable, and silently ineffective disturbance paths with explicit seeded modes. Dynamics noise targets MuJoCo velocity, not an observation approximation.
- Portable launcher, correct CUDA device remapping, supported algorithm choices, consistent noise forwarding to all algorithms, and opt-in remote logging.
- Plain-weight evaluation over matched seed/noise grids, including episode records, costs, violations, and upper-tail cost CVaR.
- Run metadata and dependency versions, duplicate-run protection, regression tests, and CPU CI.

Original notices and history remain intact. New experiment defaults and correctness fixes can change results; no improved benchmark scores are claimed.
