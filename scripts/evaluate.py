#!/usr/bin/env python3
"""Evaluate policy weights over a reproducible grid of noise strengths and seeds."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from train import create_environment
from spinup.algos.ppolag.core import MLPActorCritic
from spinup.algos.cvarppolag.cvarppolag import compute_cvar
from spinup.utils.env_api import reset_env, step_env


def evaluate(model, env, seeds, max_steps, cost_limit):
    episodes = []
    model.eval()
    try:
        for seed in seeds:
            obs = reset_env(env, seed=seed)
            reward = cost = 0.0
            ended = False
            for step in range(1, max_steps + 1):
                action = model.act(torch.as_tensor(obs, dtype=torch.float32), deterministic=True)
                obs, r, c, terminated, truncated, _ = step_env(env, action)
                reward += r
                cost += c
                if terminated or truncated:
                    ended = True
                    break
            episodes.append(dict(seed=seed, reward=reward, cost=cost, length=step,
                                 budget_exceeded=cost > cost_limit, cutoff=not ended))
    finally:
        env.close()
    costs = [e['cost'] for e in episodes]
    return dict(episodes=episodes, mean_reward=float(np.mean([e['reward'] for e in episodes])),
                mean_cost=float(np.mean(costs)), worst_cost=max(costs),
                upper_cost_cvar_10=compute_cvar(costs, alpha=0.1, upper=True),
                violation_rate=float(np.mean([e['budget_exceeded'] for e in episodes])))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_dir', type=Path)
    p.add_argument('--checkpoint', default='last', help='last, best, or epoch number')
    p.add_argument('--env', help='Defaults to training environment')
    p.add_argument('--noise', type=float, nargs='+', default=[0.0, 0.1, 0.3])
    p.add_argument('--seeds', type=int, nargs='+', default=[100, 101, 102, 103, 104])
    p.add_argument('--max-steps', type=int, default=1000)
    p.add_argument('--cost-limit', type=float)
    p.add_argument('--disturb-part', choices=['observation', 'action', 'dynamics', 'multi_source'], default='observation')
    p.add_argument('--disturb-type', choices=['white_noise', 'impulse', 'periodic'], default='white_noise')
    p.add_argument('--output', type=Path, default=Path('evaluation.json'))
    a = p.parse_args()
    if a.max_steps < 1 or any(not np.isfinite(x) or x < 0 for x in a.noise):
        p.error('max-steps must be positive and noise strengths finite and nonnegative')
    if len(set(a.seeds)) != len(a.seeds) or min(a.seeds) < 0:
        p.error('seeds must be distinct nonnegative integers')
    config = json.loads((a.run_dir / 'config.json').read_text())
    env_name = a.env or config['env_name']
    cost_limit = a.cost_limit if a.cost_limit is not None else config.get('cost_limit', 25)
    if not np.isfinite(cost_limit) or cost_limit < 0:
        p.error('cost-limit must be finite and nonnegative')
    checkpoint = a.checkpoint
    if checkpoint == 'last':
        numbers = [int(f.stem[7:]) for f in (a.run_dir / 'pyt_save').glob('weights*.pt') if f.stem[7:].isdigit()]
        if not numbers:
            p.error('No numbered weights checkpoints found')
        checkpoint = str(max(numbers))
    elif checkpoint != 'best' and not checkpoint.isdigit():
        p.error('checkpoint must be last, best, or an epoch number')
    weights_path = a.run_dir / 'pyt_save' / f'weights{checkpoint}.pt'
    base = create_environment(env_name)()
    try:
        model = MLPActorCritic(base.observation_space, base.action_space, **config['ac_kwargs'])
    finally:
        base.close()
    model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
    results = []
    for strength in a.noise:
        env = create_environment(env_name, strength, a.disturb_part, a.disturb_type)()
        results.append(dict(noise=strength, **evaluate(model, env, a.seeds, a.max_steps, cost_limit)))
    report = dict(environment=env_name, checkpoint=str(weights_path.resolve()),
                  cost_limit=cost_limit, disturbance=a.disturb_part, noise_type=a.disturb_type,
                  note='Zero cost in ordinary Gymnasium tasks is not a safety benchmark. Cutoffs are partial episodes.',
                  results=results)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'Evaluation saved to {a.output}')


if __name__ == '__main__':
    main()
