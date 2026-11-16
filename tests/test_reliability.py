import csv
import importlib
import json
from pathlib import Path
import subprocess
import sys

import gymnasium as gym
import numpy as np
import pytest
import torch

from spinup.utils.env_api import reset_env, step_env, unpack_step
from spinup.utils.noise import EnvironmentNoiseWrapper
from spinup.utils.training import normalize_advantages, penalty_loss
from spinup.utils.fuzzynet import FuzzyNet, choquet_integral, choquet_integral_dual
from spinup.algos.cvarppolag.cvarppolag import compute_cvar
from spinup.algos.ppolag.core import MLPActorCritic

ROOT = Path(__file__).resolve().parents[1]


class TinyCostEnv(gym.Env):
    """Finite, seeded task with explicit cost and controllable episode endings."""
    observation_space = gym.spaces.Box(-10, 10, (3,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)

    def __init__(self, length=5, truncated=False):
        self.length, self.truncated = length, truncated
        self.resets = self.steps = 0
        self.closed = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.resets += 1
        self.t = 0
        return self.np_random.uniform(-0.1, 0.1, 3).astype(np.float32), {}

    def step(self, action):
        self.steps += 1
        self.t += 1
        self.last_action = action
        end = self.t == self.length
        return np.full(3, self.t / 10, np.float32), 1.0, end and not self.truncated, end and self.truncated, {'cost': 2.0}

    def close(self):
        self.closed = True


@pytest.mark.parametrize('result,flags,cost', [
    (([0], 1, True, {'TimeLimit.truncated': True, 'cost': 3}), (False, True), 3),
    (([0], 1, True, {'constraint_violation': 2}), (True, False), 2),
    (([0], 1, False, True, {}), (False, True), 0),
    (([0], 1, 4, True, False, {}), (True, False), 4),
])
def test_step_formats(result, flags, cost):
    converted = unpack_step(result)
    assert converted[2] == cost
    assert converted[3:5] == flags


@pytest.mark.parametrize('result', [([0], np.nan, False, False, {}), ([np.inf], 1, False, False, {})])
def test_nonfinite_environment_fails(result):
    with pytest.raises(ValueError, match='non-finite'):
        unpack_step(result)


def test_step_clips_continuous_actions():
    env = TinyCostEnv()
    reset_env(env, seed=9)
    step_env(env, np.array([100]))
    assert env.last_action.item() == 1


def test_legacy_reset_preserves_entire_observation():
    class Legacy:
        def reset(self):
            return np.array([1, 2, 3])
    np.testing.assert_array_equal(reset_env(Legacy()), [1, 2, 3])


@pytest.mark.parametrize('kind', ['white_noise', 'impulse', 'periodic'])
@pytest.mark.parametrize('part', ['observation', 'action'])
def test_noise_seeded_and_resettable(kind, part):
    env = TinyCostEnv(length=200)
    wrapped = EnvironmentNoiseWrapper(env, 0.5, part, kind)
    assert env.resets == env.steps == 0  # Construction must not probe the simulator.
    def rollout():
        observations = [wrapped.reset(seed=123)[0]]
        actions = []
        for _ in range(25):
            observations.append(wrapped.step(np.array([0.2], np.float32))[0])
            actions.append(env.last_action.copy())
        return np.array(observations), np.array(actions)
    first = rollout()
    np.random.seed(999)
    second = rollout()
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert np.max(np.abs(second[1])) <= 1
    assert second[0].dtype == np.float32


def test_dynamics_not_silently_replaced_with_observation_noise():
    with pytest.raises(ValueError, match='MuJoCo'):
        EnvironmentNoiseWrapper(TinyCostEnv(), disturb_part='dynamics')


@pytest.mark.parametrize('deviation', [-2., 2.])
def test_multiplier_moves_toward_constraint_satisfaction(deviation):
    p = torch.tensor(1., requires_grad=True)
    optimizer = torch.optim.SGD([p], lr=0.1)
    penalty_loss(p, deviation).backward()
    optimizer.step()
    assert (p.item() - 1) * deviation > 0


def test_constant_advantages_are_finite_zero():
    np.testing.assert_array_equal(normalize_advantages(np.ones(8)), np.zeros(8))
    with pytest.raises(ValueError):
        normalize_advantages(np.array([np.nan]))


def test_cvar_opposite_tails_and_fractional_boundary():
    assert compute_cvar([0, 10], 0.75, upper=True) == pytest.approx(20 / 3)
    assert compute_cvar([0, 10], 0.75) == pytest.approx(10 / 3)
    assert compute_cvar([0, 10], 0.1, upper=True) == 10
    with pytest.raises(ValueError):
        compute_cvar([1, 2], 0)


def test_fuzzy_additive_measure_and_gradients():
    model = FuzzyNet(3, n_singletons=5, use_softmax=True, device='cpu')
    g = model(torch.zeros(4, 3))
    torch.testing.assert_close(g.sum(1), torch.ones(4), atol=1e-7, rtol=1e-7)
    values = torch.tensor([-3., -1., 0., 2., 4.]).expand(4, -1)
    expected = (g * values).sum(1, keepdim=True)
    result = choquet_integral(g, values, use_lambda=False)
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(choquet_integral_dual(g, values, use_lambda=False), expected)
    result.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize('name', ['ppolag', 'fuzppolag', 'cvarppolag'])
@pytest.mark.parametrize('truncated', [False, True])
def test_training_crosses_epoch_boundary_and_saves_updated_weights(tmp_path, monkeypatch, name, truncated):
    torch.set_num_threads(1)
    env = TinyCostEnv(truncated=truncated)
    module = importlib.import_module(f'spinup.algos.{name}.{name}')
    algorithm = getattr(module, 'fuzzyppolag' if name == 'fuzppolag' else name)
    original_actor = module.core.MLPActorCritic
    actors = []
    before = []
    def actor(*args, **kwargs):
        model = original_actor(*args, **kwargs)
        actors.append(model)
        before.append({k: v.clone() for k, v in model.state_dict().items()})
        return model
    finished = []
    buffer_cls = module.FuzBuffer if name == 'fuzppolag' else module.PPOBuffer
    original_finish = buffer_cls.finish_path
    def finish(buffer, last_val=0, last_cval=0):
        finished.append((env.t, float(last_val), float(last_cval)))
        return original_finish(buffer, last_val, last_cval)
    monkeypatch.setattr(buffer_cls, 'finish_path', finish)
    kwargs = dict(env_fn=lambda: env, actor_critic=actor, env_name='CustomCost-v0', device='cpu',
                  ac_kwargs={'hidden_sizes': (8,)}, seed=3, epochs=2, steps_per_epoch=3,
                  max_ep_len=10, train_pi_iters=2, train_v_iters=2, cost_limit=1,
                  logger_kwargs={'output_dir': str(tmp_path)}, save_freq=1)
    if name == 'fuzppolag':
        kwargs.update(level=3, fuz_freq=1)
    algorithm(**kwargs)
    assert env.steps == 6 and env.resets == 2 and env.closed
    ending = next(x for x in finished if x[0] == 5)
    if not truncated:
        assert ending[1:] == (0., 0.)
    else:
        assert ending[1:] != (0., 0.)
    rows = list(csv.DictReader((tmp_path / 'progress.txt').open(), delimiter='\t'))
    assert [int(row['Episodes']) for row in rows] == [0, 1]
    assert float(rows[1]['EpRisk']) == 10
    assert float(rows[1]['Penalty']) > float(rows[0]['Penalty'])
    assert all(np.isfinite(float(row['LossPi'])) for row in rows)
    saved = torch.load(tmp_path / 'pyt_save/weights1.pt', weights_only=True)
    assert any(not torch.equal(saved[k], before[0][k]) for k in saved)
    for key in saved:
        torch.testing.assert_close(saved[key], actors[0].state_dict()[key])


def test_deterministic_discrete_policy():
    env = gym.make('CartPole-v1')
    model = MLPActorCritic(env.observation_space, env.action_space)
    obs = torch.zeros(4)
    assert model.act(obs, True).item() in (0, 1)
    assert model.act(obs, True) == model.act(obs, True)
    env.close()


def test_launcher_gpu_mapping_paths_and_secrets():
    import importlib.util
    spec = importlib.util.spec_from_file_location('launcher', ROOT / 'scripts/run_training.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    command, env, dry = module.build_command(['-a', 'fuzppolag', '-e', 'Pendulum-v1', '-g', '3',
        '--dry-run', '--swanlab-api-key', 'test-secret', '--train-eps', '0.2'])
    assert env['CUDA_VISIBLE_DEVICES'] == '3'
    assert command[command.index('--device') + 1] == 'cuda:0'
    assert 'test-secret' not in command and env['SWANLAB_API_KEY'] == 'test-secret'
    assert '--train_eps' in command and Path(command[1]).is_absolute() and dry


def test_evaluation_cli_roundtrip(tmp_path):
    run = tmp_path / 'run'
    (run / 'pyt_save').mkdir(parents=True)
    env = gym.make('Pendulum-v1')
    model = MLPActorCritic(env.observation_space, env.action_space, hidden_sizes=(8,))
    torch.save(model.state_dict(), run / 'pyt_save/weights0.pt')
    (run / 'config.json').write_text(json.dumps({'env_name': 'Pendulum-v1', 'ac_kwargs': {'hidden_sizes': [8]}}))
    output = tmp_path / 'report.json'
    cmd = [sys.executable, str(ROOT / 'scripts/evaluate.py'), str(run), '--noise', '0', '0.1',
           '--seeds', '4', '5', '--max-steps', '5', '--output', str(output)]
    subprocess.run(cmd, check=True, capture_output=True)
    first = output.read_text()
    subprocess.run(cmd, check=True, capture_output=True)
    assert output.read_text() == first
    report = json.loads(first)
    assert len(report['results']) == 2
    assert all(e['cutoff'] for r in report['results'] for e in r['episodes'])
    env.close()


def test_mpi_parameter_and_scalar_gradient_copy(monkeypatch):
    import spinup.utils.mpi_pytorch as mpi
    monkeypatch.setattr(mpi, 'num_procs', lambda: 2)
    monkeypatch.setattr(mpi, 'mpi_avg', lambda x: np.asarray(x) * 0.5)
    p = torch.tensor(1., requires_grad=True)
    (p * 6).backward()
    mpi.mpi_avg_grads(p)
    assert p.grad.item() == 3
    model = torch.nn.Linear(2, 1)
    monkeypatch.setattr(mpi, 'broadcast', lambda x: x.fill(7))
    mpi.sync_params(model)
    assert all(torch.all(p == 7) for p in model.parameters())


def test_single_process_mpi_statistics():
    from spinup.utils.mpi_tools import mpi_statistics_scalar
    assert mpi_statistics_scalar([2, 4], with_min_and_max=True) == (3, 1, 2, 4)
    assert np.isnan(mpi_statistics_scalar([])[0])
