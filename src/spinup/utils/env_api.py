"""Normalize Gym, Gymnasium and Safety-Gymnasium without probing the simulator."""
import numpy as np


def reset_env(env, seed=None):
    if seed is not None:
        env.action_space.seed(seed)
        try:
            result = env.reset(seed=seed)
        except TypeError as exc:
            if "seed" not in str(exc) or not hasattr(env, "seed"):
                raise
            env.seed(seed)
            result = env.reset()
    else:
        result = env.reset()
    return result[0] if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict) else result


def unpack_step(result):
    if len(result) == 6:
        obs, reward, cost, terminated, truncated, info = result
    elif len(result) in (4, 5):
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
        else:
            obs, reward, done, info = result
            truncated = bool(info.get("TimeLimit.truncated", False))
            terminated = bool(done) and not truncated
        cost = info.get("cost", info.get("constraint_violation", 0.0))
    else:
        raise ValueError("Expected 4, 5 or 6 values from environment.step")
    if not np.isfinite(reward) or not np.isfinite(cost) or not np.isfinite(obs).all():
        raise ValueError("Environment returned non-finite observation, reward or cost")
    return obs, float(reward), float(cost), bool(terminated), bool(truncated), info


def step_env(env, action):
    if hasattr(env.action_space, "low"):
        action = np.clip(action, env.action_space.low, env.action_space.high)
    else:
        action = int(np.asarray(action).item())
    return unpack_step(env.step(action))
