"""Seeded disturbances with explicit observation, action and MuJoCo dynamics modes."""
import numpy as np


class EnvironmentNoiseWrapper:
    def __init__(self, env, train_eps=1.0, disturb_part="observation", disturb_type="white_noise"):
        if not np.isfinite(train_eps) or train_eps < 0:
            raise ValueError("train_eps must be finite and nonnegative")
        if disturb_part not in {"observation", "action", "dynamics", "multi_source"}:
            raise ValueError("Unknown disturbance target")
        if disturb_type not in {"white_noise", "impulse", "periodic"}:
            raise ValueError("Unknown disturbance type")
        if disturb_part in {"action", "multi_source"} and not hasattr(env.action_space, "low"):
            raise ValueError("Action disturbances require a continuous Box action space")
        self.env, self.train_eps = env, train_eps
        self.disturb_part, self.disturb_type = disturb_part, disturb_type
        self.rng = np.random.default_rng()
        self.step_count = 0
        self._physics = None
        if disturb_part in {"dynamics", "multi_source"}:
            base = env.unwrapped
            self._physics = getattr(base, "task", base)
            if not hasattr(self._physics, "data") or not hasattr(self._physics, "model"):
                raise ValueError("Dynamics disturbances require MuJoCo model/data; use observation or action noise")

    def _noise(self, shape):
        if self.disturb_type == "white_noise":
            return self.rng.normal(0, self.train_eps, size=shape)
        if self.disturb_type == "impulse":
            elapsed = self.step_count - 20
            scale = self.train_eps * 10 * 0.9 ** elapsed if 0 <= elapsed < 80 else 0
            return self.rng.normal(0, scale, size=shape)
        return np.full(shape, self.train_eps * np.sin(2 * np.pi * self.step_count / 1000))

    def _observation(self, obs):
        if self.disturb_part in {"observation", "multi_source"}:
            obs = np.asarray(obs)
            return (obs + self._noise(obs.shape)).astype(obs.dtype)
        return obs

    def reset(self, **kwargs):
        if kwargs.get("seed") is not None:
            self.rng = np.random.default_rng(kwargs["seed"])
        self.step_count = 0
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
            return self._observation(result[0]), result[1]
        return self._observation(result)

    def step(self, action):
        self.step_count += 1
        if self.disturb_part in {"action", "multi_source"}:
            action = np.asarray(action)
            action = np.clip(action + self._noise(action.shape), self.action_space.low, self.action_space.high)
        if self._physics is not None:
            import mujoco
            data = self._physics.data
            data.qvel[:] += self._noise(data.qvel.shape)
            mujoco.mj_forward(self._physics.model, data)
        result = self.env.step(action)
        return (self._observation(result[0]), *result[1:])

    def __getattr__(self, name):
        # Avoid recursion during unpickling, before env has been restored.
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)
