"""Shared numerical and constraint-update helpers."""
import numpy as np
import torch
from spinup.utils.mpi_tools import mpi_statistics_scalar


def normalize_advantages(values):
    if not np.isfinite(values).all():
        raise ValueError("Advantages contain non-finite values")
    mean, std = mpi_statistics_scalar(values)
    return (values - mean) / max(float(std), 1e-8)


def penalty_loss(parameter, cost_deviation):
    """Gradient descent must increase the multiplier when the budget is exceeded."""
    return -torch.nn.functional.softplus(parameter) * cost_deviation


def validate_training(steps, epochs, pi_iters, v_iters, max_ep_len, save_freq, workers):
    for name, value in dict(steps=steps, epochs=epochs, pi_iters=pi_iters,
                            v_iters=v_iters, max_ep_len=max_ep_len, save_freq=save_freq).items():
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if steps % workers:
        raise ValueError("steps_per_epoch must be divisible by the number of MPI workers")
