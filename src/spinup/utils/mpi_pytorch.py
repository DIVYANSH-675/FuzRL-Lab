import multiprocessing
import numpy as np
import os
import torch
# from mpi4py import MPI
from spinup.utils.mpi_tools import broadcast, mpi_avg, num_procs, proc_id

def setup_pytorch_for_mpi():
    """
    Avoid slowdowns caused by each separate process's PyTorch using
    more than its fair share of CPU resources.
    """
    #print('Proc %d: Reporting original number of Torch threads as %d.'%(proc_id(), torch.get_num_threads()), flush=True)
    if torch.get_num_threads()==1:
        return
    fair_num_threads = max(int(torch.get_num_threads() / num_procs()), 1)
    torch.set_num_threads(fair_num_threads)
    #print('Proc %d: Reporting new number of Torch threads as %d.'%(proc_id(), torch.get_num_threads()), flush=True)

def mpi_avg_grads(module):
    """ Average contents of gradient buffers across MPI processes. """
    if num_procs()==1:
        return

    # Handle both modules (with .parameters()) and individual parameters
    if hasattr(module, 'parameters'):
        # It's a module
        params = module.parameters()
    else:
        # It's a single parameter or iterable of parameters
        params = [module] if hasattr(module, 'grad') else module

    for p in params:
        if p.grad is not None:
            averaged = mpi_avg(p.grad.detach().cpu().numpy())
            p.grad.copy_(torch.as_tensor(averaged, dtype=p.grad.dtype, device=p.grad.device))


def sync_params(module):
    """Broadcast parameters and copy results back to their original device."""
    if num_procs() == 1:
        return
    for p in module.parameters():
        values = p.detach().cpu().numpy().copy()
        broadcast(values)
        with torch.no_grad():
            p.copy_(torch.as_tensor(values, dtype=p.dtype, device=p.device))
