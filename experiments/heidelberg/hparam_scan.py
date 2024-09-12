import os
from functools import partial
from pathlib import Path

import jax

# jax.distributed.initialize()  # type: ignore
import numpy as np

from heidelberg_v01 import load_datasets, run_theta_ensemble
from hyperparam_scan_util import GridScan, computed, vary

assert (
    Path.cwd().as_posix().endswith("experiments/heidelberg")
), f"Unexpected cwd: {os.getcwd()}"

print(f"Total device count: {jax.device_count()}")
print(f"Device count per task: {jax.local_device_count()}")
devices = jax.local_devices()
print(f"Local devices: {devices}")

datasets = load_datasets("data", verbose=True)

config_grid = {
    # "devices": [
    #     {
    #         "id": d.id,
    #         "kind": d.device_kind,
    #         "platform": d.platform,
    #         "memory_stats": d.memory_stats(),
    #     }
    #     for d in devices
    # ],
    "device_count": len(devices),
    "seed": 0,
    # Neuron
    "tau": 6 / np.pi,
    "I0": 5 / 4,
    "eps": 1e-6,
    # Network
    "Nin_virtual": vary(1, 2, 4, 8, 12, 16, 24),  # #Virtual input neurons = N_bin - 1
    "Nhidden": 100,
    "Nlayer": 2,  # Number of layers
    "Nout": 20,
    "w_scale": 0.5,  # Scaling factor of initial weights
    # Trial
    "T": 2.0,
    "K": 300,  # Maximal number of simulated ordinary spikes
    "dt": 0.001,  # Step size used to compute state traces
    # Training
    "gamma": 1e-2,
    "Nbatch": 500,
    "lr": 4e-3,
    "tau_lr": 1e2,
    "beta1": 0.9,
    "beta2": 0.999,
    "p_flip": 0.0,
    "Nepochs": 20,
    "Ntrain": None,  # Number of training samples
    # SHD Quantization
    "Nt": vary(2, 4, 6, 8, 10, 12, 14, 16, 18),
    "Nin_data": 700,
    "Nin": computed(lambda Nin_data, Nt: Nin_data * Nt),
    # Ensemble
    "Nsamples": 3,
}

scan = GridScan.load_or_create("main", root="results")


scan.run(
    partial(run_theta_ensemble, datasets, progress_bar=None),
    config_grid,
    show_metrics=("acc_max_epoch", "acc_max_mean", "acc_max_std"),
    if_trial_exists="recompute_if_error",
    n_processes=1,
)
# print(scan.load_trials())
