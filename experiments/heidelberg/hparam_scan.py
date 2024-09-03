from functools import partial

import jax
import numpy as np
from heidelberg_v01 import load_datasets, run_theta_ensemble
from hyperparam_scan_util import computed, scan_grid, vary

devices = jax.devices()
assert (
    len(devices) == 1 and devices[0].device_kind == "gpu"
), f"Expected a single GPU, got {devices}"

datasets = load_datasets("data", verbose=True)


config_grid = {
    "seed": 0,
    # Neuron
    "tau": 6 / np.pi,
    "I0": 5 / 4,
    "eps": 1e-6,
    # Network
    "Nin_virtual": vary(12, 16, 20),  # #Virtual input neurons = N_bin - 1
    "Nhidden": vary(40, 60, 80, 100),
    "Nlayer": vary(2, 3),  # Number of layers
    "Nout": 20,
    "w_scale": 0.5,  # Scaling factor of initial weights
    # Trial
    "T": 2.0,
    "K": vary(50, 100, 150, 200),  # Maximal number of simulated ordinary spikes
    "dt": 0.001,  # Step size used to compute state traces
    # Training
    "gamma": 1e-2,
    "Nbatch": 1000,
    "lr": 4e-3,
    "tau_lr": 1e2,
    "beta1": 0.9,
    "beta2": 0.999,
    "p_flip": vary(0.0, 0.02, 0.04),
    "Nepochs": 10,
    "Ntrain": None,  # Number of training samples
    # SHD Quantization
    "Nt": vary(8, 12, 16),
    "Nin_data": 700,
    "Nin": computed(lambda Nin_data, Nt: Nin_data * Nt),
    # Ensemble
    "Nsamples": 3,
}

scan_grid(
    partial(run_theta_ensemble, datasets),
    config_grid,
    version=1,
    show_metrics=("acc_max_epoch", "acc_max_mean", "acc_max_std"),
    if_trial_exists="recompute_if_error",
)
