import argparse
import os
from functools import partial
from pathlib import Path

import jax
import numpy as np

# jax.distributed.initialize()  # type: ignore
from heidelberg_v02_more_metrics import load_datasets, run_theta_ensemble
from hyperparam_scan_util import GridScan, computed, vary

assert (
    Path.cwd().as_posix().endswith("experiments/heidelberg")
), f"Unexpected cwd: {os.getcwd()}"

devices = jax.local_devices()
print(f"Local devices: {devices}")


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
    "tau": vary(
        # 25 points between 2^-4 = 0.0625 and 2^8 = 256
        # (13 points at exact powers of 2, 12 points in between)
        *np.logspace(-4, 8, num=13, base=2)
    ),
    # "tau": 1,
    "I0": 5 / 4,
    "eps": 1e-6,
    # Network
    "Nin_virtual": 16,  # #Virtual input neurons = N_bin - 1
    "Nhidden": 128,
    "Nlayer": vary(2, 3),  # Number of layers (hidden layers + output layer)
    "Nout": 20,
    "w_scale": 0.5,  # Scaling factor of initial weights
    # Trial
    "T": vary(
        # 25 points between 2^-4 = 0.0625 and 2^8 = 256
        # (13 points at exact powers of 2, 12 points in between)
        *np.logspace(-4, 8, num=13, base=2)
    ),
    # "T": 4,
    "K": 700,  # Maximal number of simulated ordinary spikes
    # "Kin":   # Maximal number of input spikes
    "dt": 0.001,  # Step size used to compute state traces
    # Training
    "gamma": 1e-2,
    "Nbatch": 1000,
    "lr": 4e-3,
    "tau_lr": 1e2,
    "beta1": 0.9,
    "beta2": 0.999,
    "p_flip": 0.0,
    "Nepochs": 50,
    "Ntrain": None,  # Number of training samples
    # SHD Quantization
    # "Nt": vary(2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 40, 48, 64, 80, 96, 128),
    # "Nin_data": 700,
    # "Nin": computed(lambda Nin_data, Nt: Nin_data * Nt),
    "Nin": 700,
    # Ensemble
    "Nsamples": 3,
    # Data transformation
    "normalize_times": True,
    # Output function
    "out_func": "max_over_time_potential",
    # Readout layer params (e.g. used by max_over_time_potential)
    "readout_V0": vary(64, 128, 256),
    "readout_w": computed(lambda readout_V0: readout_V0),
}

scan = GridScan.load_or_create("main_v2.1_max_over_time", root="results")

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--id", type=str)
arg_parser.add_argument("--n-jobs", type=int)
arg_parser.add_argument("--job-index", type=int)
arg_parser.add_argument("--preview", action="store_true")
arg_parser.add_argument("--with-params", action="store_true")
args = arg_parser.parse_args()

datasets = load_datasets("data", verbose=True) if not args.preview else None

scan.run(
    partial(
        run_theta_ensemble,
        datasets,
        progress_bar="script",
        return_params=args.with_params,
    ),
    config_grid,
    show_metrics=(
        "loss_final_mean",
        "activity_final_mean",
        "acc_max_epoch",
        "acc_max_mean",
        "acc_max_std",
        "acc_ord_max_epoch",
        "acc_ord_max_mean",
        "acc_ord_max_std",
    ),
    base_id=args.id,
    n_jobs=args.n_jobs,
    job_index=args.job_index,
    preview=args.preview,
)
