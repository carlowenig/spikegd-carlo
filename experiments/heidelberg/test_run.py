import os
import pickle
from pathlib import Path

import jax
import yaml

# jax.distributed.initialize()  # type: ignore
from heidelberg_v02_more_metrics import load_datasets, run_theta_ensemble
from spikegd.utils.misc import standardize_value

assert (
    Path.cwd().as_posix().endswith("experiments/heidelberg")
), f"Unexpected cwd: {os.getcwd()}"

devices = jax.local_devices()
print(f"Local devices: {devices}")


name = "v2.1_max_over_time_2_test"

# EITHER DEFINE A CONFIG MANUALLY
config = {
    "device_count": len(devices),
    "seed": 0,
    # Neuron
    "tau": 1,
    "I0": 5 / 4,
    "eps": 1e-6,
    # Network
    "Nin_virtual": 16,  # #Virtual input neurons = N_bin - 1
    "Nhidden": 128,
    "Nlayer": 3,  # Number of layers (hidden layers + output layer)
    "Nout": 20,
    "w_scale": 0.5,  # Scaling factor of initial weights
    # Trial
    "T": 2,
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
    "Nepochs": 10,
    "Ntrain": None,  # Number of training samples
    # SHD Quantization
    # "Nt": 10,
    # "Nin_data": 700,
    # "Nin": computed(lambda Nin_data, Nt: Nin_data * Nt),
    "Nin": 700,
    # Ensemble
    "Nsamples": 1,
    # Data transformation
    "normalize_times": True,
    # Output function
    "out_func": "max_over_time_potential",
    # Readout layer params (e.g. used by max_over_time_potential)
    "readout_V0": 0,
    "readout_w_factor": 1,
    "readout_tau_factor": 1,
    "readout_I0": 0,
}

# ...OR LOAD A CONFIG
# scan = GridScan.load("main_v2.1", root="results")
# trial = scan.load_trial("d843e4f7e1f26c0038cbb40204f8c6a7")
# print(trial)
# config = trial.config.copy()
# config["Nepochs"] = 100
# config["Nsamples"] = 5

datasets = load_datasets("data", verbose=True)

metrics = run_theta_ensemble(
    datasets, config, progress_bar="script", return_params=True
)
params = metrics.pop("params")

result_dict = {"config": config, "metrics": metrics}

path = Path("results/test_runs") / name
path.mkdir(parents=True, exist_ok=True)

with open(path / "results.yaml", "w") as f:
    yaml.dump(standardize_value(result_dict), f, sort_keys=False)

with open(path / "params.pickle", "wb") as f:
    pickle.dump(params, f)
