import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from jax import jit, random, value_and_grad, vmap
from jaxtyping import Array, ArrayLike, Float, Int
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from shd import SHD
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import trange as trange_script
from tqdm.notebook import trange as trange_notebook

from spikegd.models import AbstractPhaseOscNeuron, AbstractPseudoPhaseOscNeuron
from spikegd.utils.plotting import formatter, petroff10

# %%
############################
### Data loading
############################


def normalize_times(times, neurons, dt, t_max=1, signal_percentage=0.05):
    N_signal = int(len(times) * signal_percentage)

    start = times[N_signal] - dt
    end = times[-N_signal] + dt
    new_times = (times - start) / (end - start) * t_max
    mask = (new_times >= 0) & (new_times < t_max)
    return new_times[mask], neurons[mask]


# def quantize_dataset(
#     dataset: SHD, config: dict, dtype=torch.int8, normalize_trials=True
# ):
#     sample_indices = range(dataset.Nsamples)
#     N_bin = config["Nin_virtual"] + 1

#     if N_bin > torch.iinfo(dtype).max:
#         raise ValueError(
#             f"Number of bins N_bin = {N_bin} exceeds maximum value for {dtype}: {torch.iinfo(dtype).max}"
#         )

#     N_t = config["Nt"]
#     t_max = config.get("tmax", 1 if normalize_trials else dataset.t_max + 1e-6)

#     N_in = dataset.N  # Number of input neurons
#     x_arr = np.zeros((len(sample_indices), N_in * N_t), dtype=np.int32)
#     dt = t_max / N_t

#     bin_edges = np.arange(0, t_max + dt / 2, dt)
#     assert bin_edges.shape == (
#         N_t + 1,
#     ), f"Expected bin_edges to have shape ({N_t + 1}), got {bin_edges.shape}"

#     max_count = 0

#     for i, sample_index in enumerate(sample_indices):
#         times = dataset.times_arr[sample_index]
#         neurons = dataset.units_arr[sample_index]

#         if normalize_trials:
#             times, neurons = normalize_times(times, neurons, dt, t_max)

#         # Use numpy's digitize to bin the spike times
#         binned_spikes = np.digitize(times, bin_edges) - 1

#         # Count spikes in each bin for each neuron
#         count_arr = np.bincount(
#             neurons * N_t + binned_spikes, minlength=N_in * N_t
#         ).reshape(N_in, N_t)

#         count_step = config.get("spike_count_step", 1)
#         count_offset = config.get("spike_count_offset", 0)
#         count_arr = count_arr // count_step + count_offset

#         # Apply the quantization
#         x_arr[i] = np.maximum(N_bin - 1 - count_arr, 0).ravel()

#         max_count = max(max_count, count_arr.max())

#     if config.get("verbose", False):  # and abs(max_count - N_bin) > 0:
#         print(
#             f"Max spike count in single time interval ({max_count}) "
#             f"is very different to what is allowed by N_bin ({N_bin})"
#         )

#     # Create tensor dataset
#     data = torch.as_tensor(x_arr, dtype=dtype)
#     targets = torch.as_tensor(dataset.label_index_arr, dtype=torch.int8)

#     # print("Quantized data memory:", data.element_size() * data.numel() / 1e6, "MB")

#     assert data.shape[:1] == targets.shape == (dataset.Nsamples,)

#     return TensorDataset(data, targets)


def homogenize_dataset(dataset: SHD, config: dict, normalize_trials=True):
    N = dataset.Nsamples
    Kin = config["Kin"]  # max number of input spikes per neuron
    Nin = config["Nin"]
    T = config["T"]

    # Array containing the spike times or T if no spike
    # lost_spikes = {}
    times_arr = np.full((N, Kin), T, dtype=float)
    neurons_arr = np.full((N, Kin), -1, dtype=int)

    Nlost_norm_vals = []
    Nlost_cap_vals = []

    print(
        "Mean number of spikes per trial:",
        np.mean([len(times) for times in dataset.times_arr]),
    )
    print(
        "Max number of spikes per trial:",
        max(len(times) for times in dataset.times_arr),
    )
    print("Kin:", Kin)

    for i in trange_script(N):
        times = dataset.times_arr[i]
        neurons = dataset.units_arr[i]
        assert len(times) == len(neurons)

        Nspike_total = len(times)

        if normalize_trials:
            times, neurons = normalize_times(times, neurons, 0)

        Nlost_norm_vals.append(Nspike_total - len(times))
        Nspike_total = len(times)

        # remove spikes over Kin
        times = times[:Kin]
        neurons = neurons[:Kin]

        Nlost_cap_vals.append(Nspike_total - len(times))

        times_arr[i, : len(times)] = times
        neurons_arr[i, : len(neurons)] = neurons

        # for j in range(Nin):
        #     spike_times = times[neurons == j]
        #     Nspike = min(Kin, len(spike_times))
        #     x_arr[i, j, :Nspike] = spike_times[:Nspike]
        #     if (Nlost := len(spike_times) - Nspike) > 0:
        #         lost_spikes[j] = lost_spikes.get(j, 0) + Nlost

    print("Mean lost spikes due to normalization:", np.mean(Nlost_norm_vals))
    print("Mean lost spikes due to Kin-cap:", np.mean(Nlost_cap_vals))

    # Create tensor dataset
    times_tensor = torch.as_tensor(times_arr)
    neurons_tensor = torch.as_tensor(neurons_arr)
    targets = torch.as_tensor(dataset.label_index_arr, dtype=torch.int8)

    return TensorDataset(times_tensor, neurons_tensor, targets)


def load_datasets(root: str, verbose: bool = False) -> tuple[SHD, SHD]:
    train_set = SHD(root, mode="train", download=False, verbose=verbose)
    test_set = SHD(root, mode="test", download=False, verbose=verbose)
    return train_set, test_set


def load_data(datasets: tuple[SHD, SHD], config: dict) -> tuple[DataLoader, DataLoader]:
    """
    Creates DataLoaders.
    """
    Nbatch: int = config["Nbatch"]
    Nin: int = config["Nin"]
    Nout: int = config["Nout"]
    # Nt: int = config["Nt"]

    train_set, test_set = datasets

    torch.manual_seed(config["seed"])

    # Training set
    assert Nin == train_set.N, (
        f"config.Nin does not match number of neurons in train dataset ({train_set.N}). "
        f"Got {Nin}, expected {train_set.N}."
    )
    assert Nout == train_set.Nlabel, (
        f"config.Nout does not match number of labels in train dataset."
        f"Got {Nout}, expected {train_set.Nlabel}."
    )

    # print("Quantizing training set...")
    train_set = homogenize_dataset(train_set, config)
    # print("Train set size:", len(train_set))

    if (Ntrain := config.get("Ntrain")) is not None:
        train_set = Subset(train_set, np.arange(Ntrain))
        # print(f"Using reduced train_set size of {Ntrain}")

    train_loader = DataLoader(train_set, batch_size=Nbatch, shuffle=True)

    # Test set
    assert Nin == test_set.N, (
        f"config.Nin does not match number of neurons in test dataset ({test_set.N}). "
        f"Got {Nin}, expected {test_set.N}."
    )
    assert Nout == test_set.Nlabel, (
        f"config.Nout does not match number of labels in test dataset."
        f"Got {Nout}, expected {test_set.Nlabel}."
    )
    # print("Quantizing test set...")
    test_set = homogenize_dataset(test_set, config)
    # print("Test set size:", len(test_set))

    test_loader = DataLoader(test_set, batch_size=1_000, shuffle=True)

    return train_loader, test_loader


# %%
############################
### Initialization
############################


def init_weights(key: Array, config: dict) -> tuple[Array, list]:
    """
    Initializes input and network weights.
    """
    ### Unpack arguments
    Nin: int = config["Nin"]
    Nhidden: int = config["Nhidden"]
    Nlayer: int = config["Nlayer"]
    Nout: int = config["Nout"]
    w_scale: float = config["w_scale"]

    ### Initialize weights
    key, subkey = random.split(key)
    weights = []
    width = w_scale / jnp.sqrt(Nin)
    weights_in = random.uniform(subkey, (Nhidden, Nin), minval=-width, maxval=width)
    weights.append(weights_in)
    width = w_scale / jnp.sqrt(Nhidden)
    for _ in range(1, Nlayer - 1):
        key, subkey = random.split(key)
        weights_hidden = random.uniform(
            subkey, (Nhidden, Nhidden), minval=-width, maxval=width
        )
        weights.append(weights_hidden)
    key, subkey = random.split(key)
    weights_out = random.uniform(subkey, (Nout, Nhidden), minval=-width, maxval=width)
    weights.append(weights_out)

    return key, weights


def init_phi0(neuron: AbstractPhaseOscNeuron, config: dict) -> Array:
    """
    Initializes initial phase of neurons.
    """
    ### Unpack arguments
    Nhidden: int = config["Nhidden"]
    Nlayer: int = config["Nlayer"]
    Nout: int = config["Nout"]
    N = Nhidden * (Nlayer - 1) + Nout
    theta = neuron.Theta()

    ### Initialize initial phase
    phi0 = theta / 2 * jnp.ones(N)
    return phi0


# %%
############################
### Model
############################


def eventffwd(
    neuron: AbstractPhaseOscNeuron,
    p: list,
    times_in: Float[Array, " Nspikes"],
    neurons_in: Int[Array, " Nspikes"],
    config: dict,
) -> tuple:
    """
    Simulates a feedforward network with time-to-first-spike input encoding.
    """
    ### Unpack arguments
    Kin: int = config["Kin"]
    Nin: int = config["Nin"]
    Nhidden: int = config["Nhidden"]
    Nlayer: int = config["Nlayer"]  # currently has to be at least 2 (1hidden)
    Nout: int = config["Nout"]
    N = Nhidden * (Nlayer - 1) + Nout
    weights: list = p[0]
    phi0: Array = p[1]
    x0 = phi0[jnp.newaxis]

    ### Input
    # Times computed as spike times of a LIF neuron with constant input Iconst
    # I_const = jnp.arange(Nin_virtual, 0, -1)
    # V_th = 0.01 * Nin_virtual
    # times_in = jnp.where(
    #     I_const > V_th, T * jnp.log(I_const / (I_const - V_th)), jnp.inf
    # )
    # neurons_in = jnp.arange(Nin_virtual)
    # spikes_in = (times_in, neurons_in)

    # input = [[t_00, t_01, t_03], [t_10, T, T], [t_20, t_21, T]]
    # => times_in [t_00, t_01, t_03, t_10, T, T, t_20, t_21, T]
    # (first index: neuron, second index: spike)
    # times_in = input.ravel()

    # neurons_in = jnp.tile(jnp.arange(Nin), Kin)
    # assert len(times_in) == len(neurons_in)
    # assert neurons_in[:Nin].all() == 0
    # spikes_in = (times_in, neurons_in)
    spikes_in = (times_in, neurons_in)

    ### Input weights
    weights_in = jnp.zeros((N, Nin))
    weights_in = weights_in.at[:Nhidden].set(weights[0])

    # weights_in_virtual = jnp.zeros((N, Nin_virtual))
    # virt_indices = jnp.arange(Nin_virtual)
    # indicator_matrix = input[:, jnp.newaxis] == virt_indices
    # weights_in_virtual = weights_in_virtual.at[:Nhidden].set(
    #     weights_in @ indicator_matrix
    # )

    ### Network weights
    weights_net = jnp.zeros((N, N))
    for i in range(Nlayer - 2):
        slice_in = slice(i * Nhidden, (i + 1) * Nhidden)
        slice_out = slice((i + 1) * Nhidden, (i + 2) * Nhidden)
        weights_net = weights_net.at[slice_out, slice_in].set(weights[i + 1])
    weights_net = weights_net.at[N - Nout :, N - Nout - Nhidden : N - Nout].set(
        weights[-1]
    )

    # Run simulation
    out = neuron.event(x0, weights_net, weights_in, spikes_in, config)

    return out


def outfn(
    neuron: AbstractPseudoPhaseOscNeuron, out: tuple, p: list, config: dict
) -> Array:
    """
    Computes output spike times given simulation results.
    """
    ### Unpack arguments
    Nin: int = config["Nin"]
    Nhidden: int = config["Nhidden"]
    Nlayer: int = config["Nlayer"]
    Nout: int = config["Nout"]
    N = Nhidden * (Nlayer - 1) + Nout
    weights = p[0]
    times: Array = out[0]
    spike_in: Array = out[1]
    neurons: Array = out[2]
    x: Array = out[3]

    ### Run network as feedforward rate ANN
    Kord = jnp.sum(neurons >= 0)  # Number of ordinary spikes
    x_end = x[Kord]
    pseudo_rates = jnp.zeros(Nin)
    for i in range(Nlayer - 1):
        input = neuron.linear(pseudo_rates, weights[i])
        x_end_i = x_end[:, i * Nhidden : (i + 1) * Nhidden]
        pseudo_rates = neuron.construct_ratefn(x_end_i)(input)
    input = neuron.linear(pseudo_rates, weights[Nlayer - 1])

    ### Spike times for each learned neuron
    def compute_tout(i: ArrayLike) -> Array:
        ### Potential ordinary output spike times
        mask = (neurons == N - Nout + i) & (spike_in == False)  # noqa: E712
        Kout = jnp.sum(mask)  # Number of ordinary output spikes
        t_out_ord = times[jnp.argmax(mask)]

        ### Pseudospike time
        t_out_pseudo = neuron.t_pseudo(x_end[:, N - Nout + i], input[i], 1, config)

        ### Output spike time
        t_out = jnp.where(0 < Kout, t_out_ord, t_out_pseudo)

        return t_out

    t_outs = vmap(compute_tout)(jnp.arange(Nout))

    return t_outs


def lossfn(
    t_outs: Float[Array, " Nout"], label: Int[Array, ""], config: dict
) -> tuple[Array, Array]:
    """
    Compute cross entropy loss and if the network prediction was correct.
    """
    T: float = config["T"]
    gamma: float = config["gamma"]
    log_softmax = jax.nn.log_softmax(-t_outs)
    regu = jnp.exp(t_outs / T) - 1
    loss = -log_softmax[label] + gamma * regu[label]
    correct = jnp.argmin(t_outs) == label
    return loss, correct


def simulatefn(
    neuron: AbstractPseudoPhaseOscNeuron,
    p: list,
    input: tuple[Float[Array, "Batch Nspikes"], Int[Array, "Batch Nspikes"]],
    labels: Int[Array, " Batch"],
    config: dict,
) -> tuple[Array, Array]:
    """
    Simulates the network and computes the loss and accuracy for batched input.
    """
    times_in, neurons_in = input
    outs = vmap(eventffwd, in_axes=(None, None, 0, 0, None))(
        neuron, p, times_in, neurons_in, config
    )
    t_outs = vmap(outfn, in_axes=(None, 0, None, None))(neuron, outs, p, config)
    loss, correct = vmap(lossfn, in_axes=(0, 0, None))(t_outs, labels, config)
    mean_loss = jnp.mean(loss)
    accuracy = jnp.mean(correct)
    return mean_loss, accuracy


def probefn(
    neuron: AbstractPseudoPhaseOscNeuron,
    p: list,
    input: tuple[Float[Array, "Batch Nspikes"], Int[Array, "Batch Nspikes"]],
    labels: Int[Array, " Batch"],
    config: dict,
) -> tuple:
    """
    Computes several metrics.
    """

    ### Unpack arguments
    T: float = config["T"]
    Nhidden: int = config["Nhidden"]
    Nlayer: int = config["Nlayer"]
    Nout: int = config["Nout"]
    N = Nhidden * (Nlayer - 1) + Nout
    Nbatch: int = config["Nbatch"]

    ### Batched functions
    @vmap
    def batch_eventffwd(times_in, neurons_in):
        return eventffwd(neuron, p, times_in, neurons_in, config)

    @vmap
    def batch_outfn(outs):
        return outfn(neuron, outs, p, config)

    @vmap
    def batch_lossfn(t_outs, labels):
        return lossfn(t_outs, labels, config)

    ### Run network
    times_in, neurons_in = input
    outs = batch_eventffwd(times_in, neurons_in)
    times: Array = outs[0]
    spike_in: Array = outs[1]
    neurons: Array = outs[2]
    t_outs = batch_outfn(outs)

    ### Loss and accuracy with pseudospikes
    loss, correct = batch_lossfn(t_outs, labels)
    mean_loss = jnp.mean(loss)
    acc = jnp.mean(correct)

    ### Loss and accuracy without pseudospikes
    t_out_ord = jnp.where(t_outs < T, t_outs, T)
    loss_ord, correct_ord = batch_lossfn(t_out_ord, labels)
    mean_loss_ord = jnp.mean(loss_ord)
    acc_ord = jnp.mean(correct_ord)

    ### Activity and silent neurons
    mask = (spike_in == False) & (neurons < N - Nout) & (neurons >= 0)  # noqa: E712
    activity = jnp.sum(mask) / (Nbatch * (N - Nout))
    silent_neurons = jnp.isin(
        jnp.arange(N - Nout), jnp.where(mask, neurons, -1), invert=True
    )

    ### Activity and silent neurons until first output spike
    t_out_first = jnp.min(t_out_ord, axis=1)
    mask = (
        (spike_in == False)  # noqa: E712
        & (neurons < N - Nout)
        & (neurons >= 0)
        & (times < t_out_first[:, jnp.newaxis])
    )
    activity_first = jnp.sum(mask) / (Nbatch * (N - Nout))
    silent_neurons_first = jnp.isin(
        jnp.arange(N - Nout), jnp.where(mask, neurons, -1), invert=True
    )

    ### Pack results in dictionary
    metrics = {
        "loss": mean_loss,
        "acc": acc,
        "loss_ord": mean_loss_ord,
        "acc_ord": acc_ord,
        "activity": activity,
        "activity_first": activity_first,
    }
    silents = {
        "silent_neurons": silent_neurons,
        "silent_neurons_first": silent_neurons_first,
    }

    return metrics, silents


# %%
############################
### Training
############################


def run(
    neuron: AbstractPseudoPhaseOscNeuron,
    data_loaders: tuple[DataLoader, DataLoader],
    config: dict,
    progress_bar: str | None = None,
) -> tuple[dict, dict]:
    """
    Trains a feedforward network with time-to-first-spike encoding on MNIST.

    The pixel values are binned into `Nin_virtual+1` bins, each corresponding to an
    input spike time except for the last bin, which is ignored. The effect of all inputs
    in each bin is captured by a virtual input neuron under the hood to speed up the
    simulation. See `transform_image` and `eventffwd` for details. The trained
    parameters `p` are the feedforward weights of the network and the initial phases of
    the neurons.

    Args:
        neuron:
            Phase oscillator model including pseudodynamics.
        config:
            Simulation configuration. Needs to contain the following items:
                `seed`: Random seed
                `Nin`: Number of input neurons, has to be 28*28 for MNIST
                `Nin_virtual`: Number of virtual input neurons
                `Nhidden`: Number of hidden neurons per layer
                `Nlayer`: Number of layers
                `Nout`: Number of output neurons, has to be 10 for MNIST
                `w_scale`: Scale of the initial weights
                `T`: Trial duration
                `K`: Maximal number of simulated ordinary spikes
                `dt`: Integration time step (for state traces)
                `gamma`: Regularization strength
                `Nbatch`: Batch size
                `lr`: Learning rate
                `tau_lr`: Learning rate decay time constant
                `beta1`: Adabelief parameter
                `beta2`: Adabelief parameter
                `p_flip`: Probability of flipping input pixels
                `Nepochs`: Number of epochs
        progress_bar:
            Whether to use 'notebook' or 'script' tqdm progress bar or `None`.
    Returns:
        A dictionary containing detailed learning dynamics.
    """

    init_start = time.perf_counter()

    ### Unpack arguments
    seed: int = config["seed"]
    # Nin_virtual: int = config["Nin_virtual"]
    Nhidden: int = config["Nhidden"]
    Nlayer: int = config["Nlayer"]
    Nout: int = config["Nout"]
    N = Nhidden * (Nlayer - 1) + Nout
    Nepochs: int = config["Nepochs"]
    p_flip: float = config["p_flip"]
    lr: float = config["lr"]
    tau_lr: float = config["tau_lr"]
    beta1: float = config["beta1"]
    beta2: float = config["beta2"]
    theta = neuron.Theta()
    if progress_bar == "notebook":
        trange = trange_notebook
    elif progress_bar == "script":
        trange = trange_script
    else:
        trange = range

    ### Set up the simulation
    init_compile_start = time.perf_counter()

    # Gradient
    @jit
    @partial(value_and_grad, has_aux=True)
    def gradfn(
        p: list,
        input: tuple[Float[Array, "Batch Nspikes"], Int[Array, "Batch Nspikes"]],
        labels: Int[Array, " Batch"],
    ) -> tuple[Array, Array]:
        loss, acc = simulatefn(neuron, p, input, labels, config)
        return loss, acc

    # Regularization
    # @jit
    # def flip(key: Array, input: Array) -> tuple[Array, Array]:
    #     key, subkey = jax.random.split(key)
    #     mask = jax.random.bernoulli(subkey, p=p_flip, shape=input.shape)
    #     return key, jnp.where(mask, Nin_virtual - input, input)

    # Optimization step
    @jit
    def trial(
        p: list,
        input: tuple[Float[Array, "Batch Nspikes"], Int[Array, "Batch Nspikes"]],
        labels: Int[Array, " Batch"],
        opt_state: optax.OptState,
    ) -> tuple:
        (loss, acc), grad = gradfn(p, input, labels)
        updates, opt_state = optim.update(grad, opt_state)
        p = optax.apply_updates(p, updates)  # type: ignore
        p[1] = jnp.clip(p[1], 0, theta)
        return loss, acc, p, opt_state

    # Probe network
    @jit
    def jprobefn(p, input, labels):
        return probefn(neuron, p, input, labels, config)

    init_compile_time = time.perf_counter() - init_compile_start

    def probe(p: list) -> dict:
        metrics = {
            "loss": 0.0,
            "acc": 0.0,
            "loss_ord": 0.0,
            "acc_ord": 0.0,
            "activity": 0.0,
            "activity_first": 0.0,
        }
        silents = {
            "silent_neurons": jnp.ones(N - Nout, dtype=bool),
            "silent_neurons_first": jnp.ones(N - Nout, dtype=bool),
        }
        steps = len(test_loader)
        for data in test_loader:
            times_in, neurons_in, labels = (
                jnp.array(data[0]),
                jnp.array(data[1]),
                jnp.array(data[2]),
            )
            metric, silent = jprobefn(p, (times_in, neurons_in), labels)
            metrics = {k: metrics[k] + metric[k] / steps for k in metrics}
            silents = {k: silents[k] & silent[k] for k in silents}
        for k, v in silents.items():
            metrics[k] = jnp.mean(v).item()
        return metrics

    ### Simulation

    # Data
    train_loader, test_loader = data_loaders

    # Parameters
    key = random.PRNGKey(seed)
    init_weights_start = time.perf_counter()
    key, weights = init_weights(key, config)
    init_weights_time = time.perf_counter() - init_weights_start

    init_phi0_start = time.perf_counter()
    phi0 = init_phi0(neuron, config)
    init_phi0_time = time.perf_counter() - init_phi0_start

    p = [weights, phi0]
    p_init = [weights, phi0]

    # Optimizer
    optim_init_start = time.perf_counter()
    schedule = optax.exponential_decay(lr, int(tau_lr * len(train_loader)), 1 / jnp.e)
    optim = optax.adabelief(schedule, b1=beta1, b2=beta2)
    opt_state = optim.init(p)
    init_optim_time = time.perf_counter() - optim_init_start

    # Metrics
    metrics: dict[str, Array | list] = {k: [v] for k, v in probe(p).items()}

    init_time = time.perf_counter() - init_start

    train_start = time.perf_counter()

    # Training
    for epoch in trange(Nepochs):
        for data in train_loader:
            times_in, neurons_in, labels = (
                jnp.array(data[0]),
                jnp.array(data[1]),
                jnp.array(data[2]),
            )
            # key, input = flip(key, input)
            loss, acc, p, opt_state = trial(
                p, (times_in, neurons_in), labels, opt_state
            )
        # Probe network
        metric = probe(p)
        metrics = {k: v + [metric[k]] for k, v in metrics.items()}

    train_time = time.perf_counter() - train_start

    if jnp.any(jnp.isnan(jnp.array(metrics["loss"]))):
        print(
            "Warning: A NaN appeared. "
            "Likely not enough spikes have been simulated. "
            "Try increasing `K`."
        )

    metrics = {k: jnp.array(v) for k, v in metrics.items()}
    p_end = p
    metrics["p_init"] = p_init
    metrics["p_end"] = p_end

    perf_metrics = {
        "init_time": init_time,
        "init_compile_time": init_compile_time,
        "init_weights_time": init_weights_time,
        "init_phi0_time": init_phi0_time,
        "init_optim_time": init_optim_time,
        "train_time": train_time,
        "epoch_time": train_time / Nepochs,
    }

    return metrics, perf_metrics


# %%
############################
### Examples
############################


def run_example(
    p: list, neuron: AbstractPseudoPhaseOscNeuron, data_loaders, config: dict
) -> dict:
    """
    Simulates the network for a single example input given the parameters `p`.
    """

    ### Unpack arguments
    Nhidden: int = config["Nhidden"]
    Nlayer: int = config["Nlayer"]
    Nout: int = config["Nout"]
    N = Nhidden * (Nlayer - 1) + Nout

    ### Set up the simulation
    @jit
    def jeventffwd(p, times_in, neurons_in):
        return eventffwd(neuron, p, times_in, neurons_in, config)

    @jit
    def joutfn(out, p):
        return outfn(neuron, out, p, config)

    ### Run simulation

    # Data
    _, test_loader = data_loaders

    times_in, neurons_in, label = next(iter(test_loader))
    times_in, neurons_in, label = (
        jnp.array(times_in[2]),
        jnp.array(neurons_in[2]),
        jnp.array(label[2]),
    )
    out = jeventffwd(p, input)
    t_outs = joutfn(out, p)

    ### Prepare results
    times: Array = out[0]
    spike_in: Array = out[1]
    neurons: Array = out[2]

    trace_ts, trace_xs = neuron.traces(p[1][jnp.newaxis], out, config)
    trace_phis = trace_xs[:, 0]
    trace_Vs = neuron.iPhi(trace_phis)

    spiketimes = []
    for i in range(N):
        times_i = times[~spike_in & (neurons == i)]
        spiketimes.append(times_i)
    predicted = jnp.argmin(t_outs)

    ### Pack results in dictionary
    results = {
        "input": input,
        "label": label,
        "predicted": predicted,
        "trace_ts": trace_ts,
        "trace_phis": trace_phis,
        "trace_Vs": trace_Vs,
        "spiketimes": spiketimes,
    }

    return results


# %%
############################
### Plotting
############################


def plot_spikes(ax: Axes, example: dict, config: dict) -> None:
    ### Unpack arguments
    T: float = config["T"]
    Nhidden: int = config["Nhidden"]
    Nlayer: int = config["Nlayer"]
    Nout: int = config["Nout"]
    N = (Nlayer - 1) * Nhidden + Nout
    spiketimes: Array = example["spiketimes"]

    ### Plot spikes
    tick_len = 2
    ax.eventplot(spiketimes, colors="k", linewidths=0.5, linelengths=tick_len)
    patch = Rectangle((0, Nhidden - 1 / 2), T, Nhidden, color="k", alpha=0.2, zorder=0)
    ax.add_patch(patch)
    ax.text(
        T,
        0,
        r"$1^\mathrm{st}$ hidden",
        ha="right",
        va="bottom",
        color="k",
        alpha=0.2,
        zorder=1,
    )
    ax.text(
        T,
        Nhidden - 1 / 2,
        r"$2^\mathrm{nd}$ hidden",
        ha="right",
        va="bottom",
        color="white",
        zorder=1,
    )
    ax.text(
        T,
        2 * Nhidden - 1 / 2,
        "Output",
        ha="right",
        va="bottom",
        color="k",
        alpha=0.2,
        zorder=1,
    )

    ### Formatting
    ax.set_xticks([0, T])
    ax.set_xlim(0, T)
    ax.set_xlabel("Time $t$", labelpad=-3)
    ax.set_yticks(
        [0, Nhidden - 1, 2 * Nhidden - 1, N - 1],
        [str(1), str(Nhidden), str(2 * Nhidden), str(N)],
    )
    ax.set_ylim(-tick_len / 2, N - 1 + tick_len / 2)
    ax.set_ylabel("Neuron", labelpad=-0.1)


def plot_error(ax: Axes, metrics: dict, config: dict, ylog=True) -> None:
    ### Unpack arguments
    Nepochs: int = config["Nepochs"]
    acc: Array = metrics["acc"]
    mean_acc = jnp.mean(acc, 0)
    std_acc = jnp.std(acc, 0)
    acc_ord: Array = metrics["acc_ord"]
    mean_acc_ord = jnp.mean(acc_ord, 0)
    std_acc_ord = jnp.std(acc_ord, 0)
    epochs = jnp.arange(1, Nepochs + 2)

    ### Plot classification error
    ax.plot(epochs, 1 - mean_acc_ord, label="Excl. pseudo", c="C0", zorder=1)
    ax.fill_between(
        epochs,
        1 - mean_acc_ord - std_acc_ord,
        1 - mean_acc_ord + std_acc_ord,
        alpha=0.3,
        color="C0",
    )
    ax.plot(epochs, 1 - mean_acc, label="Incl. pseudo", c="C1", zorder=0)
    ax.fill_between(
        epochs,
        1 - mean_acc - std_acc,
        1 - mean_acc + std_acc,
        alpha=0.3,
        color="C1",
    )
    ax.legend()

    ### Formatting
    ax.set_xlabel("Epochs + 1", labelpad=-1)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(formatter)
    ax.set_ylim(0.01, 1)
    ax.set_ylabel("Test error", labelpad=-3)
    if ylog:
        ax.set_yscale("log")
    ax.yaxis.set_major_formatter(formatter)


def plot_traces(ax: Axes, example: dict, config: dict) -> None:
    ### Unpack arguments
    T: float = config["T"]
    Nhidden: int = config["Nhidden"]
    Nlayer: int = config["Nlayer"]
    Nout: int = config["Nout"]
    N = (Nlayer - 1) * Nhidden + Nout

    ### Unpack example
    trace_ts: Array = example["trace_ts"]
    trace_Vs: Array = example["trace_Vs"]

    ### Plot
    ax.axhline(0, c="gray", alpha=0.3, zorder=-1)
    ax.plot([-0.1, -0.1], [0, 1], c="k", clip_on=False)
    for i in range(10):
        ax.plot(trace_ts, trace_Vs[:, N - Nout + i], color=petroff10[i])
        ax.text((i % 5) * 0.15, -4 - (i // 5) * 3, str(i), color=petroff10[i])

    ### Formatting
    ax.set_xticks([0, T])
    ax.set_xlim(0, T)
    ax.set_xlabel("Time $t$", labelpad=-3)
    ax.set_yticks([])
    ax.set_ylim(-8, 8)
    ax.set_ylabel("Potential $V$")
    ax.spines["left"].set_visible(False)


# %%
############################
### Run functions
############################


def run_theta(
    datasets,
    config: dict,
    data_loaders: tuple[DataLoader, DataLoader] | None = None,
    **kwargs,
):
    """
    Wrapper to train a network of Theta neurons with the given configuration.

    See docstring of `run` and article for more information.
    """
    from spikegd.theta import ThetaNeuron

    if data_loaders is None:
        data_loaders = load_data(datasets, config)

    tau, I0, eps = config["tau"], config["I0"], config["eps"]
    neuron = ThetaNeuron(tau, I0, eps)
    metrics, perf_metrics = run(neuron, data_loaders, config, **kwargs)
    return metrics, perf_metrics


def summarize_ensemble_metrics(ensemble_metrics: dict, Nepochs: int) -> dict:
    metrics: dict = {}
    # epoch 0 is the initial state, other epochs are counted from 1
    epoch_metrics = [{} for _ in range(Nepochs + 1)]

    # print(
    #     {
    #         k: (type(v), f"array{v.shape}" if hasattr(v, "shape") else "no-shape")
    #         for k, v in ensemble_metrics.items()
    #     }
    # )

    for key, value in ensemble_metrics.items():
        if key in ["p_init", "p_end"]:
            continue

        is_global = value.ndim == 1

        if is_global:
            metrics[f"{key}_mean"] = float(jnp.mean(value))
            metrics[f"{key}_std"] = float(jnp.std(value))
        else:
            min_mean = None
            min_mean_epoch = None
            max_mean = None
            max_mean_epoch = None

            # print("local:", key, value.shape, value)

            if value.ndim != 2:
                raise ValueError(
                    f"Expected 2 dimensional value array, got {value.ndim} dimensions."
                )

            if value.shape[1] != Nepochs + 1:
                raise ValueError(
                    f"Expected {Nepochs + 1} (Nepochs + 1) values, "
                    f"got {value.shape[1]} in {key}"
                )

            for epoch in range(Nepochs + 1):
                mean = float(jnp.mean(value[:, epoch]))
                std = float(jnp.std(value[:, epoch]))

                epoch_dict = epoch_metrics[epoch]
                epoch_dict[f"{key}_mean"] = mean
                epoch_dict[f"{key}_std"] = std

                if min_mean is None or mean < min_mean:
                    min_mean = mean
                    min_mean_epoch = epoch
                if max_mean is None or mean > max_mean:
                    max_mean = mean
                    max_mean_epoch = epoch

            # Also store init, final, min and max values for convenience
            metrics[f"{key}_init_mean"] = epoch_metrics[0][f"{key}_mean"]
            metrics[f"{key}_init_std"] = epoch_metrics[0][f"{key}_std"]

            metrics[f"{key}_final_mean"] = epoch_metrics[Nepochs][f"{key}_mean"]
            metrics[f"{key}_final_std"] = epoch_metrics[Nepochs][f"{key}_std"]

            if min_mean_epoch is not None:
                metrics[f"{key}_min_epoch"] = min_mean_epoch
                metrics[f"{key}_min_mean"] = min_mean
                metrics[f"{key}_min_std"] = epoch_metrics[min_mean_epoch][f"{key}_std"]

            if max_mean_epoch is not None:
                metrics[f"{key}_max_epoch"] = max_mean_epoch
                metrics[f"{key}_max_mean"] = max_mean
                metrics[f"{key}_max_std"] = epoch_metrics[max_mean_epoch][f"{key}_std"]

    metrics["epochs"] = epoch_metrics

    return metrics


def run_theta_ensemble(
    datasets,
    config: dict,
    data_loaders: tuple[DataLoader, DataLoader] | None = None,
    **kwargs,
) -> dict:
    seed = config.get("seed", 0)
    Nsamples = config.get("Nsamples", 1)
    Nepochs = config["Nepochs"]

    key = random.PRNGKey(seed)
    seeds = random.randint(key, (Nsamples,), 0, jnp.uint32(2**32 - 1), dtype=jnp.uint32)
    metrics_list = []

    # load data once if not provided
    if data_loaders is None:
        data_loaders = load_data(datasets, config)

    for seed in seeds:
        config_theta = {**config, "seed": seed}
        metrics, perf_metrics = run_theta(
            datasets, config_theta, data_loaders, **kwargs
        )
        metrics_list.append(metrics | perf_metrics)
    metrics = jax.tree.map(lambda *args: jnp.stack(args), *metrics_list)

    return summarize_ensemble_metrics(metrics, Nepochs)
