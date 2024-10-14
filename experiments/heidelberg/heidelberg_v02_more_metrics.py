import time
import warnings
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from jax import jit, random, value_and_grad, vmap
from jaxtyping import Array, ArrayLike, Bool, Float, Int
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from tqdm import trange as trange_script
from tqdm.notebook import trange as trange_notebook

from shd import SHD
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


def quantize_dataset(dataset: SHD, config: dict, dtype=torch.int8):
    sample_indices = range(dataset.Nsamples)
    Nin_virtual = config["Nin_virtual"]
    N_bin = Nin_virtual

    if N_bin > torch.iinfo(dtype).max:
        raise ValueError(
            f"Number of bins N_bin = {N_bin} exceeds maximum value for {dtype}: {torch.iinfo(dtype).max}"
        )

    _normalize_times = config["normalize_times"]
    normalize_times_signal_percentage = config.get(
        "normalize_times_signal_percentage", 0.05
    )

    t_max = config.get("tmax", 1 if _normalize_times else dataset.t_max + 1e-6)

    N_in = dataset.N  # Number of input neurons
    X_arr = np.zeros((len(sample_indices), N_in, Nin_virtual), dtype=np.int32)
    dt = t_max / N_bin
    normalize_times_dt = config.get("normalize_times_dt", dt)

    bin_edges = np.arange(0, t_max + dt / 2, dt)
    assert bin_edges.shape == (
        N_bin + 1,
    ), f"Expected bin_edges to have shape ({N_bin + 1}), got {bin_edges.shape}"

    for i, sample_index in enumerate(sample_indices):
        times = dataset.times_arr[sample_index]
        neurons = dataset.units_arr[sample_index]

        if _normalize_times:
            times, neurons = normalize_times(
                times,
                neurons,
                dt=normalize_times_dt,
                t_max=t_max,
                signal_percentage=normalize_times_signal_percentage,
            )

        # Use numpy's digitize to bin the spike times
        binned_spikes = np.digitize(times, bin_edges) - 1

        # Count spikes in each bin for each neuron and exclude last bin
        count_arr = np.bincount(
            neurons * N_bin + binned_spikes, minlength=N_in * N_bin
        ).reshape(N_in, N_bin)

        count_step = config.get("spike_count_step", 1)
        count_offset = config.get("spike_count_offset", 0)
        count_arr = count_arr // count_step + count_offset

        X_arr[i] = -count_arr

    # Create tensor dataset
    data = torch.as_tensor(X_arr, dtype=dtype)
    targets = torch.as_tensor(dataset.label_index_arr, dtype=torch.int8)

    # print("Quantized data memory:", data.element_size() * data.numel() / 1e6, "MB")

    assert data.shape[:1] == targets.shape == (dataset.Nsamples,)

    return TensorDataset(data, targets)


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

    train_set, test_set = datasets

    torch.manual_seed(config["seed"])

    # Training set
    assert (
        Nin == train_set.N
    ), f"config.Nin ({Nin}) does not match number of neurons in train dataset ({train_set.N})."
    assert Nout == train_set.Nlabel, (
        f"config.Nout does not match number of labels in train dataset."
        f"Got {Nout}, expected {train_set.Nlabel}."
    )

    # print("Quantizing training set...")
    train_set = quantize_dataset(train_set, config)
    # print("Train set size:", len(train_set))

    if (Ntrain := config.get("Ntrain")) is not None:
        train_set = Subset(train_set, np.arange(Ntrain))
        # print(f"Using reduced train_set size of {Ntrain}")

    train_loader = DataLoader(train_set, batch_size=Nbatch, shuffle=True)

    # Test set
    assert (
        Nin == test_set.N
    ), f"config.Nin ({Nin}) does not match number of neurons in test dataset ({test_set.N})."
    assert Nout == test_set.Nlabel, (
        f"config.Nout does not match number of labels in test dataset."
        f"Got {Nout}, expected {test_set.Nlabel}."
    )
    # print("Quantizing test set...")
    test_set = quantize_dataset(test_set, config)
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
    input: Int[Array, " Nin Nvirt"],
    config: dict,
) -> tuple:
    """
    Simulates a feedforward network with time-to-first-spike input encoding.
    """
    ### Unpack arguments
    Nin_virtual: int = config["Nin_virtual"]
    Nhidden: int = config["Nhidden"]
    Nlayer: int = config["Nlayer"]  # currently has to be at least 2 (1hidden)
    Nout: int = config["Nout"]
    N = Nhidden * (Nlayer - 1) + Nout
    T: float = config["T"]
    weights: list = p[0]
    phi0: Array = p[1]
    x0 = phi0[jnp.newaxis]

    ### Input
    # Times computed as spike times of a LIF neuron with constant input Iconst
    # I_const = jnp.arange(Nin_virtual, 0, -1)
    # V_th = 0.001 * Nin_virtual
    # times_in = jnp.where(
    #     I_const > V_th, T * jnp.log(I_const / (I_const - V_th)), jnp.inf
    # )
    # TODO: Linear times (bins)
    times_in = jnp.arange(Nin_virtual) * T / Nin_virtual
    neurons_in = jnp.arange(Nin_virtual)
    spikes_in = (times_in, neurons_in)

    # ### Input weights
    weights_in = weights[0]
    weights_in_virtual = (
        jnp.zeros((N, Nin_virtual)).at[:Nhidden].set(weights_in @ input)
    )

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
    out = neuron.event(x0, weights_net, weights_in_virtual, spikes_in, config)

    return out


# def leaky_integrate(
#     times: Float[Array, " Nt"],
#     V_0: Float[Array, ""],
#     w: Float[Array, ""],
#     spike_mask: Bool[Array, " Nt"],
#     config: dict,
# ) -> Float[Array, " Nt"]:
#     Nt = len(times)  # number of time steps
#     assert len(spike_mask) == Nt

#     # weights for actual spikes or 0 otherwise
#     w_sp = jnp.where(spike_mask, w, 0)

#     I0: float = config["I0"]
#     tau: float = config["tau"]

#     def spike_scanner(
#         V: Float[Array, ""], i: Int[Array, ""]
#     ) -> tuple[Float[Array, ""], Float[Array, ""]]:
#         t_prev = jnp.where(i == 0, 0, times[i - 1])
#         t_next = times[i]

#         V_next = (V - I0) * jnp.exp(-(t_next - t_prev) / tau) + I0 + w_sp[i]

#         return V_next, V_next

#     _, V_arr = jax.lax.scan(spike_scanner, V_0, jnp.arange(Nt))

#     return V_arr


def assert_equal(*args, **kwargs):
    arg_list = [(f"arg {i}", arg) for i, arg in enumerate(args)] + list(kwargs.items())
    first_name, first_arg = arg_list.pop(0)

    for name, arg in arg_list:
        assert (
            arg == first_arg
        ), f"{name} does not equal {first_name}: {arg!r} != {first_arg!r}"


def assert_shape(shape: tuple[int, ...], *args, **kwargs):
    arg_list = [(f"array {i}", arg) for i, arg in enumerate(args)] + list(
        kwargs.items()
    )

    for name, arg in arg_list:
        assert (
            arg_shape := jnp.shape(arg)
        ) == shape, f"{name} has invalid shape. Expected {shape}, got {arg_shape}"


def assert_same_shape(*args, **kwargs):
    arg_list = [(f"array {i}", arg) for i, arg in enumerate(args)] + list(
        kwargs.items()
    )

    first_name, first_arg = arg_list.pop(0)
    first_shape = jnp.shape(first_arg)

    for name, arg in arg_list:
        assert (
            (shape := jnp.shape(arg)) == first_shape
        ), f"{name} has a different shape {shape} than {first_name} {first_shape}"


@partial(
    vmap, in_axes=(None, None, 0, 0, None, None)
)  # vectorize along output neurons (weight vector (Nhidden,) -> weight matrix (Nout, Nhidden))
@partial(vmap, in_axes=(0, None, None, None, None, None))  # vectorize along eval_times
def leaky_integrate(
    eval_time: Float[Array, ""],
    spikes: tuple[Float[Array, " K"], Int[Array, " K"]],
    weights: Float[Array, " N"],
    V_0: Float[Array, ""],
    input_spike_mask: Bool[Array, " K"],
    config: dict,
) -> Float[Array, ""]:
    """"""
    I_0 = config["I0"]
    tau = config["tau"]

    spike_times, spike_neurons = spikes
    assert_same_shape(
        spike_times=spike_times,
        spike_neurons=spike_neurons,
        input_spike_mask=input_spike_mask,
    )

    N = config["N"]
    assert_shape((N,), weights=weights)
    # assert jnp.any(spike_neurons >= N), "spike neuron index above N"
    # assert jnp.any(spike_neurons < 0), "negative spike neuron index"

    spike_weights = weights[spike_neurons]

    return (
        (V_0 - I_0) * jnp.exp(-eval_time / tau)
        + I_0
        + jnp.sum(
            spike_weights * jnp.exp(-(eval_time - spike_times) / tau),
            where=(spike_times <= eval_time) & input_spike_mask,
        )
    )


def outfn(
    neuron: AbstractPseudoPhaseOscNeuron, out: tuple, p: list, config: dict
) -> Float[Array, " Nout"]:
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

    out_func = config["out_func"]

    if out_func == "first_spike_time":
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

    elif out_func == "max_over_time_potential":
        T: float = config["T"]
        readout_V0: float = config["readout_V0"]
        readout_w_factor: float = config["readout_w_factor"]
        readout_tau_factor: float = config["readout_tau_factor"]
        readout_I0: float = config["readout_I0"]
        eval_times = jnp.append(times, T)  # evaluate at every spike time
        # TODO: only use spike times in last hidden layer?

        output_weights = weights[Nlayer - 1]
        assert_shape(
            (Nout, Nhidden), output_weights=output_weights
        )  # (see init_weights, output layer)

        # is_spike_in_last_hidden_layer = (
        #     (neurons >= Nhidden * (Nlayer - 2))
        #     & (neurons < Nhidden * (Nlayer - 1))
        #     & (spike_in == False)
        # )

        weight_matrix = jnp.zeros((Nout, N))

        last_hidden_start = N - Nout - Nhidden
        last_hidden_end = N - Nout

        weight_matrix.at[:, last_hidden_start:last_hidden_end].set(output_weights)

        is_spike_in_last_hidden_layer = (
            (neurons >= last_hidden_start)
            & (neurons < last_hidden_end)
            & (spike_in == False)
        )

        V_0 = jnp.full(Nout, readout_V0)

        V_arr = leaky_integrate(
            eval_times,
            (times, neurons),
            weight_matrix * readout_w_factor,
            V_0,
            is_spike_in_last_hidden_layer,
            {"tau": config["tau"] * readout_tau_factor, "I0": readout_I0, "N": N},
        )
        assert_shape((Nout, len(eval_times)), V_arr=V_arr)

        t_outs = -jnp.max(V_arr, axis=1)  # max over time

    else:
        raise ValueError(f"Unknown output function: {out_func}")

    assert_shape((Nout,), t_outs=t_outs)

    return t_outs


def lossfn(
    t_outs: Float[Array, " Nout"], label: Int[Array, ""], config: dict
) -> tuple[Array, Array]:
    """
    Compute cross entropy loss and if the network prediction was correct.
    """
    T: float = config["T"]
    log_softmax = jax.nn.log_softmax(-t_outs)
    loss = -log_softmax[label]

    if config["out_func"] == "first_spike_time":
        regu = jnp.exp(t_outs / T) - 1
        gamma: float = config["gamma"]
        loss += gamma * regu[label]

    correct = jnp.argmin(t_outs) == label
    return loss, correct


def simulatefn(
    neuron: AbstractPseudoPhaseOscNeuron,
    p: list,
    input: Int[Array, "Batch Nin"],
    labels: Int[Array, " Batch"],
    config: dict,
) -> tuple[Array, Array, tuple]:
    """
    Simulates the network and computes the loss and accuracy for batched input.
    """
    outs = vmap(eventffwd, in_axes=(None, None, 0, None))(neuron, p, input, config)
    t_outs = vmap(outfn, in_axes=(None, 0, None, None))(neuron, outs, p, config)
    loss, correct = vmap(lossfn, in_axes=(0, 0, None))(t_outs, labels, config)
    mean_loss = jnp.mean(loss)
    accuracy = jnp.mean(correct)
    return mean_loss, accuracy, outs


def probefn(
    neuron: AbstractPseudoPhaseOscNeuron,
    p: list,
    input: Int[Array, "Batch Nin Nvirt"],
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
    def batch_eventffwd(input):
        return eventffwd(neuron, p, input, config)

    @vmap
    def batch_outfn(outs):
        return outfn(neuron, outs, p, config)

    @vmap
    def batch_lossfn(t_outs, labels):
        return lossfn(t_outs, labels, config)

    ### Run network
    outs = batch_eventffwd(input)
    times: Array = outs[0]  # shape (Nbatch, Nspike)
    spike_in: Array = outs[1]
    neurons: Array = outs[2]
    # each outfn call returns an array of shape (Nout,),
    # so batch_outfn returns an array of shape (Nbatch, Nout).
    t_outs = batch_outfn(outs)  # shape (Nbatch, Nout)

    ### Loss and accuracy with pseudospikes
    loss, correct = batch_lossfn(t_outs, labels)
    mean_loss = jnp.mean(loss)
    loss_nan_ratio = jnp.mean(jnp.isnan(loss))
    acc = jnp.mean(correct)

    ### Loss and accuracy without pseudospikes (if using first_spike_time)
    t_out_ord = (
        jnp.where(t_outs < T, t_outs, T)
        if config["out_func"] == "first_spike_time"
        else t_outs
    )
    loss_ord, correct_ord = batch_lossfn(t_out_ord, labels)
    mean_loss_ord = jnp.mean(loss_ord)
    acc_ord = jnp.mean(correct_ord)

    ### Activity and silent neurons
    Ninternal = N - Nout
    spike_is_internal = (spike_in == False) & (neurons < Ninternal) & (neurons >= 0)  # noqa: E712
    activity = jnp.sum(spike_is_internal) / (Nbatch * Ninternal)
    silent_neurons = jnp.isin(
        jnp.arange(Ninternal), jnp.where(spike_is_internal, neurons, -1), invert=True
    )

    ### Activity and silent neurons until first output spike
    t_out_first = jnp.min(t_out_ord, axis=1)
    spike_is_internal_pre_output = spike_is_internal & (
        times < t_out_first[:, jnp.newaxis]
    )
    activity_first = jnp.sum(spike_is_internal_pre_output) / (Nbatch * Ninternal)
    silent_neurons_first = jnp.isin(
        jnp.arange(Ninternal),
        jnp.where(spike_is_internal_pre_output, neurons, -1),
        invert=True,
    )

    ### Output activity
    spike_is_output = (spike_in == False) & (neurons >= N - Nout) & (neurons < N)  # noqa: E712
    # out_spike_mask.shape = (Nbatch, Nspike)

    # total number of output spikes in all batch samples & all output neurons
    Nspike_out_total = jnp.sum(spike_is_output)
    activity_out = Nspike_out_total / (Nbatch * Nout)

    # assign true output neuron index (from true label) to each spike in each sample
    neurons_true = N - Nout + labels[:, jnp.newaxis].astype(jnp.int16)

    # total number of output spikes in all batch samples
    # for the output neuron with the true label
    spike_is_true_output = spike_is_output & (neurons == neurons_true)
    Nspike_out_true_total = jnp.sum(spike_is_true_output)  # noqa: E712
    activity_out_true = Nspike_out_true_total / Nbatch

    # total number of output spikes in all batch samples & all output neurons
    # *except* the one with the true label
    spike_is_false_output = spike_is_output & (neurons != neurons_true)
    Nspike_out_false_total = jnp.sum(spike_is_false_output)
    activity_out_false = Nspike_out_false_total / (Nbatch * (Nout - 1))

    ### Output value
    # in each batch, get t_out for true neuron
    output_neuron_is_true_label = labels[:, jnp.newaxis] == jnp.arange(Nout)
    # shape (Nbatch, Nout)

    out_val_true = jnp.mean(t_outs, where=output_neuron_is_true_label)
    out_val_false = jnp.mean(t_outs, where=~output_neuron_is_true_label)

    ### Pack results in dictionary
    metrics = {
        "loss": mean_loss,
        "loss_nan_ratio": loss_nan_ratio,
        "acc": acc,
        "loss_ord": mean_loss_ord,
        "acc_ord": acc_ord,
        "activity": activity,
        "activity_first": activity_first,
        "activity_out": activity_out,
        "activity_out_true": activity_out_true,
        "activity_out_false": activity_out_false,
        "out_val_true": out_val_true,
        "out_val_false": out_val_false,
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
) -> tuple[dict, dict, dict, dict, dict]:
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
    Nin_virtual: int = config["Nin_virtual"]
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
    T: float = config["T"]

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
        p: list, input: Int[Array, "Batch Nin"], labels: Int[Array, " Batch"]
    ) -> tuple[Array, tuple]:
        loss, acc, outs = simulatefn(neuron, p, input, labels, config)
        return loss, (acc, outs)

    # Regularization
    @jit
    def flip(key: Array, input: Array) -> tuple[Array, Array]:
        key, subkey = jax.random.split(key)
        mask = jax.random.bernoulli(subkey, p=p_flip, shape=input.shape)
        return key, jnp.where(mask, Nin_virtual - input, input)

    # Optimization step
    @jit
    def trial(
        p: list,
        input: Int[Array, "Batch Nin"],
        labels: Int[Array, " Batch"],
        opt_state: optax.OptState,
    ) -> tuple:
        (loss, (acc, outs)), grad = gradfn(p, input, labels)
        updates, opt_state = optim.update(grad, opt_state)
        p = optax.apply_updates(p, updates)  # type: ignore
        p[1] = jnp.clip(p[1], 0, theta)
        return loss, acc, p, opt_state, outs, grad

    # Probe network
    @jit
    def jprobefn(p, input, labels):
        return probefn(neuron, p, input, labels, config)

    init_compile_time = time.perf_counter() - init_compile_start

    def probe(p: list, input, labels) -> dict:
        # metrics = {
        #     "loss": 0.0,
        #     "loss_nan_ratio": 0.0,
        #     "acc": 0.0,
        #     "loss_ord": 0.0,
        #     "acc_ord": 0.0,
        #     "activity": 0.0,
        #     "activity_first": 0.0,
        #     "activity_out": 0.0,
        #     "activity_out_true": 0.0,
        #     "activity_out_false": 0.0,
        #     "out_val_true": 0.0,
        #     "out_val_false": 0.0,
        # }
        # silents = {
        #     "silent_neurons": jnp.ones(N - Nout, dtype=bool),
        #     "silent_neurons_first": jnp.ones(N - Nout, dtype=bool),
        # }
        # steps = len(loader)
        # for data in loader:
        #     input, labels = jnp.array(data[0]), jnp.array(data[1])
        #     metric, silent = jprobefn(p, input, labels)
        #     metrics = {k: metrics[k] + metric[k] / steps for k in metrics}
        #     silents = {k: silents[k] & silent[k] for k in silents}

        metrics, silents = jprobefn(p, input, labels)

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
    test_input = jnp.array(test_loader.dataset.tensors[0])  # type: ignore
    test_labels = jnp.array(test_loader.dataset.tensors[1])  # type: ignore

    test_metrics: dict[str, Array | list] = {
        k: [v] for k, v in probe(p, test_input, test_labels).items()
    }

    train_input = jnp.array(train_loader.dataset.tensors[0])  # type: ignore
    train_labels = jnp.array(train_loader.dataset.tensors[1])  # type: ignore

    train_metrics: dict = {
        k: [v] for k, v in probe(p, train_input, train_labels).items()
    }

    opt_metrics: dict = {
        "weight_grad": [0.0],
        "weight_in_grad": [0.0],
        "weight_out_grad": [0.0],
    }

    init_time = time.perf_counter() - init_start

    train_start = time.perf_counter()

    opt_time_total = 0.0
    probe_time_total = 0.0
    N_probe_samples = len(train_loader.dataset) + len(test_loader.dataset)  # type: ignore

    p_best_acc = p_init
    epoch_best_acc = 0
    best_acc = test_metrics["acc"][-1]

    # Training
    # (start from 1 since 0th epoch is used for initial state)
    for epoch in trange(1, Nepochs + 1):
        opt_start = time.perf_counter()

        for data in train_loader:
            input, labels = jnp.array(data[0]), jnp.array(data[1])
            key, input = flip(key, input)

            loss, acc, p, opt_state, outs, grad = trial(p, input, labels, opt_state)

            out_ts = outs[0]
            assert_equal(out_ts_last_dim_size=out_ts.shape[-1], K=config["K"])

            out_final_ts = out_ts[:, -1]

            if jnp.any(out_final_ts < T):
                warnings.warn(
                    f"{jnp.sum(out_final_ts < T)} of {len(out_final_ts)} simulated "
                    f"spikes appeared before trial end, starting at "
                    f"{out_final_ts[out_final_ts < T].min()}."
                    f"Try increasing K to simulate enough spikes."
                )

            # for layer in range(Nlayer):
            #     print(f"LAYER {layer}")

            #     weights_grad = grad[0][layer]
            #     if jnp.any(jnp.isnan(weights_grad)):
            #         warnings.warn(
            #             f"Found {jnp.sum(jnp.isnan(weights_grad))} nan weight gradients (of {jnp.size(weights_grad)} weights)."
            #         )

            #     print(f"Mean weight grad: {jnp.mean(weights_grad)}")
            #     print(f"Min weight grad: {jnp.min(weights_grad)}")
            #     print(f"Max weight grad: {jnp.max(weights_grad)}")

        opt_time = time.perf_counter() - opt_start
        opt_time_total += opt_time

        # Probe network
        probe_start = time.perf_counter()

        test_metric = probe(p, test_input, test_labels)
        test_metrics = {k: v + [test_metric[k]] for k, v in test_metrics.items()}

        train_metric = probe(p, train_input, train_labels)
        train_metrics = {k: v + [train_metric[k]] for k, v in train_metrics.items()}

        weight_grad = grad[0]
        weight_mean_grads = jnp.array(
            [jnp.mean(layer_grad) for layer_grad in weight_grad]
        )
        opt_metrics["weight_grad"].append(jnp.mean(weight_mean_grads))
        opt_metrics["weight_in_grad"].append(weight_mean_grads[0])
        opt_metrics["weight_out_grad"].append(weight_mean_grads[-1])

        probe_time = time.perf_counter() - probe_start
        probe_time_total += probe_time

        if test_metric["acc"] > best_acc:
            best_acc = test_metric["acc"]
            p_best_acc = p
            epoch_best_acc = epoch

    train_time = time.perf_counter() - train_start

    if jnp.any(jnp.isnan(jnp.array(test_metrics["loss"]))):
        print(
            "Warning: A NaN appeared. "
            "Likely not enough spikes have been simulated. "
            "Try increasing `K`."
        )

    test_metrics = {k: jnp.array(v) for k, v in test_metrics.items()}
    train_metrics = {k: jnp.array(v) for k, v in train_metrics.items()}
    opt_metrics = {k: jnp.array(v) for k, v in opt_metrics.items()}

    params = {
        "init": p_init,
        "final": p,
        "best_acc": p_best_acc,
        "_meta": {
            "epochs": {
                "init": 0,
                "final": epoch,
                "best_acc": epoch_best_acc,
            }
        },
    }

    perf_metrics = {
        "init_time": init_time,
        "init_compile_time": init_compile_time,
        "init_weights_time": init_weights_time,
        "init_phi0_time": init_phi0_time,
        "init_optim_time": init_optim_time,
        "train_time": train_time,
        "epoch_time": train_time / Nepochs,
        "opt_time_total": opt_time_total,
        "opt_time_per_epoch": opt_time_total / Nepochs,
        "opt_time_per_sample": opt_time_total / (Nepochs * len(train_loader)),
        "probe_time_total": probe_time_total,
        "probe_time_per_epoch": probe_time_total / Nepochs,
        "probe_time_per_sample": probe_time_total / (Nepochs * N_probe_samples),
    }

    return params, test_metrics, train_metrics, opt_metrics, perf_metrics


# %%
############################
### Examples
############################


def run_example(
    p: list,
    neuron: AbstractPseudoPhaseOscNeuron,
    dataset: Dataset,
    config: dict,
    sample_index=0,
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
    def jeventffwd(p, input):
        return eventffwd(neuron, p, input, config)

    @jit
    def joutfn(out, p):
        return outfn(neuron, out, p, config)

    ### Run simulation

    # Data
    input, label = map(jnp.array, dataset[sample_index])
    # input, label = jnp.array(input[sample_index]), jnp.array(label[sample_index])
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


def plot_output_spikes(ax: Axes, example: dict, config: dict) -> None:
    ### Unpack arguments
    T: float = config["T"]
    Nhidden: int = config["Nhidden"]
    Nlayer: int = config["Nlayer"]
    Nout: int = config["Nout"]
    N = (Nlayer - 1) * Nhidden + Nout
    spiketimes: Array = example["spiketimes"]

    spiketimes = spiketimes[N - Nout :]

    ### Plot spikes
    tick_len = 2
    ax.eventplot(spiketimes, colors="k", linewidths=0.5, linelengths=tick_len)

    ### Formatting
    ax.set_xlim(0, T)
    ax.set_xlabel("Time $t$", labelpad=-3)
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
    data_loaders: tuple[DataLoader, DataLoader],
    config: dict,
    **kwargs,
):
    """
    Wrapper to train a network of Theta neurons with the given configuration.

    See docstring of `run` and article for more information.
    """
    from spikegd.theta import ThetaNeuron

    tau, I0, eps = config["tau"], config["I0"], config["eps"]
    neuron = ThetaNeuron(tau, I0, eps)
    return run(neuron, data_loaders, config, **kwargs)


def run_theta_example(dataset: Dataset, p: list, config: dict, **kwargs) -> dict:
    from spikegd.theta import ThetaNeuron

    tau, I0, eps = config["tau"], config["I0"], config["eps"]
    neuron = ThetaNeuron(tau, I0, eps)
    return run_example(p, neuron, dataset, config, **kwargs)


def summarize_ensemble_metrics(ensemble_metrics: dict, Nepochs: int) -> dict:
    metrics: dict = {}
    # epoch 0 is the initial state, other epochs are counted from 1
    epoch_metrics = [{} for _ in range(Nepochs + 1)]

    for key, value in ensemble_metrics.items():
        # scalars have shape (Nsample,); epoch arrays have shape (Nsample, Nepochs+1)
        is_scalar = value.ndim == 1

        if is_scalar:
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

    if any(len(m) > 0 for m in epoch_metrics):
        metrics["epochs"] = epoch_metrics

    return metrics


def run_theta_ensemble(
    datasets,
    config: dict,
    data_loaders: tuple[DataLoader, DataLoader] | None = None,
    return_params=False,
    **kwargs,
) -> dict:
    seed = config.get("seed", 0)
    Nsamples = config.get("Nsamples", 1)
    Nepochs = config["Nepochs"]

    key = random.PRNGKey(seed)
    seeds = random.randint(key, (Nsamples,), 0, jnp.uint32(2**32 - 1), dtype=jnp.uint32)
    params_list = []
    test_metrics_list = []
    train_metrics_list = []
    opt_metrics_list = []
    perf_metrics_list = []

    # load data once if not provided
    if data_loaders is None:
        data_loaders = load_data(datasets, config)

    for seed in seeds:
        config_theta = {**config, "seed": seed}
        params, test_metrics, train_metrics, opt_metrics, perf_metrics = run_theta(
            data_loaders, config_theta, **kwargs
        )
        params_list.append(params)
        test_metrics_list.append(test_metrics)
        train_metrics_list.append(train_metrics)
        opt_metrics_list.append(opt_metrics)
        perf_metrics_list.append(perf_metrics)

    test_metrics = jax.tree.map(lambda *args: jnp.stack(args), *test_metrics_list)
    train_metrics = jax.tree.map(lambda *args: jnp.stack(args), *train_metrics_list)
    opt_metrics = jax.tree.map(lambda *args: jnp.stack(args), *opt_metrics_list)
    perf_metrics = jax.tree.map(lambda *args: jnp.stack(args), *perf_metrics_list)

    # summarized_test_metrics = summarize_ensemble_metrics(test_metrics, Nepochs)
    # summarized_train_metrics = summarize_ensemble_metrics(train_metrics, Nepochs)
    # summarized_opt_metrics = summarize_ensemble_metrics(opt_metrics, Nepochs)
    # summarized_perf_metrics = summarize_ensemble_metrics(perf_metrics, Nepochs)

    # metrics: dict = {
    #     **summarized_test_metrics,
    #     **{f"train_{key}": value for key, value in summarized_train_metrics.items()},
    #     **summarized_opt_metrics,
    #     **summarized_perf_metrics,
    # }

    metrics = summarize_ensemble_metrics(
        {
            **test_metrics,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **opt_metrics,
            **perf_metrics,
        },
        Nepochs,
    )

    if return_params:
        metrics["params"] = params_list

    return metrics
