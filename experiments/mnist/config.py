import jax.numpy as jnp
from pydantic import BaseModel, PositiveInt


class MNISTExperimentConfig(BaseModel):
    seed: int = 0
    Nsample: PositiveInt = 1
    # Neuron
    tau: float = 6 / jnp.pi
    I0: float = 5 / 4
    eps: float = 1e-6
    # Network
    Nin: PositiveInt = 784
    Nin_virtual: PositiveInt = 1  # #Virtual input neurons = #Pixel value bins - 1
    Nhidden: PositiveInt = 100
    Nlayer: PositiveInt = 3  # Number of layers
    Nout: PositiveInt = 10
    w_scale: float = 0.5  # Scaling factor of initial weights
    # Trial
    T: float = 2.0
    K: PositiveInt = 200  # Maximal number of simulated ordinary spikes
    dt: float = 0.001  # Step size used to compute state traces
    # Training
    gamma: float = 1e-2
    Nbatch: PositiveInt = 1000
    lr: float = 4e-3
    tau_lr: float = 1e2
    beta1: float = 0.9
    beta2: float = 0.999
    p_flip: float = 0.02
    Nepochs: PositiveInt = 100
    Ntrain: PositiveInt | None = None  # Maximum number of training samples
