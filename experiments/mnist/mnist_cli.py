from pathlib import Path
from config import MNISTExperimentConfig
from spikegd.utils.experiment import Experiment


def run_single_theta(config: MNISTExperimentConfig) -> dict:
    """
    Wrapper to train a network of Theta neurons with the given configuration.

    See docstring of `run` and article for more information.
    """
    from mnist import run
    from spikegd.theta import ThetaNeuron

    neuron = ThetaNeuron(config.tau, config.I0, config.eps)
    metrics = run(neuron, config.model_dump(), progress_bar="notebook")
    return metrics


def run_multiple_thetas(config: MNISTExperimentConfig):
    import jax
    import jax.numpy as jnp

    key = jax.random.PRNGKey(config.seed)
    seeds = jax.random.randint(
        key, (config.Nsample,), 0, jnp.uint32(2**32 - 1), dtype=jnp.uint32
    )
    metrics_list = []
    for seed in seeds:
        config_theta = config.model_copy(update={"seed": seed})
        metrics = run_single_theta(config_theta)
        metrics_list.append(metrics)
    metrics_example = metrics_list[0]
    metrics = jax.tree.map(lambda *args: jnp.stack(args), *metrics_list)
    return metrics


mnist_experiment = Experiment(
    Path(__file__).parent, "MNIST", MNISTExperimentConfig, run_multiple_thetas
)

if __name__ == "__main__":
    mnist_experiment.init_cli()
