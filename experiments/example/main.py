from spikegd.utils.experiment_new import ExperimentDefinition

experiment = ExperimentDefinition(__file__)

if __name__ == "__main__":
    experiment.init_cli()