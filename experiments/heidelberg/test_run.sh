#!/bin/bash

cd "$(dirname "$0")"

sbatch test_run.slurm

bash watch_output.sh