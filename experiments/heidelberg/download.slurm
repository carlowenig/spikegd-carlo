#!/bin/bash
#SBATCH --partition=mlgpu_devel
#SBATCH --time=0:00:30
#SBATCH --gpus=1
#SBATCH --account=ag_ifg_memmesheimer
#SBATCH --output=slurm_output/%x_%j.out

source ~/.bashrc

echo "Current working directory: $PWD"
if [[ "$PWD" != */experiments/heidelberg ]]; then
    echo "Error: This script must be run from the experiments/heidelberg directory."
    exit 1
fi 

source ../../.venv/bin/activate
echo "Using Python venv at: $(which python)"
echo ""

python -u download.py