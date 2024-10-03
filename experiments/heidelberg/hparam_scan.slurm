#!/bin/bash
#SBATCH --array=1-13
#SBATCH --partition=mlgpu_short
#SBATCH --time=8:00:00
#SBATCH --gpus=1
#SBATCH --account=ag_ifg_memmesheimer
#SBATCH --output=slurm_output/%x_%A_%a.out

source ~/.bashrc

echo "Current working directory: $PWD"
if [[ "$PWD" != */experiments/heidelberg ]]; then
    echo "Error: This script must be run from the experiments/heidelberg directory."
    exit 1
fi 

if [[ "$SLURM_ARRAY_TASK_MIN" != 1 ]]; then
    echo "Error: The minimal array task id should be 1"
    exit 1
fi

source ../../.venv/bin/activate
echo "Using Python venv at: $(which python)"
echo ""

echo "--- SCRIPT START ---"
python -u hparam_scan.py --id $SLURM_ARRAY_JOB_ID --n-jobs $SLURM_ARRAY_TASK_COUNT --job-index $SLURM_ARRAY_TASK_ID
echo "--- SCRIPT END ---"