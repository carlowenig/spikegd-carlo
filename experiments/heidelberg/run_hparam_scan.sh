#!/bin/bash
#SBATCH --partition=mlgpu_devel
#SBATCH --time=0:00:30
#SBATCH --ntasks=1
#SBATCH --gpus=1

module load Python/3.12
source ../../.venv/bin/activate
python hparam_scan.py