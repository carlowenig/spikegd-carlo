#!/bin/bash
#SBATCH --partition=mlgpu_devel
#SBATCH --time=0:05:00
#SBATCH --gpus=1
#SBATCH --account=ag_ifg_memmesheimer

source ~/.bashrc

source .venv/bin/activate
pip install --upgrade pip
pip install --upgrade -e .