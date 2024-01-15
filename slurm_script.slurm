#!/bin/bash
#SBATCH --job-name=PS
#SBATCH --partition=gpu               # Take a node from the 'gpu' partition
#SBATCH --time=01:00:00              # temps maximum d'execution demande (HH:MM:SS)
#SBATCH --export=ALL                  # Export your environment to the compute node
#SBATCH --gres=gpu:1                 
#SBATCH --ntasks=1
#SBATCH --mem=10G                    
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

echo "Running job on $(hostname)"
echo "CUDA_DEVICE: $CUDA_VISIBLE_DEVICES"

echo "computation start $(date)"
cd /scratch2/jsalle/Patchscope/
/scratch2/jsalle/.conda/envs/patchscope_venv/bin/python3 main.py
                                                            