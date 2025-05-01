#!/bin/bash
#SBATCH --job-name=chess_rl_FEN
#SBATCH -A blumelab
#SBATCH --output=/home/zrc3hc/CHESS_RL_RIV/slurm/logs/chess_rl_%j.out
#SBATCH --error=/home/zrc3hc/CHESS_RL_RIV/slurm/logs/chess_rl_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --mem=100G

module load miniforge

# Run your Python script
python /sfs/gpfs/tardis/home/zrc3hc/CHESS_RL_RIV/LearningEngine_FEN.py
