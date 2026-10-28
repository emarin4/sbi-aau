#!/bin/bash
#SBATCH --job-name=perturber_dicts
#SBATCH --account=upa160
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emarin4@uw.edu
#SBATCH --output=logs/perturber_dicts_%j.out
#SBATCH --error=logs/perturber_dicts_%j.err

source ~/.bashrc
conda activate sbi-stream

cd /expanse/lustre/projects/upa160/lmarin/aau_sbi_project/run_sims/perturbers

# make sure the logs folder exists before SLURM tries to write into it
mkdir -p logs

python perturber_dicts_inputs_sequential.py
