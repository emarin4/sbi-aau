#!/bin/bash
#SBATCH --job-name=stream_sims_disbatch
#SBATCH --account=upa160
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=emarin4@uw.edu
#SBATCH --output=logs/disbatch_%j.out
#SBATCH --error=logs/disbatch_%j.err

source ~/.bashrc
conda activate sbi-stream

cd /expanse/lustre/projects/upa160/lmarin/aau_sbi_project/run_sims/simulations

disBatch taskfile_929_copy.txt
#python -u simulate_worker_combined.py perturbers_batch_0000.pkl 0 10000