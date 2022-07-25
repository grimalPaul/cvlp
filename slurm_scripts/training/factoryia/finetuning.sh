#!/usr/bin/env sh
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH -J finetuning
#SBATCH --gres=gpu:2
#SBATCH --partition=gpuv100
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=7
#SBATCH -w node27
#SBATCH --mem=45G

source /home/users/pgrimal/.bashrc
source activate cvlp
