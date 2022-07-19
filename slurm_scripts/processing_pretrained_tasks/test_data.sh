#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J test
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=0-06:00:00

source /home/pgrimal/.bashrc
source activate cvlp

python -m cvlep.pretrain_data