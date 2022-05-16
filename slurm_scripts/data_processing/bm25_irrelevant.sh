#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J irrelevant_BM25
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --mem=4G
#SBATCH --time=0-06:00:00