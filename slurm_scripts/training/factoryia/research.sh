#!/usr/bin/env sh
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH -J research
#SBATCH --gres=gpu:1
#SBATCH --partition=classicgpu,gpup5000short,gpup6000
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --dependency=58212
source /home/users/pgrimal/.bashrc
source activate cvlp

echo "clip multitask"

python -m search \
    --dataset_path=/home/users/pgrimal/data/datasets/cvlp/miniviquae/test \
    --config=experiments/ir/VL/experiments/clip_multitask/multitask.json \
    --metrics_path=experiments/ir/VL/clip_multitask/ \
    --k=100 \
    --batch_size=128

echo "DONE"