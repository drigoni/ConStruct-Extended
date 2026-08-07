#!/bin/bash
#SBATCH --job-name=moses_ring_length_at_least_4_debug
#SBATCH --output=/home/drigoni/repository/ConStruct-Thesis/slurm_jobs/out/moses_ring_length_at_least_4_debug_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint=A6000|A5000|A100|V100|RTX2000
#SBATCH --time=16:00:00
#SBATCH --partition=debug,testing

export SLURM_MASTER_PORT=$(expr 10000 + $(echo -n ${SLURM_JOBID} | tail -c 4))
export SLURM_MASTER_NODE="$(scontrol show hostnames ${SLURM_JOB_NODELIST} | head -n 1)"
export MASTER_ADDR=${SLURM_MASTER_NODE}
export MASTER_PORT=${SLURM_MASTER_PORT}
export NODE_RANK=${SLURM_PROCID}
export WORLD_SIZE=$(($SLURM_NTASKS_PER_NODE * $SLURM_JOB_NUM_NODES))

cd /home/drigoni/repository/ConStruct-Thesis
export LD_PRELOAD="$CONDA_PREFIX/lib/libgomp.so.1"
export WANDB_TIMEOUT=300
export WANDB_START_METHOD=thread
export WANDB_NETWORK_TIMEOUT=300
export WANDB_HTTP_TIMEOUT=300
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1

echo "Running debug/edge_insertion/ring_length_at_least/moses_debug_ring_length_at_least_4"
python main.py +experiment=debug/edge_insertion/ring_length_at_least/moses_debug_ring_length_at_least_4
