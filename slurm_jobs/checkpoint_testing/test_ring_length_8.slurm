#!/bin/bash
#SBATCH --job-name=test_ring_length_8
#SBATCH --output=/home/drigoni/repository/ConStruct-Thesis/slurm_jobs/logs/checkpoint_testing/test_ring_length_8_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G  
#SBATCH --gres=gpu:1
#SBATCH --exclude=vgpu5-0
#SBATCH --time=0-23:00:00
#SBATCH --constraint=A6000|A5000
#SBATCH --partition=allgroups,testing,debug

### ============ VARIABLES SETTING
export SLURM_MASTER_PORT=$(expr 10000 + $(echo -n ${SLURM_JOBID} | tail -c 4))
export SLURM_MASTER_NODE="$(scontrol show hostnames ${SLURM_JOB_NODELIST} | head -n 1)" # such as gn11
# multiprocessing variables
export MASTER_ADDR=${SLURM_MASTER_NODE}
export MASTER_PORT=${SLURM_MASTER_PORT}
export NODE_RANK=${SLURM_PROCID}
export WORLD_SIZE=$(($SLURM_NTASKS_PER_NODE * $SLURM_JOB_NUM_NODES))


### ============ SOME PRINT COMMANDS
echo ''
echo ''
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo 'Date: ' $(date)
echo 'Directory: ' $(pwd)
echo "--------------------------------------------- "
echo "Number of nodes: " ${SLURM_JOB_NUM_NODES}
echo "Ntasks per node: "  ${SLURM_NTASKS_PER_NODE}
echo "NGPUs per node: "  ${SLURM_GPUS_PER_NODE}
echo "Nodelist: " ${SLURM_JOB_NODELIST}  
echo "--------------------------------------------- "        
echo "SLURM_MASTER_PORT: " ${SLURM_MASTER_PORT}
echo "SLURM_MASTER_NODE: " ${SLURM_MASTER_NODE}
echo "--------------------------------------------- "
srun bash -c 'echo "SLURM_PROCID: " ${SLURM_PROCID}'    # note the need to use srun and bash to print the right value
srun echo "TORCH_DEVICE_COUNT: " $(python -c 'import torch; print(torch.cuda.device_count())')
srun nvidia-smi
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo ''
echo ''

# Load environment
cd /home/drigoni/repository/ConStruct-Thesis

export LD_PRELOAD="$CONDA_PREFIX/lib/libgomp.so.1"
# WandB fixes to prevent hanging
export WANDB_TIMEOUT=300
export WANDB_START_METHOD=thread
export WANDB_CONSOLE=wrap
export WANDB_SILENT=false
export WANDB_NETWORK_TIMEOUT=300
export WANDB_HTTP_TIMEOUT=300
# CUDA optimizations
export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0


echo "=== Testing Ring Length ≤ 8 Constraint ==="
echo "Using No-Constraint Checkpoint: checkpoints/qm9_thesis_no_constraint/epoch-94.ckpt"
echo "Constraint: Maximum ring size of 8 atoms"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"

# Run the experiment with ring length ≤ 8 constraint
cd ConStruct
python main.py \
    +experiment=thesis/no_constraint/qm9_thesis_no_constraint \
    general.test_only=checkpoints/qm9_thesis_no_constraint/epoch-94.ckpt \
    general.name=test_ring_length_8 \
    general.final_model_samples_to_generate=10000 \
    model.rev_proj=ring_length_at_most \
    model.max_ring_length=8 \
    general.wandb=disabled \
    general.final_model_chains_to_save=0

echo "Ring Length ≤ 8 test completed at $(date)"
