#!/bin/bash
#SBATCH --job-name=qm9_no_constraint_thesis
#SBATCH --output=/home/drigoni/repository/ConStruct-Thesis/ConStruct/logs/thesis/qm9_no_constraint_thesis_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=31G
#SBATCH --gres=gpu:1
#SBATCH --exclude=vgpu5-0
#SBATCH --constraint=A6000|A5000|A100|V100|RTX2000
#SBATCH --time=0-23:00:00
#SBATCH --partition=testing,allgroups,debug



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
#source /conf/shared-software/anaconda/etc/profile.d/conda.sh
#conda activate construct-env

export LD_PRELOAD="$CONDA_PREFIX/lib/libgomp.so.1"

# WandB fixes to prevent hanging
export WANDB_TIMEOUT=300  # 5-minute timeout for WandB operations
export WANDB_START_METHOD=thread  # Use thread-based WandB
export WANDB_CONSOLE=wrap  # Better console output
export WANDB_SILENT=false  # Show WandB progress
export WANDB_NETWORK_TIMEOUT=300
export WANDB_HTTP_TIMEOUT=300

# CUDA optimizations
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_CUDNN_BENCHMARK=1

# Force unbuffered output for better logging
export PYTHONUNBUFFERED=1


echo "=== Running Experiment: qm9_no_constraint_thesis ==="
echo "Config: thesis/no_constraint/qm9_thesis_no_constraint"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"

# Run the experiment
cd ConStruct
python main.py +experiment=thesis/no_constraint/qm9_thesis_no_constraint

echo "Experiment completed at $(date)"
