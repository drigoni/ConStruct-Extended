#!/bin/bash
#SBATCH --job-name=test_no_constraint
#SBATCH --output=/home/drigoni/repository/ConStruct-Thesis/ConStruct/logs/checkpoint_testing/test_no_constraint_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G  
#SBATCH --gres=gpu:1
#SBATCH --exclude=vgpu5-0
#SBATCH --time=40:00:00
#SBATCH --constraint=A6000|A5000
#SBATCH --partition=allgroups,testing

# Load environment
cd /home/drigoni/repository/ConStruct-Thesis
source /conf/shared-software/anaconda/etc/profile.d/conda.sh
conda activate construct-env

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

# CUDA verification
python -c 'import torch; print("torch version:", torch.__version__); print("CUDA available:", torch.cuda.is_available()); print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")'

echo "=== Testing No Constraint (Baseline) ==="
echo "Using No-Constraint Checkpoint: checkpoints/qm9_thesis_no_constraint/epoch-94.ckpt"
echo "Constraint: No structural constraints (natural QM9 distribution)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"

# Run the experiment with no constraint (baseline)
cd ConStruct
python main.py \
    +experiment=thesis/no_constraint/qm9_thesis_no_constraint \
    general.test_only=checkpoints/qm9_thesis_no_constraint/epoch-94.ckpt \
    general.name=test_no_constraint \
    general.final_model_samples_to_generate=10000 \
    general.wandb=disabled \
    general.final_model_chains_to_save=0

echo "No constraint test completed at $(date)"
echo "Results saved to: outputs/checkpoint_testing/no_constraint/" 