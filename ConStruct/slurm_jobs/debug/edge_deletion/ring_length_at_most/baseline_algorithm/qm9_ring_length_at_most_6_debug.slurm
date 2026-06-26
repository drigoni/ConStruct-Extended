#!/bin/bash
#SBATCH --job-name=qm9_ring_length_at_most_6_debug
#SBATCH --output=/home/drigoni/repository/ConStruct-Thesis/ConStruct/logs/qm9_ring_length_at_most_6_debug_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G  
#SBATCH --gres=gpu:1
#SBATCH --exclude=vgpu5-0
#SBATCH --time=16:00:00
#SBATCH --partition=debug

# Load environment
cd /home/drigoni/repository/ConStruct-Thesis
source /conf/shared-software/anaconda/etc/profile.d/conda.sh
conda activate construct-env

export LD_PRELOAD="$CONDA_PREFIX/lib/libgomp.so.1"

# WandB fixes to prevent hanging
export WANDB_TIMEOUT=300  # 5-minute timeout for WandB operations
export WANDB_START_METHOD=thread  # Use thread-based WandB
export WANDB_CONSOLE=wrap  # Better console output
export WANDB_SILENT=false  # Show WandB progress
export WANDB_NETWORK_TIMEOUT=300
export WANDB_HTTP_TIMEOUT=300

# Force unbuffered output for better logging
export PYTHONUNBUFFERED=1

# CUDA verification
python -c 'import torch; print("torch version:", torch.__version__); print("CUDA available:", torch.cuda.is_available()); print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")'

echo "=== Running Experiment: ring_length_at_most_6_debug ==="
echo "Config: debug/edge_deletion/ring_length_at_most/qm9_debug_ring_length_at_most_6"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"

# Run the experiment
cd ConStruct
python main.py +experiment=debug/edge_deletion/ring_length_at_most/qm9_debug_ring_length_at_most_6 

echo "Experiment completed at $(date)"
