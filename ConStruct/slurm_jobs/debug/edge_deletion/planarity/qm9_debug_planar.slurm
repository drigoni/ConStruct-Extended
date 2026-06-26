#!/bin/bash
#SBATCH --job-name=qm9_debug_planar
#SBATCH --output=/home/drigoni/repository//repository//repository//repository/ConStruct-Thesis/ConStruct/logs/debug/qm9_debug_planar_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G  
#SBATCH --gres=gpu:1
#SBATCH --exclude=vgpu5-0
#SBATCH --time=16:00:00
#SBATCH --partition=allgroups,testing

# Load environment
cd /home/drigoni/repository//repository//repository//repository/ConStruct-Thesis
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

echo "=== Running QM9 Debug Planarity Experiment ==="
echo "Config: debug/edge_deletion/planarity/qm9_debug_planar"
echo "Dataset: /home/drigoni/repository//repository//repository//repository/ConStruct-Thesis/data/qm9"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"

# Verify QM9 dataset exists
echo "=== Checking QM9 dataset ==="
if [ -d "/home/drigoni/repository//repository//repository//repository/ConStruct-Thesis/data/qm9" ]; then
    echo "✅ QM9 dataset found at /home/drigoni/repository//repository//repository//repository/ConStruct-Thesis/data/qm9"
else
    echo "❌ QM9 dataset not found at /home/drigoni/repository//repository//repository//repository/ConStruct-Thesis/data/qm9"
    exit 1
fi

echo "=== Running Debug Planarity Experiment ==="
echo "This will test planarity constraint enforcement on QM9"

# Run the experiment
cd ConStruct
python main.py +experiment=debug/edge_deletion/planarity/qm9_debug_planar dataset.datadir=/home/drigoni/repository//repository//repository//repository/ConStruct-Thesis/data/qm9

echo "=== Experiment completed ==="
echo "Check the log files for results:"
echo "- SLURM output: /home/drigoni/repository//repository//repository//repository/ConStruct-Thesis/ConStruct/logs/qm9_debug_planar_$SLURM_JOB_ID.out"
echo "Time: $(date)" 