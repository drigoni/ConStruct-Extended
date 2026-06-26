#!/bin/bash
#SBATCH --job-name=qm9_thesis_planar
#SBATCH --output=/home/drigoni/repository/ConStruct-Thesis/slurm_jobs/logs/thesis/qm9_thesis_planar_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --exclude=vgpu5-0
#SBATCH --time=48:00:00
#SBATCH --partition=testing

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

# CUDA optimizations
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_CUDNN_BENCHMARK=1

# Force unbuffered output for better logging
export PYTHONUNBUFFERED=1

# CUDA verification
python -c 'import torch; print("torch version:", torch.__version__); print("CUDA available:", torch.cuda.is_available()); print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")'

echo "=== Running QM9 Thesis Planarity Experiment ==="
echo "Config: thesis/edge_deletion/planarity/qm9_thesis_planar"
echo "Dataset: /home/drigoni/repository/ConStruct-Thesis/data/qm9"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"

# Verify QM9 dataset exists
echo "=== Checking QM9 dataset ==="
if [ -d "/home/drigoni/repository/ConStruct-Thesis/data/qm9" ]; then
    echo "✅ QM9 dataset found at /home/drigoni/repository/ConStruct-Thesis/data/qm9"
else
    echo "❌ QM9 dataset not found at /home/drigoni/repository/ConStruct-Thesis/data/qm9"
    exit 1
fi

echo "=== Running Thesis Planarity Experiment ==="
echo "This will test planarity constraint enforcement on QM9 with full thesis settings"

# Run the experiment
cd ConStruct
python main.py +experiment=thesis/edge_deletion/planarity/qm9_thesis_planar dataset.datadir=/home/drigoni/repository/ConStruct-Thesis/data/qm9

echo "=== Experiment completed ==="
echo "Check the log files for results:"
echo "- SLURM output: /home/drigoni/repository/ConStruct-Thesis/slurm_jobs/logs/qm9_thesis_planar_$SLURM_JOB_ID.out"
echo "Time: $(date)" 