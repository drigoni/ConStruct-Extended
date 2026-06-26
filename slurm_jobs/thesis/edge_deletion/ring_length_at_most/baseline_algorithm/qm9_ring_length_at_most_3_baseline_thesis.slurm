#!/bin/bash
#SBATCH --job-name=qm9_ring_length_at_most_3_thesis_baseline
#SBATCH --output=/home/drigoni/repository/ConStruct-Thesis/slurm_jobs/logs/qm9_ring_length_at_most_3_thesis_baseline_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --exclude=vgpu5-0
#SBATCH --constraint=A6000|A5000
#SBATCH --time=48:00:00
#SBATCH --partition=allgroups,testing

# Load environment
cd /home/drigoni/repository/ConStruct-Thesis

# Only activate conda if not already in the right environment
if [ "$CONDA_DEFAULT_ENV" != "construct-env" ]; then
    source /conf/shared-software/anaconda/etc/profile.d/conda.sh
    conda activate construct-env
fi

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

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

echo "=== Running Experiment: ring_length_at_most_3_thesis ==="
echo "Config: thesis/edge_deletion/ring_length_at_most/qm9_thesis_ring_length_at_most_3"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"

# Run the experiment
cd ConStruct
python main.py +experiment=thesis/edge_deletion/ring_length_at_most/qm9_thesis_ring_length_at_most_3 model.use_incremental_length=False

echo "Experiment completed at $(date)"
