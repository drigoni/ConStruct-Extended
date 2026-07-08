#!/bin/bash
#SBATCH --job-name=run_all_tests
#SBATCH --output=/home/drigoni/repository/ConStruct-Thesis/slurm_jobs/logs/checkpoint_testing/run_all_tests_%j.out
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

cd /home/drigoni/repository/ConStruct-Thesis

echo "=== Starting All Constraint Tests ==="
echo "Time: $(date)"
echo "Job ID: $SLURM_JOB_ID"

# Create results directory
mkdir -p ConStruct/slurm_jobs/checkpoint_testing/results/logs

# Function to submit job and wait for completion
submit_and_wait() {
    local job_name=$1
    local script_path=$2
    echo "Submitting $job_name..."
    
    # Submit the job
    local job_id=$(sbatch --parsable "$script_path")
    echo "  Submitted $job_name with job ID: $job_id"
    
    # Wait for completion
    echo "  Waiting for $job_name to complete..."
    while squeue -j $job_id | grep -q $job_id; do
        sleep 30
        echo "  $job_name still running... (job ID: $job_id)"
    done
    
    echo "  $job_name completed!"
    echo "  Logs available at: ConStruct/logs/checkpoint_testing/${job_name}_*.out"
    echo ""
}

# Submit all tests in sequence
echo "1. Testing No Constraint (Baseline)..."
submit_and_wait "test_no_constraint" "ConStruct/slurm_jobs/checkpoint_testing/test_no_constraint.slurm"

echo "2. Testing Ring Count ≤ 0..."
submit_and_wait "test_ring_count_0" "ConStruct/slurm_jobs/checkpoint_testing/test_ring_count_0.slurm"

echo "3. Testing Ring Count ≤ 1..."
submit_and_wait "test_ring_count_1" "ConStruct/slurm_jobs/checkpoint_testing/test_ring_count_1.slurm"

echo "4. Testing Ring Count ≤ 2..."
submit_and_wait "test_ring_count_2" "ConStruct/slurm_jobs/checkpoint_testing/test_ring_count_2.slurm"

echo "5. Testing Ring Count ≤ 3..."
submit_and_wait "test_ring_count_3" "ConStruct/slurm_jobs/checkpoint_testing/test_ring_count_3.slurm"

echo "6. Testing Ring Count ≤ 4..."
submit_and_wait "test_ring_count_4" "ConStruct/slurm_jobs/checkpoint_testing/test_ring_count_4.slurm"

echo "7. Testing Ring Count ≤ 5..."
submit_and_wait "test_ring_count_5" "ConStruct/slurm_jobs/checkpoint_testing/test_ring_count_5.slurm"

echo "8. Testing Ring Length ≤ 3..."
submit_and_wait "test_ring_length_3" "ConStruct/slurm_jobs/checkpoint_testing/test_ring_length_3.slurm"

echo "9. Testing Ring Length ≤ 4..."
submit_and_wait "test_ring_length_4" "ConStruct/slurm_jobs/checkpoint_testing/test_ring_length_4.slurm"

echo "10. Testing Ring Length ≤ 5..."
submit_and_wait "test_ring_length_5" "ConStruct/slurm_jobs/checkpoint_testing/test_ring_length_5.slurm"

echo "11. Testing Ring Length ≤ 6..."
submit_and_wait "test_ring_length_6" "ConStruct/slurm_jobs/checkpoint_testing/test_ring_length_6.slurm"

echo "12. Testing Ring Length ≤ 7..."
submit_and_wait "test_ring_length_7" "ConStruct/slurm_jobs/checkpoint_testing/test_ring_length_7.slurm"

echo "13. Testing Ring Length ≤ 8..."
submit_and_wait "test_ring_length_8" "ConStruct/slurm_jobs/checkpoint_testing/test_ring_length_8.slurm"

echo "14. Testing Planarity..."
submit_and_wait "test_planar" "ConStruct/slurm_jobs/checkpoint_testing/test_planar.slurm"

echo "=== All Tests Completed! ==="
echo "Time: $(date)"
echo ""
echo "Results Summary:"
echo "================"
echo "All tests used checkpoint: checkpoints/qm9_thesis_no_constraint/epoch=94.ckpt"
echo "All tests generated: 10,000 samples each"
echo "All tests saved to WandB with testing notes"
echo "All tests used unique seeds for reproducibility"
echo ""
echo "Output directories:"
echo "  - No constraint: outputs/checkpoint_testing/no_constraint/"
echo "  - Ring count tests: outputs/checkpoint_testing/ring_count_*/"
echo "  - Ring length tests: outputs/checkpoint_testing/ring_length_*/"
echo "  - Planarity: outputs/checkpoint_testing/planar/"
echo ""
echo "Logs: ConStruct/logs/checkpoint_testing/"
echo "WandB: Check your project for all test runs" 