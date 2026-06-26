# 🧪 Checkpoint Testing Infrastructure

This directory contains scripts and configurations for testing trained ConStruct model checkpoints without retraining. It's designed to evaluate model performance, generate samples, and measure constraint satisfaction.

## 🎯 Purpose

After training experiments complete, this infrastructure allows you to:

1. **Load trained checkpoints** and run evaluation pipelines
2. **Generate molecular samples** for analysis
3. **Measure constraint satisfaction** (ring counts, ring lengths, etc.)
4. **Compute comprehensive metrics** (FCD, Valid, Unique, Novel, etc.)
5. **Compare results** across different constraint configurations
6. **Save detailed logs** and metrics for analysis

## 📁 Directory Structure

```
ConStruct/slurm_jobs/checkpoint_testing/
├── README.md                           # This file
├── run_test_no_constraint.slurm       # No constraint testing SLURM script
├── run_test_ring_count_at_most.slurm  # Ring count constraint testing
├── run_test_ring_length_at_most.slurm # Ring length constraint testing
├── configs/                           # Testing configurations
│   ├── no_constraint_test.yaml       # No constraint testing config
│   ├── ring_count_at_most_test.yaml  # Ring count testing config
│   └── ring_length_at_most_test.yaml # Ring length testing config
└── results/                           # Auto-created results folder
    ├── YYYYMMDD_HHMMSS_experiment/   # Timestamped result folders
    ├── logs/                          # Detailed execution logs
    └── metrics/                       # Computed metrics and outputs
```

## 🚀 **Quick Start (Recommended)**

### **1. One-Command Testing**
```bash
# Test no constraint model (automatically finds best checkpoint)
python ConStruct/slurm_jobs/checkpoint_testing/quick_test.py qm9_thesis_no_constraint

# Test ring count constraint (automatically finds best checkpoint)
python ConStruct/slurm_jobs/checkpoint_testing/quick_test.py qm9_thesis_ring_count_at_most_3

# Test ring length constraint (automatically finds best checkpoint)
python ConStruct/slurm_jobs/checkpoint_testing/quick_test.py qm9_thesis_ring_length_at_most_6
```

### **2. Manual Testing with Smart Checkpoint Selection**
```bash
# Test the best model (recommended)
sbatch ConStruct/slurm_jobs/checkpoint_testing/run_test_no_constraint.slurm \
    --ckpt_path ConStruct/checkpoints/qm9_thesis_no_constraint \
    --config_path ConStruct/slurm_jobs/checkpoint_testing/configs/no_constraint_test.yaml \
    --ckpt_type best

# Test the last model (for comparison)
sbatch ConStruct/slurm_jobs/checkpoint_testing/run_test_no_constraint.slurm \
    --ckpt_path ConStruct/checkpoints/qm9_thesis_no_constraint \
    --config_path ConStruct/slurm_jobs/checkpoint_testing/configs/no_constraint_test.yaml \
    --ckpt_type last

# Test ring count constraint with best checkpoint
sbatch ConStruct/slurm_jobs/checkpoint_testing/run_test_ring_count_at_most.slurm \
    --ckpt_path ConStruct/checkpoints/qm9_thesis_ring_count_at_most_3 \
    --config_path ConStruct/slurm_jobs/checkpoint_testing/configs/ring_count_at_most_test.yaml \
    --max_rings 3 \
    --ckpt_type best
```

### **3. Manual Testing with Best Checkpoint**
```bash
# First, find the best checkpoint
python ConStruct/slurm_jobs/checkpoint_testing/identify_best_checkpoint.py \
    --experiment qm9_thesis_no_constraint

# Then use the recommended checkpoint path in your testing
```

## 🔧 **SLURM Script Features**

### **Automatic Setup**
- Creates timestamped results folders automatically
- Sets up proper Python environment (`construct-env`)
- Configures CUDA and GPU resources
- Handles logging and output management

### **Smart Checkpoint Selection**
- **`--ckpt_type best`**: Automatically finds the best checkpoint (lowest NLL)
- **`--ckpt_type last`**: Automatically finds the last checkpoint
- **Directory Input**: Can pass experiment directory instead of specific checkpoint file
- **Fallback Support**: Works with both new clean structure and old nested structure

### **Flexible Parameters**
- `--ckpt_path`: Path to trained checkpoint file OR experiment directory
- `--config_path`: YAML configuration file
- `--ckpt_type`: 'best' (default) or 'last' checkpoint
- `--batch_size`: Batch size for testing (default: 128)
- `--num_samples`: Number of samples to generate (default: 10000 - matches thesis experiments)
- `--gpu_memory`: GPU memory allocation (default: 16G)

### **Resource Management**
- GPU: 1x A6000/A5000 (configurable)
- Memory: 16GB RAM (configurable)
- Time: 4 hours (configurable)
- CPU: 4 cores (configurable)

## 📊 **Testing Configurations**

### **Smart Config Inheritance**
- **Inherits from existing defaults**: `general_default.yaml`, `train_default.yaml`
- **No parameter duplication**: Reuses your existing training settings
- **Easy maintenance**: Update defaults once, affects all test configs
- **Consistent behavior**: Test configs match your training setup

### **No Constraint Testing**
- **Purpose**: Baseline molecular generation without constraints
- **Config**: `no_constraint_test.yaml`
- **Use Case**: Compare against constrained models

### **Ring Count Testing**
- **Purpose**: Test models with maximum ring count constraints
- **Config**: `ring_count_at_most_test.yaml`
- **Parameters**: `max_rings` (0, 1, 2, 3, 4, 5)
- **Use Case**: Evaluate ring complexity control

### **Ring Length Testing**
- **Purpose**: Test models with maximum ring length constraints
- **Config**: `ring_length_at_most_test.yaml`
- **Parameters**: `max_ring_length` (3, 4, 5, 6, 7, 8)
- **Use Case**: Evaluate ring size control

## 📈 **Output and Metrics**

### **Generated Files**
- **Logs**: Detailed execution logs with timestamps
- **Metrics**: Core and structural molecular metrics
- **Samples**: Generated molecular structures (if enabled)
- **Reports**: Comprehensive evaluation reports

### **Output Consistency**
- **Same format as training**: Metrics and sample outputs match training-time sampling
- **Direct comparison**: Results are directly comparable with training experiments
- **Standardized naming**: Consistent file and folder naming conventions
- **Thesis experiment defaults**: Uses same sample counts and settings as your thesis

### **Core Metrics**
- **FCD**: Fréchet ChemNet Distance
- **Valid**: Validity percentage
- **Unique**: Uniqueness percentage
- **Novel**: Novelty percentage

### **Structural Metrics**
- **Ring Count Distribution**: Actual vs. expected ring counts
- **Ring Length Distribution**: Actual vs. expected ring lengths
- **Constraint Satisfaction**: Percentage of samples meeting constraints

## 🎛️ Customization

### **Modifying SLURM Scripts**
Edit the variables at the top of each SLURM script:
```bash
# Job configuration
JOB_NAME="test_experiment_name"
GPU_MEMORY="16G"
TIME_LIMIT="04:00:00"
BATCH_SIZE=128
NUM_SAMPLES=1000
```

### **Modifying YAML Configs**
Edit the testing configurations in `configs/`:
```yaml
general:
  name: "custom_test_name"
  test_only: "path/to/checkpoint.ckpt"

model:
  # Model-specific parameters
  max_rings: 3
  max_ring_length: 5
```

## 🔍 Troubleshooting

### **Common Issues**
1. **Checkpoint not found**: Verify `--ckpt_path` is correct
2. **Config not found**: Verify `--config_path` exists
3. **GPU memory issues**: Reduce `--batch_size` or increase `--gpu_memory`
4. **Time limit exceeded**: Increase `--time_limit` in SLURM script

### **Debug Mode**
Add `--debug` flag to SLURM scripts for verbose logging:
```bash
sbatch ConStruct/slurm_jobs/checkpoint_testing/run_test_no_constraint.slurm \
    --ckpt_path ConStruct/checkpoints/debug.ckpt \
    --config_path ConStruct/slurm_jobs/checkpoint_testing/configs/no_constraint_test.yaml \
    --debug
```

## 📚 Related Documentation

- **Training Guide**: See `ConStruct/slurm_jobs/EXPERIMENT_GUIDE.md`
- **Constraint Implementation**: See `reports/RING_CONSTRAINTS_IMPLEMENTATION_SUMMARY.md`
- **Project Overview**: See `README.md`

## �� Contributing

When adding new constraint types or testing scenarios:
1. Create corresponding SLURM script
2. Create corresponding YAML config
3. Update this README with new examples
4. Test with existing checkpoints

---

**Last Updated**: $(date +%Y-%m-%d)
**ConStruct Version**: Thesis Implementation 