# 🧪 QM9 Constraint Testing Suite

## 📋 Overview
Complete testing suite for all QM9 molecular generation constraints using the **best no-constraint checkpoint** (`epoch-94.ckpt` with NLL: 67.92).

## 🎯 **Complete Constraint Coverage**

### **Ring Count Constraints** (6 tests)
| Constraint | File | Seed | Description |
|------------|------|------|-------------|
| `max_rings ≤ 0` | `test_ring_count_0.slurm` | 200 | Acyclic molecules only |
| `max_rings ≤ 1` | `test_ring_count_1.slurm` | 201 | Maximum 1 ring |
| `max_rings ≤ 2` | `test_ring_count_2.slurm` | 202 | Maximum 2 rings |
| `max_rings ≤ 3` | `test_ring_count_3.slurm` | 300 | Maximum 3 rings |
| `max_rings ≤ 4` | `test_ring_count_4.slurm` | 204 | Maximum 4 rings |
| `max_rings ≤ 5` | `test_ring_count_5.slurm` | 205 | Maximum 5 rings |

### **Ring Length Constraints** (6 tests)
| Constraint | File | Seed | Description |
|------------|------|------|-------------|
| `max_ring_length ≤ 3` | `test_ring_length_3.slurm` | 301 | 3-atom rings max |
| `max_ring_length ≤ 4` | `test_ring_length_4.slurm` | 302 | 4-atom rings max |
| `max_ring_length ≤ 5` | `test_ring_length_5.slurm` | 303 | 5-atom rings max |
| `max_ring_length ≤ 6` | `test_ring_length_6.slurm` | 500 | 6-atom rings max |
| `max_ring_length ≤ 7` | `test_ring_length_7.slurm` | 304 | 7-atom rings max |
| `max_ring_length ≤ 8` | `test_ring_length_8.slurm` | 305 | 8-atom rings max |

### **Other Constraints** (2 tests)
| Constraint | File | Seed | Description |
|------------|------|------|-------------|
| **No Constraint** | `test_no_constraint.slurm` | 100 | Baseline comparison |
| **Planar** | `test_planar.slurm` | 400 | Planarity constraint |

## 🚀 **How to Run**

### **Option 1: Run All Tests Sequentially**
```bash
sbatch ConStruct/slurm_jobs/checkpoint_testing/run_all_tests.slurm
```
- **Pros**: Organized, sequential execution
- **Cons**: Takes longer (runs one at a time)
- **Best for**: Systematic testing, thesis work

### **Option 2: Run Individual Tests**
```bash
# Test specific constraints
sbatch ConStruct/slurm_jobs/checkpoint_testing/test_ring_count_0.slurm
sbatch ConStruct/slurm_jobs/checkpoint_testing/test_planar.slurm
sbatch ConStruct/slurm_jobs/checkpoint_testing/test_ring_length_6.slurm
```

### **Option 3: Run Tests in Parallel (Advanced)**
```bash
# Submit multiple tests simultaneously
for script in ConStruct/slurm_jobs/checkpoint_testing/test_*.slurm; do
    sbatch "$script"
done
```
- **Pros**: Faster execution
- **Cons**: May overwhelm cluster, harder to track
- **Best for**: Quick testing when cluster is free

## 🔧 **Configuration Details**

### **Common Settings Across All Tests**
- **Checkpoint**: `checkpoints/qm9_thesis_no_constraint/epoch-94.ckpt`
- **Samples**: 10,000 molecules per test
- **WandB**: Enabled with testing notes
- **Seeds**: Removed (using default seeds)
- **GPU**: 1 GPU, 4 CPU cores, 16GB RAM
- **Time**: 4 hours per test

### **Constraint-Specific Settings**
```yaml
# Ring Count Tests
model.rev_proj: ring_count_at_most
model.max_rings: [0, 1, 2, 3, 4, 5]
model.use_incremental: true

# Ring Length Tests  
model.rev_proj: ring_length_at_most
model.max_ring_length: [3, 4, 5, 6, 7, 8]
model.use_incremental_length: true

# Planarity Test
model.rev_proj: planar

# No Constraint Test
model.rev_proj: null
```

## 📊 **Expected Outputs**

### **File Structure**
```
ConStruct/outputs/checkpoint_testing/
├── no_constraint/
├── ring_count_0/
├── ring_count_1/
├── ring_count_2/
├── ring_count_3/
├── ring_count_4/
├── ring_count_5/
├── ring_length_3/
├── ring_length_4/
├── ring_length_5/
├── ring_length_6/
├── ring_length_7/
├── ring_length_8/
└── planar/
```

### **WandB Integration**
- **Project**: Your QM9 thesis project
- **Tags**: "TESTING" prefix for easy identification
- **Notes**: Clear descriptions of each constraint test
- **Metrics**: Generation quality, constraint satisfaction, diversity

## 🎯 **Thesis Analysis Benefits**

### **Comprehensive Coverage**
- **14 different constraint configurations**
- **Systematic comparison** across constraint types
- **Baseline performance** (no constraint)
- **Constraint impact analysis**

### **Statistical Robustness**
- **10,000 samples** per constraint
- **Reproducible results** (using default seeds)
- **Variance estimation**

### **Professional Documentation**
- **WandB tracking** for all experiments
- **Organized outputs** by constraint type
- **Clear naming conventions**
- **Easy result comparison**

## ⚠️ **Important Notes**

### **Checkpoint Usage**
- All tests use the **same checkpoint** (`epoch-94.ckpt`)
- This is the **best performing** no-constraint model
- Projectors enforce constraints **during sampling only**
- No retraining required

### **Resource Management**
- **Sequential execution** recommended for thesis work
- **Parallel execution** possible but monitor cluster usage
- **4-hour time limit** per test (should be sufficient)
- **GPU memory**: 16GB should handle QM9 molecules

### **Monitoring Progress**
- **SLURM logs**: Check individual job outputs
- **WandB**: Real-time progress tracking
- **Output directories**: Results saved incrementally
- **Master script**: Overall progress summary

## 🚀 **Quick Start Commands**

```bash
# 1. Test the setup with no constraint
sbatch ConStruct/slurm_jobs/checkpoint_testing/test_no_constraint.slurm

# 2. Run all tests systematically
sbatch ConStruct/slurm_jobs/checkpoint_testing/run_all_tests.slurm

# 3. Check progress
squeue -u $USER
tail -f ConStruct/logs/checkpoint_testing/test_*.out

# 4. Monitor WandB
# Open your WandB project and look for "TESTING:" runs
```

## 📈 **Expected Timeline**
- **Individual test**: ~2-4 hours
- **All tests sequentially**: ~2-3 days
- **All tests in parallel**: ~4-6 hours (if cluster allows)

## 🎉 **Success Criteria**
- ✅ All 14 tests complete successfully
- ✅ 10,000 samples generated per constraint
- ✅ WandB runs properly tracked
- ✅ Output directories created and populated
- ✅ Constraint satisfaction verified
- ✅ Results ready for thesis analysis

---

**Created**: $(date)
**Checkpoint**: `epoch-94.ckpt` (NLL: 67.92)
**Total Tests**: 14
**Total Samples**: 140,000 molecules
**Purpose**: QM9 Constraint Analysis for Thesis 