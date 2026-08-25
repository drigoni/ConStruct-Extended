
---
## Constrained Molecular Graph Generation with Diffusion Models: Extending the ConStruct Framework

### 🔬 **Fork Attribution & Extensions**

This repository is a **fork and extension** of the original [ConStruct](https://github.com/manuelmlmadeira/ConStruct) implementation by Madeira et al.

**Original Work**: [ConStruct - Generative Modelling of Structurally Constrained Graphs](https://github.com/manuelmlmadeira/ConStruct)  
**Original Authors**: Manuel Madeira et al.  
**Original Paper**: "Generative Modelling of Structurally Constrained Graphs"  
**License**: MIT

#### **Extensions in This Fork**:
- **Ring-Based Constraints**: Extended the model to include comprehensive ring count and ring length constraints
- **Molecular Dataset Focus**: Extensive testing and validation on molecular datasets, starting with QM9
- **Edge-Deletion Constraints**: Implemented "at most" constraints for ring count and ring length
- **Organized Experiment Structure**: Created systematic experiment configurations for debug and thesis-level testing
- **SLURM Integration**: Added comprehensive SLURM job scripts for cluster execution
- **Constraint Validation**: Implemented robust constraint satisfaction monitoring and validation

#### **Key Differences from Original**:
- **Constraint Types**: Added `ring_count_at_most` and `ring_length_at_most` projectors
- **Molecular Focus**: Optimized for molecular graph generation with QM9 dataset
- **Experiment Organization**: Structured configs and scripts for systematic constraint testing
- **Cluster Support**: Enhanced SLURM integration for high-performance computing environments

---

### 🚦 Bulletproof Environment Setup Instructions 
> 
> These steps are based on real-world cluster, GPU, RDKit, PyTorch, and graph-tool nightmares.
>  
> **You MUST follow the order and warnings below, or your environment will break.**

---

### 1. **Create and Activate Your Conda Environment**

```bash
conda create -y -c conda-forge -n construct python=3.10
conda activate construct
```

### 2. **Install graph-tool (optional)**

```bash
conda install -c conda-forge graph-tool=2.45
python -c "import graph_tool as gt"
```
⚠️ NOTE:
* *graph-tool* is **only required for non-molecular datasets** (e.g., tree, planar, lobster).
* If you work only with molecular datasets (QM9, etc.), you can skip installing graph-tool to avoid compatibility headaches.

### 3. **Install PyTorch (CUDA 11.8), then torch-geometric**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install torch_geometric rdkit fcd tabulate seaborn
# Optional dependencies:
# pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.12.0+cu126.html
python -c "import torch; print(torch.cuda.is_available())"
# Should print True if GPU is visible.
```



### 4. **Check RDKit Works**

```bash
python -c "from rdkit import Chem"
# No error means it's fine.
```

### 5. **Install the Rest of Your Requirements**

```bash
pip install -r requirements.txt
# (If requirements.txt has torch or rdkit, double check they don't get downgraded!)
```

### 6. **Install Your Own Package (Editable Dev Mode, If Needed)**

```bash
pip install -e .
```

### 7. **Compile ORCA if Needed**

```bash
cd ./ConStruct/analysis/orca
g++ -O2 -std=c++11 -o orca orca.cpp
cd -
```

### 9. **Test Everything**

```bash
python -c "import fcd; print(hasattr(fcd, 'load_ref_model'))"
python -c "import torch; print(torch.cuda.is_available())"
```

Both should print `True` or not error.

---

#### ⚠️ **CRITICAL WARNINGS!**

* **Never** use `pip install fcd` (without `--no-deps`) after torch/rdkit, or you’ll nuke your versions.
* **Never** install `cuda` libraries via conda. Cluster GPUs already have drivers.
* **Never** install both `fcd` and `fcd_torch` in the same env unless you know why.
* **Always** check for libstdc++ or libgomp errors (see troubleshooting below).

---

### 🔗 **Summary Table**

| Step                    | Command                                                                       |
| ----------------------- | ----------------------------------------------------------------------------- |
| Create env              | `conda create -y -c conda-forge -n construct python=3.9 rdkit=2023.03.2`      |
| Activate env            | `conda activate construct`                                                    |
| Install graph-tool      | `conda install -c conda-forge graph-tool=2.45`                                |
| Install PyTorch         | `pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118` |
| Install torch-geometric | `pip install torch-geometric==2.3.1`                                          |
| Install fcd             | `pip install --no-deps fcd`                                                   |
| Other packages          | `pip install -r requirements.txt`                                             |
| Your package            | `pip install -e .`                                                            |
| Compile ORCA            | `g++ -O2 -std=c++11 -o orca orca.cpp`                                         |

---

## 🆘 **Troubleshooting**

### fcd/rdkit libstdc++ error:

If you get something like
`ImportError: ... libstdc++.so.6: version 'GLIBCXX_3.4.29' not found ...`
run:

```bash
find $CONDA_PREFIX -name "libstdc++.so.6"
LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6 python -c "from rdkit import Chem; import fcd; print(hasattr(fcd, 'load_ref_model'))"
```

If it fixes things, **make it permanent**:

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"' > $CONDA_PREFIX/etc/conda/activate.d/zz_preload_libstdcxx.sh
chmod +x $CONDA_PREFIX/etc/conda/activate.d/zz_preload_libstdcxx.sh
```

### graph-tool/libgomp error:

If you see
`libgomp-a34b3233.so.1: version 'GOMP_5.0' not found (required by ...)`
run:

```bash
export LD_PRELOAD="$CONDA_PREFIX/lib/libgomp.so.1"
python test_env.py
```

If it works, make it permanent:

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_PRELOAD="$CONDA_PREFIX/lib/libgomp.so.1"' > $CONDA_PREFIX/etc/conda/activate.d/zz_preload_libgomp.sh
chmod +x $CONDA_PREFIX/etc/conda/activate.d/zz_preload_libgomp.sh
```

If still not working, add to every SLURM script after `conda activate`:

```bash
export LD_PRELOAD="$CONDA_PREFIX/lib/libgomp.so.1"
```

---

## **Quick Cluster Sanity Check Script**

Paste and run these one by one in your **(construct)** environment:

1. **RDKit Basic Import**

   ```bash
   python -c "from rdkit import Chem; print(Chem.MolFromSmiles('CCO') is not None)"
   ```
2. **PyTorch + CUDA Check**

   ```bash
   python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
   ```
3. **torch-geometric Check**

   ```bash
   python -c "import torch_geometric; print(torch_geometric.__version__)"
   ```
4. **fcd Import and Model Load**

   ```bash
   python -c "import fcd; print(hasattr(fcd, 'load_ref_model')); m = fcd.load_ref_model(); print(m is not None)"
   ```
5. **Your Own Package Import**

   ```bash
   python -c "import ConStruct; print('ConStruct imported\!')"
   ```
6. **(Optional) Try a minimal fcd score calculation**

   ```bash
   python -c "import fcd; s = fcd.get_fcd(['CCO', 'CCC'], ['CCO', 'CCN']); print('FCD score:', s)"
   ```

If all these work: **your env is cluster-proof**.

---

## Run the code

### 🚀 **Organized Experiment Structure**

The codebase includes a comprehensive, organized experiment structure for testing different constraint types. 

**Note**: Edge-insertion constraints are documented but not yet implemented in the current codebase.

#### **Directory Structure**
```
configs/experiment/
├── debug/                          # Debug-level experiments (quick testing)
│   ├── no_constraint/             # No constraint experiments
│   └── edge_deletion/             # Edge-deletion constraints ("at most")
│       ├── planarity/             # Planarity constraints
│       ├── ring_count_at_most/   # Ring count "at most" constraints
│       └── ring_length_at_most/  # Ring length "at most" constraints
└── thesis/                         # Thesis-level experiments (full-scale)
    ├── no_constraint/             # No constraint experiments
    └── edge_deletion/             # Edge-deletion constraints ("at most")
        ├── planarity/             # Planarity constraints
        ├── ring_count_at_most/   # Ring count "at most" constraints
        └── ring_length_at_most/  # Ring length "at most" constraints

ConStruct/slurm_jobs/
├── debug/                          # Debug-level SLURM scripts
│   ├── no_constraint/             # No constraint scripts
│   └── edge_deletion/             # Edge-deletion scripts
└── thesis/                         # Thesis-level SLURM scripts
    ├── no_constraint/             # No constraint scripts
    └── edge_deletion/             # Edge-deletion scripts
```

#### **Constraint Types**

**Edge-Deletion Constraints ("At Most")**:
- **Purpose**: Limit maximum ring count, ring length, or enforce planarity
- **Transition**: `absorbing_edges`
- **Projectors**: `ring_count_at_most`, `ring_length_at_most`, `planar`
- **Use Case**: Generate molecules with limited ring complexity or planar structures

**No Constraint**:
- **Purpose**: Baseline training without any constraints
- **Transition**: `absorbing_edges`
- **Projector**: `null`
- **Use Case**: Generate molecules without structural constraints

**Edge-Insertion Constraints ("At Least")**:
- **Status**: Implemented in the current codebase
- **Note**: Use the `edge_insertion` transition together with `ring_count_at_least` or `ring_length_at_least` projectors in the model configuration

### 🧪 **Running Experiments (some examples)**

#### **Direct Python Execution**
```bash
# Debug experiments
python ConStruct/main.py \
  --config-name experiment/debug/no_constraint/qm9_debug_no_constraint.yaml \
  --config-path configs/

# Ring count at most 2 (debug)
python ConStruct/main.py \
  --config-name experiment/debug/edge_deletion/ring_count_at_most/qm9_debug_ring_count_at_most_2.yaml \
  --config-path configs/

# Planarity constraint (debug)
python ConStruct/main.py \
  --config-name experiment/debug/edge_deletion/planarity/qm9_debug_planar.yaml \
  --config-path configs/

# Thesis experiments
python ConStruct/main.py \
  --config-name experiment/thesis/edge_deletion/ring_count_at_most/qm9_thesis_ring_count_at_most_3.yaml \
  --config-path configs/

# Ring length at most 5 (thesis)
python ConStruct/main.py \
  --config-name experiment/thesis/edge_deletion/ring_length_at_most/qm9_thesis_ring_length_at_most_5.yaml \
  --config-path configs/
```

#### **SLURM Job Submission**
```bash
# Debug experiments
sbatch ConStruct/slurm_jobs/debug/no_constraint/qm9_no_constraint_debug.slurm
sbatch ConStruct/slurm_jobs/debug/edge_deletion/ring_count_at_most/qm9_ring_count_at_most_2_debug.slurm
sbatch ConStruct/slurm_jobs/debug/edge_deletion/planarity/qm9_debug_planar.slurm

# Thesis experiments
sbatch ConStruct/slurm_jobs/thesis/no_constraint/qm9_no_constraint_thesis.slurm
sbatch ConStruct/slurm_jobs/thesis/edge_deletion/ring_count_at_most/qm9_ring_count_at_most_3_thesis.slurm
sbatch ConStruct/slurm_jobs/thesis/edge_deletion/ring_length_at_most/qm9_ring_length_at_most_5_thesis.slurm
```

### Checkpoint sampling without likelihood evaluation

Set `general.sampling_only=true` and provide the checkpoint through the existing
`general.test_only` setting. Sampling-only mode runs once with `train.seed`,
computes the normal `test_sampling/*` graph metrics, and never initializes or
logs to W&B.

```bash
# Sample with the configured projection enabled.
python main.py \
  +experiment=thesis/edge_insertion/ring_count_at_least/qm9_thesis_ring_count_at_least_2 \
  general.sampling_only=true \
  general.test_only=/absolute/path/to/model.ckpt

# Evaluate the same constraint target without enforcing its projection.
python main.py \
  +experiment=thesis/edge_insertion/ring_count_at_least/qm9_thesis_ring_count_at_least_2 \
  general.sampling_only=true \
  general.test_only=/absolute/path/to/model.ckpt \
  model.use_projection=false

# Override output location and sample/visualization counts.
python main.py \
  +experiment=thesis/edge_insertion/ring_count_at_least/qm9_thesis_ring_count_at_least_2 \
  general.sampling_only=true \
  general.test_only=/absolute/path/to/model.ckpt \
  general.sampling_output_dir=/absolute/path/to/sampling-run \
  general.final_model_samples_to_generate=1000 \
  general.final_model_samples_to_save=100 \
  general.final_model_chains_to_save=0
```

Every generated graph is stored in
`<sampling_output_dir>/generated_samples_rank<N>.pkl`. Rank zero also writes
`<sampling_output_dir>/sampling_metrics.json`, containing JSON-native metrics,
histograms, timing, checkpoint and seed provenance, the configured constraint
target, and whether projection was enabled. The existing `general.test_only`
five-seed likelihood-evaluation behavior is unchanged when
`general.sampling_only=false`.


---
