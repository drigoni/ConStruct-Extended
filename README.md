
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

### Full-QM9 generalist with inference-time cycle constraints

Train a single checkpoint on unfiltered QM9. This profile uses the fully
connected `edge_insertion` terminal distribution but does not configure or run
a projector. It validates molecular validity every 10 epochs, keeps the
best-validity checkpoint, and stops after 30 validation checks without a strict
validity increase:

```bash
python main.py +experiment=training/qm9_no_constraint_edge_addition
```

Continue an existing run with the same validity-based stopping policy:

```bash
python main.py \
  +experiment=training/qm9_no_constraint_edge_addition \
  general.resume=/absolute/path/to/last.ckpt
```

At inference, `model.constraints` is the canonical list of structural targets.
The provided profiles cover the complete count setting `{disabled, 1, 2}` by
length setting `{disabled, 4, 5}` matrix:

```text
count_disabled_length_disabled
count_1_length_disabled       count_2_length_disabled
count_disabled_length_4       count_disabled_length_5
count_1_length_4              count_1_length_5
count_2_length_4              count_2_length_5
```

Load the same checkpoint for any cell. Profiles with a target enable
projection; the fully disabled profile is the unprojected baseline. Output is
written to `samples/qm9_no_constraint_edge_addition/<profile>/seed_<seed>`.

```bash
python main.py \
  +experiment=training/qm9_no_constraint_edge_addition \
  +constraint=count_2_length_5 \
  general.sampling_only=true \
  general.test_only=/absolute/path/to/model.ckpt
```

Run the initial one-seed, 10,000-sample matrix by invoking the command once per
profile with `train.seed=0` (the default). Change replication or sample count
without changing the checkpoint:

```bash
python main.py \
  +experiment=training/qm9_no_constraint_edge_addition \
  +constraint=count_1_length_4 \
  general.sampling_only=true \
  general.test_only=/absolute/path/to/model.ckpt \
  train.seed=1000 \
  general.final_model_samples_to_generate=2000
```

Thresholds remain ordinary Hydra settings. Because profile entries reference
them, `model.min_rings=3` or `model.min_ring_length=6` updates the corresponding
typed constraint. For an ad hoc configuration, set `model.constraints` to a
list containing objects of type `ring_count_at_least` with `min_rings`, type
`ring_length_at_least` with `min_ring_length`, or both. `model.use_projection`
controls enforcement independently from measurement.

Sampling metadata contains the canonical `constraints` array. Single-target
runs also retain the legacy singular `constraint` object. Joint runs report
ring-count, ring-length, and joint satisfaction and violation metrics.

### Resumable 3×3 projection matrix

After the checkpoint is available, one command generates any missing cells in
the predefined `{none, 1, 2}` ring-count by `{none, 4, 5}` maximum-cycle-length
grid and then cross-evaluates every cell against the same five targets:
molecular validity, ring count ≥1 and ≥2, and maximum cycle length ≥4 and ≥5.

```bash
python -m ConStruct.analysis.run_constraint_matrix \
  --checkpoint '/absolute/path/to/epoch=389.ckpt' \
  --seed 0 \
  --samples 10000
```

Generation runs the existing `main.py` sampling path once per profile in a
fresh process. A completed cell is reused only when its checkpoint, seed,
sample count, constraints, projection status, and rank artifacts match. The
metric phase reloads the saved graph batches and deterministically evaluates
exactly 10,000 graphs per cell, even if distributed generation created a few
extras.

The phases may also be run separately. Both commands are resumable:

```bash
python -m ConStruct.analysis.run_constraint_matrix \
  --checkpoint '/absolute/path/to/epoch=389.ckpt' \
  --phase generate

python -m ConStruct.analysis.run_constraint_matrix \
  --checkpoint '/absolute/path/to/epoch=389.ckpt' \
  --phase metrics
```

Each failed stage is attempted three times. A persistent failure exits with a
structured record under `matrix/seed_<seed>/errors/`; rerunning resumes from
that profile without repeating completed cells. To deliberately replace a
mismatched or completed cell, pass `--force-profile <profile>` during the
generation phase. Its old sample directory is archived rather than deleted.

The existing profile samples remain under
`samples/qm9_no_constraint_edge_addition/<profile>/seed_<seed>/`. Aggregate
artifacts are written under
`samples/qm9_no_constraint_edge_addition/matrix/seed_<seed>/`:

- `manifest.json` records checkpoint, seed, sample count, profiles, and phase
  completion.
- `cells/*.json` caches each profile's five post-hoc metrics.
- `metrics.json` and `metrics.csv` contain the complete matrix.
- `grids/*.md` contains five annotated 3×3 tables.
- `heatmaps/*.{png,pdf}` contains five fixed-scale 0–100 heatmaps.
- `logs/` and `errors/` preserve subprocess and failure diagnostics.

Only load generated pickle files produced locally by a trusted run; pickle is
not a safe interchange format for untrusted artifacts.

### Edge-deletion generalist and upper-bound matrix

Train the full-QM9 unconstrained `absorbing_edges` model with the same
validity-based early stopping policy:

```bash
python main.py +experiment=training/qm9_no_constraint_edge_absorbing
```

Then generate and cross-evaluate the `{none, ≤1, ≤2}` ring-count by
`{none, ≤4, ≤5}` maximum-cycle-length matrix:

```bash
python -m ConStruct.analysis.run_edge_deletion_matrix \
  --checkpoint '/absolute/path/to/model.ckpt' \
  --seed 0 \
  --samples 10000
```

The resumable samples, cell metrics, grids, and heatmaps are written below
`samples/qm9_no_constraint_edge_absorbing/`, independently from the
edge-insertion matrix.


---
