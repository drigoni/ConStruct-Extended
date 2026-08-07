# Experiment configurations

Experiments are composed with Hydra from `configs/config.yaml` and selected with
`+experiment=<path>`.

## Edge-addition lower-bound experiments

The `edge_insertion` forward process approaches a complete graph. Bond types on
that terminal graph are sampled from the dataset marginals after removing and
renormalizing the no-bond class. Reverse generation removes edges, and the
projector restores removals that would violate the configured lower bound.

Structural constraints use all unique simple graph cycles:

- `ring_count_at_least: K` means the graph contains at least `K` cycles.
- `ring_length_at_least: L` means the maximum cycle length is at least `L`.

Node counts are sampled from the empirical dataset distribution conditioned on
being large enough to satisfy the lower bound. Projectors enforce structural
properties only; RDKit validity, valency, and connectivity are evaluated after
generation.

Both QM9 and MOSES provide debug and thesis configurations for ring counts
`1, 2, 3` and maximum ring lengths `4, 5, 6` under:

```text
configs/experiment/{debug,thesis}/edge_insertion/
├── ring_count_at_least/
└── ring_length_at_least/
```

## Local examples

```bash
python main.py +experiment=debug/edge_insertion/ring_count_at_least/qm9_debug_ring_count_at_least_2
python main.py +experiment=debug/edge_insertion/ring_length_at_least/moses_debug_ring_length_at_least_5
python main.py +experiment=thesis/edge_insertion/ring_count_at_least/moses_thesis_ring_count_at_least_3
python main.py +experiment=thesis/edge_insertion/ring_length_at_least/qm9_thesis_ring_length_at_least_6
```

## SLURM examples

Matching launchers live under `slurm_jobs/{debug,thesis}/edge_insertion/`:

```bash
sbatch slurm_jobs/debug/edge_insertion/ring_count_at_least/qm9_ring_count_at_least_2_debug.slurm
sbatch slurm_jobs/thesis/edge_insertion/ring_length_at_least/moses_ring_length_at_least_5_thesis.slurm
```

Debug configurations run five epochs with 50 diffusion steps, small sampling
counts, disabled WandB, and no checkpoint saving. Thesis configurations inherit
the repository training and evaluation defaults and use the standard four-layer,
128-dimensional, 500-step model.
