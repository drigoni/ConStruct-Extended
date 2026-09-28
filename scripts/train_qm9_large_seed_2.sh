#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(dirname -- "$SCRIPT_DIR")
cd "$REPO_ROOT"

python main.py +experiment=training/seeded/qm9_no_constraint_edge_absorbing_large_seed_2
python main.py +experiment=training/seeded/qm9_no_constraint_edge_addition_large_seed_2
