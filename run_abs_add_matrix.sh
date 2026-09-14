#!/bin/sh

echo 'Running constraint matrix analysis...'

python -m ConStruct.analysis.run_constraint_matrix \
 --checkpoint checkpoints/qm9_no_constraint_edge_addition/epoch=1169.ckpt \
 --seed 0 \
 --samples 10000 \
 --experiment training/qm9_no_constraint_edge_addition_large \
 --output-root samples/qm9_no_constraint_edge_addition_large

python -m ConStruct.analysis.run_edge_deletion_matrix \
 --checkpoint checkpoints/qm9_no_constraint_edge_absorbing/epoch=679.ckpt \
 --seed 0 \
 --samples 10000 \
 --experiment training/qm9_no_constraint_edge_absorbing_large \
 --output-root samples/qm9_no_constraint_edge_absorbing_large
