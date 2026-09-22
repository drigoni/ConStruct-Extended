#!/usr/bin/env python3
"""Plot QM9 training simple-cycle statistics, normalized by training size."""
from collections import Counter
from pathlib import Path
import csv
import json
import numpy as np
import torch
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from ConStruct.projector.graph_cycles import enumerate_simple_cycles_unique

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'data/qm9/processed/train_noh.pt'

def main():
    data, slices = torch.load(SOURCE, map_location='cpu', weights_only=False)
    n = len(slices['x']) - 1
    edge_offsets = slices['edge_index'].tolist()
    node_offsets = slices['x'].tolist()
    edges = data.edge_index.numpy()
    joint = Counter()
    cache = {}
    for i in range(n):
        num_nodes = node_offsets[i + 1] - node_offsets[i]
        mol_edges = edges[:, edge_offsets[i]:edge_offsets[i + 1]]
        # PyG stores each undirected bond twice, with local node indices.
        undirected = tuple(sorted((int(u), int(v)) for u, v in mol_edges.T if u < v))
        key = (num_nodes, undirected)
        if key not in cache:
            graph = nx.Graph()
            graph.add_nodes_from(range(num_nodes))
            graph.add_edges_from(undirected)
            cycles = list(enumerate_simple_cycles_unique(graph))
            cache[key] = (len(cycles), max((len(c) for c in cycles), default=0))
        joint[cache[key]] += 1
        if (i + 1) % 20000 == 0:
            print(f'Processed {i + 1:,}/{n:,} molecules', flush=True)
    counts = np.zeros((max(y for x, y in joint) + 1, max(x for x, y in joint) + 1), dtype=int)
    for (x, y), count in joint.items():
        counts[y, x] = count
    probability = counts / n
    marginal_x = probability.sum(axis=0)
    marginal_y = probability.sum(axis=1)
    assert counts.sum() == n
    assert np.isclose(probability.sum(), 1)
    assert np.isclose(marginal_x.sum(), 1) and np.isclose(marginal_y.sum(), 1)
    assert counts[0, 0] == counts[0, :].sum() == counts[:, 0].sum()
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig = plt.figure(figsize=(13, 9))
    grid = fig.add_gridspec(2, 3, width_ratios=[5, 1.65, 0.16], height_ratios=[1.5, 5], hspace=0.08, wspace=0.12)
    ax = fig.add_subplot(grid[1, 0])
    top = fig.add_subplot(grid[0, 0], sharex=ax)
    right = fig.add_subplot(grid[1, 1], sharey=ax)
    cax = fig.add_subplot(grid[1, 2])
    im = ax.imshow(probability, origin='lower', cmap='YlGnBu', vmin=0, aspect='auto', interpolation='nearest')
    ax.set(xlabel='Number of cycles', ylabel='Maximum cycle length (atoms)', xticks=np.arange(counts.shape[1]), yticks=np.arange(counts.shape[0]))
    ax.set_xlim(-0.5, 14.5)
    ax.set_xticks(np.arange(15))
    ax.set_xticks(np.arange(-0.5, 15, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, counts.shape[0], 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.6)
    ax.tick_params(which='minor', bottom=False, left=False)
    for y, x in zip(*np.nonzero(counts)):
        if x > 14:
            continue
        value = probability[y, x]
        label = f'{value:.3f}' if value >= 0.001 else f'{value:.1e}'
        ax.text(x, y, label, ha='center', va='center', fontsize=8, color='white' if value > probability.max()*0.55 else '#172b3a')
    top.bar(np.arange(len(marginal_x)), marginal_x, width=0.8, color='#287b8e')
    top.set_ylabel('Fraction of training set')
    top.tick_params(axis='x', labelbottom=False, bottom=False)
    top.yaxis.set_major_locator(MaxNLocator(3))
    top.grid(axis='y', alpha=0.2)
    top.set_axisbelow(True)
    right.barh(np.arange(len(marginal_y)), marginal_y, height=0.8, color='#287b8e')
    right.set_xlabel('Fraction of training set')
    right.tick_params(axis='y', labelleft=False, left=False)
    right.xaxis.set_major_locator(MaxNLocator(3))
    right.grid(axis='x', alpha=0.2)
    right.set_axisbelow(True)
    fig.colorbar(im, cax=cax, label='Fraction of training set')
    fig.suptitle(f'QM9 training set: cycle count and maximum cycle length\nN = {n:,} molecules', fontsize=16, y=0.98)
    fig.text(0.1, 0.025, 'All unique undirected simple cycles; acyclic molecules have maximum length 0.\nHeatmap cells and marginal bars are counts / N; full distributions sum to 1; x-axis displays counts 0–14.', fontsize=10, color='#444444')
    fig.subplots_adjust(left=0.09, right=0.91, bottom=0.13, top=0.87)
    stem = ROOT / 'qm9_train_cycle_heatmap'
    fig.savefig(stem.with_suffix('.png'), dpi=300, bbox_inches='tight')
    fig.savefig(stem.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    with stem.with_suffix('.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['number_of_cycles', 'max_cycle_length', 'count', 'fraction_of_training_set'])
        for y in range(counts.shape[0]):
            for x in range(counts.shape[1]):
                writer.writerow([x, y, counts[y, x], probability[y, x]])
    stem.with_suffix('.json').write_text(json.dumps({'source':str(SOURCE.relative_to(ROOT)), 'dataset_size':n, 'cycle_definition':'All unique undirected simple cycles (length >= 3)', 'acyclic_max_length':0, 'normalization':'count / training dataset size', 'number_of_cycles_marginal':marginal_x.tolist(), 'max_cycle_length_marginal':marginal_y.tolist(), 'joint_sum':float(probability.sum())}, indent=2)+'\n')
    print(f'Saved {stem.name}.png and .pdf; N={n:,}, joint sum={probability.sum():.12f}, unique topologies={len(cache):,}', flush=True)

if __name__ == '__main__':
    main()
