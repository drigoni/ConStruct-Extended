#!/usr/bin/env python3
"""
Complete QM9 Dataset Analysis Script
====================================

This script performs comprehensive analysis of the QM9 dataset including:
1. TRAINING DATASET ANALYSIS - What the model learns from
2. VALIDATION DATASET ANALYSIS - Model validation data
3. TEST DATASET ANALYSIS - Model testing data
4. FULL DATASET ANALYSIS - Complete dataset characterization
5. COMPARISON ANALYSIS - All dataset splits differences

Features:
- Ring count and ring length analysis
- Planarity analysis  
- Constraint satisfaction rates
- Visualization plots
- Detailed reports for each analysis type

All results are saved to the qm9_analysis directory structure.
"""

import os
import sys
import numpy as np
import networkx as nx
from collections import Counter
import matplotlib.pyplot as plt
import json
from pathlib import Path
import time

# Add ConStruct to path
sys.path.append('/home/drigoni/repository/ConStruct-Thesis')

from ConStruct.projector.graph_cycles import enumerate_simple_cycles_unique

def create_output_dir():
    """Create output directory structure"""
    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)
    
    # Create subdirectories for different analysis types
    (output_dir / "plots").mkdir(exist_ok=True)
    (output_dir / "reports").mkdir(exist_ok=True)
    (output_dir / "data").mkdir(exist_ok=True)
    (output_dir / "train_analysis").mkdir(exist_ok=True)
    (output_dir / "val_analysis").mkdir(exist_ok=True)
    (output_dir / "test_analysis").mkdir(exist_ok=True)
    (output_dir / "full_analysis").mkdir(exist_ok=True)
    (output_dir / "comparison").mkdir(exist_ok=True)
    
    return output_dir

def load_qm9_dataset(split="train"):
    """Load QM9 dataset using ConStruct's proper loading mechanism"""
    try:
        from ConStruct.datasets.qm9_dataset import QM9Dataset
        
        print(f"🔍 Loading QM9 dataset ({split} split) using ConStruct's dataset loader...")
        
        # Load specified split
        dataset = QM9Dataset(
            split=split,
            root="/home/drigoni/repository/ConStruct-Thesis/data/qm9",
            remove_h=True,
            target_prop=None,
            transform=None,
            pre_transform=None,
            pre_filter=None,
        )
        
        print(f"✅ Loaded {split} dataset with {len(dataset)} individual molecules")
        return dataset
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def load_full_qm9_datasets():
    """Load all QM9 dataset splits"""
    try:
        from ConStruct.datasets.qm9_dataset import QM9Dataset
        
        print("🔍 Loading full QM9 dataset (train + val + test)...")
        
        datasets = {}
        for split in ["train", "val", "test"]:
            dataset = QM9Dataset(
                split=split,
                root="/home/drigoni/repository/ConStruct-Thesis/data/qm9",
                remove_h=True,
                target_prop=None,
                transform=None,
                pre_transform=None,
                pre_filter=None,
            )
            datasets[split] = dataset
            print(f"✅ Loaded {split} split: {len(dataset)} molecules")
        
        return datasets
        
    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        return None

def analyze_ring_properties(dataset, dataset_name="dataset"):
    """Analyze ring count and ring length properties"""
    print(f"\n🔍 Analyzing ring properties for {dataset_name}...")
    
    ring_count_dist = Counter()
    ring_length_dist = Counter()  # Now distribution of molecules by max ring length
    max_ring_length_per_mol = []
    total_rings = 0
    
    for i, mol in enumerate(dataset):
        if i % 1000 == 0:
            print(f"  Processing molecule {i}/{len(dataset)}")
        
        # Convert PyTorch Geometric Data to NetworkX graph
        edge_index = mol.edge_index.cpu().numpy()
        num_nodes = mol.x.shape[0]
        
        # Create adjacency matrix
        adj_matrix = np.zeros((num_nodes, num_nodes))
        for edge in edge_index.T:
            adj_matrix[edge[0], edge[1]] = 1
            adj_matrix[edge[1], edge[0]] = 1
        
        nx_graph = nx.from_numpy_array(adj_matrix)
        
        # Count rings using all simple cycles (all simple rings)
        cycles = list(enumerate_simple_cycles_unique(nx_graph))
        ring_count = len(cycles)
        ring_count_dist[ring_count] += 1
        total_rings += ring_count
        
        # Analyze maximum ring length per molecule (for consistency with sampling evaluation)
        max_length_in_mol = 0
        for cycle in cycles:
            ring_length = len(cycle)
            max_length_in_mol = max(max_length_in_mol, ring_length)
        
        # Count molecules by their maximum ring length (0 for acyclic molecules)
        ring_length_dist[max_length_in_mol] += 1
        max_ring_length_per_mol.append(max_length_in_mol)
    
    return ring_count_dist, ring_length_dist, max_ring_length_per_mol, total_rings

def analyze_planarity(dataset, dataset_name="dataset"):
    """Analyze planarity properties"""
    print(f"\n🔍 Analyzing planarity for {dataset_name}...")
    
    try:
        from ConStruct.projector.is_planar import is_planar
        planar_count = 0
        non_planar_count = 0
        
        for i, mol in enumerate(dataset):
            if i % 1000 == 0:
                print(f"  Processing molecule {i}/{len(dataset)}")
            
            # Convert PyTorch Geometric Data to NetworkX graph
            edge_index = mol.edge_index.cpu().numpy()
            num_nodes = mol.x.shape[0]
            
            # Create adjacency matrix
            adj_matrix = np.zeros((num_nodes, num_nodes))
            for edge in edge_index.T:
                adj_matrix[edge[0], edge[1]] = 1
                adj_matrix[edge[1], edge[0]] = 1
            
            nx_graph = nx.from_numpy_array(adj_matrix)
            
            if is_planar(nx_graph):
                planar_count += 1
            else:
                non_planar_count += 1
        
        return planar_count, non_planar_count
    except ImportError:
        print("⚠️ Planarity analysis not available (is_planar module not found)")
        return None, None

def calculate_constraint_rates(ring_count_dist, ring_length_dist, max_ring_length_per_mol):
    """Calculate rates for different constraint thresholds"""
    print("\n📊 Calculating constraint satisfaction rates...")
    
    total_molecules = sum(ring_count_dist.values())
    
    # Ring count rates (≤0, ≤1, ≤2, ≤3, ≤4, ≤5, ≤6, ≤7, ≤8, ≤9, >9)
    ring_count_rates = {}
    for max_rings in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        count = sum(ring_count_dist[i] for i in range(max_rings + 1))
        rate = count / total_molecules
        ring_count_rates[f"≤{max_rings}"] = (rate, count)
        print(f"  Ring count ≤{max_rings}: {count}/{total_molecules} = {rate:.3f} ({rate*100:.3f}%)")
    
    # Add >9 category
    count_greater_than_9 = total_molecules - sum(ring_count_dist[i] for i in range(10))
    rate_greater_than_9 = count_greater_than_9 / total_molecules
    ring_count_rates[">9"] = (rate_greater_than_9, count_greater_than_9)
    print(f"  Ring count >9: {count_greater_than_9}/{total_molecules} = {rate_greater_than_9:.3f} ({rate_greater_than_9*100:.3f}%)")
    
    # Ring length rates (≤0, ≤3, ≤4, ≤5, ≤6, ≤7, ≤8, ≤9, >9)
    ring_length_rates = {}
    for max_length in [0, 3, 4, 5, 6, 7, 8, 9]:
        count = sum(1 for max_len in max_ring_length_per_mol if max_len <= max_length)
        rate = count / total_molecules
        ring_length_rates[f"≤{max_length}"] = (rate, count)
        print(f"  Ring length ≤{max_length}: {count}/{total_molecules} = {rate:.3f} ({rate*100:.3f}%)")
    
    # Add >9 category for ring length
    count_length_greater_than_9 = sum(1 for max_len in max_ring_length_per_mol if max_len > 9)
    rate_length_greater_than_9 = count_length_greater_than_9 / total_molecules
    ring_length_rates[">9"] = (rate_length_greater_than_9, count_length_greater_than_9)
    print(f"  Ring length >9: {count_length_greater_than_9}/{total_molecules} = {rate_length_greater_than_9:.3f} ({rate_length_greater_than_9*100:.3f}%)")
    
    return ring_count_rates, ring_length_rates

def create_plots(ring_count_dist, ring_length_dist, planar_count, non_planar_count, 
                ring_count_rates, ring_length_rates, output_dir, analysis_name="analysis"):
    """Create comprehensive visualization plots"""
    print(f"\n📈 Creating visualization plots for {analysis_name}...")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    
    # 1. Ring Count Distribution
    plt.figure(figsize=(10, 6))
    ring_counts = sorted(ring_count_dist.keys())
    ring_count_values = [ring_count_dist[c] for c in ring_counts]
    bars = plt.bar(ring_counts, ring_count_values, alpha=0.7, color='skyblue', edgecolor='navy')
    plt.title(f'QM9 {analysis_name}: Ring Count Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Rings', fontsize=12)
    plt.ylabel('Number of Molecules', fontsize=12)
    plt.xticks(ring_counts)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, ring_count_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(ring_count_values)*0.01,
                f'{value:,}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"ring_count_distribution_{analysis_name}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Ring Length Distribution (by maximum ring length per molecule)
    plt.figure(figsize=(10, 6))
    ring_lengths = sorted(ring_length_dist.keys())
    ring_length_values = [ring_length_dist[l] for l in ring_lengths]
    bars = plt.bar(ring_lengths, ring_length_values, alpha=0.7, color='lightcoral', edgecolor='darkred')
    plt.title(f'QM9 {analysis_name}: Maximum Ring Length Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Maximum Ring Length (Number of Atoms)', fontsize=12)
    plt.ylabel('Number of Molecules', fontsize=12)
    plt.xticks(ring_lengths)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, ring_length_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(ring_length_values)*0.01,
                f'{value:,}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"ring_length_distribution_{analysis_name}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Planarity Distribution
    if planar_count is not None:
        plt.figure(figsize=(8, 8))
        labels = ['Planar', 'Non-Planar']
        sizes = [planar_count, non_planar_count]
        colors = ['lightgreen', 'lightcoral']
        explode = (0.05, 0.05)
        
        wedges, texts, autotexts = plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                                           autopct='%1.1f%%', startangle=90, shadow=True)
        plt.title(f'QM9 {analysis_name}: Planarity Distribution', fontsize=14, fontweight='bold')
        
        # Make text more readable
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"planarity_distribution_{analysis_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Constraint Satisfaction Rates
    plt.figure(figsize=(12, 8))
    
    # Ring count rates
    plt.subplot(2, 1, 1)
    ring_count_constraints = ['≤0', '≤1', '≤2', '≤3', '≤4', '≤5', '≤6', '≤7', '≤8', '≤9', '>9']
    ring_count_rates_list = [ring_count_rates[c][0] for c in ring_count_constraints]
    
    bars = plt.bar(ring_count_constraints, ring_count_rates_list, alpha=0.7, color='gold', edgecolor='orange')
    plt.title(f'Ring Count Constraint Satisfaction Rates ({analysis_name})', fontsize=12, fontweight='bold')
    plt.ylabel('Satisfaction Rate', fontsize=10)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    
    # Add percentage labels
    for bar, rate in zip(bars, ring_count_rates_list):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{rate:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Ring length rates
    plt.subplot(2, 1, 2)
    ring_length_constraints = ['≤0', '≤3', '≤4', '≤5', '≤6', '≤7', '≤8', '≤9', '>9']
    ring_length_rates_list = [ring_length_rates[c][0] for c in ring_length_constraints]
    
    bars = plt.bar(ring_length_constraints, ring_length_rates_list, alpha=0.7, color='lightblue', edgecolor='navy')
    plt.title(f'Ring Length Constraint Satisfaction Rates ({analysis_name})', fontsize=12, fontweight='bold')
    plt.xlabel('Constraint', fontsize=10)
    plt.ylabel('Satisfaction Rate', fontsize=10)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    
    # Add percentage labels
    for bar, rate in zip(bars, ring_length_rates_list):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{rate:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / f"constraint_satisfaction_rates_{analysis_name}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved plots to {output_dir}/")

def save_results(ring_count_dist, ring_length_dist, ring_count_rates, ring_length_rates, 
                planar_count, non_planar_count, total_rings, total_molecules, output_dir, analysis_name="analysis"):
    """Save analysis results to files"""
    print(f"\n💾 Saving {analysis_name} results...")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare results dictionary
    results = {
        'analysis_type': analysis_name,
        'dataset_info': {
            'total_molecules': total_molecules,
            'total_rings': total_rings,
            'avg_rings_per_molecule': total_rings / total_molecules if total_molecules > 0 else 0
        },
        'ring_count_distribution': dict(ring_count_dist),
        'ring_length_distribution': dict(ring_length_dist),
        'ring_count_rates': ring_count_rates,
        'ring_length_rates': ring_length_rates,
        'planarity': {
            'planar_molecules': planar_count,
            'non_planar_molecules': non_planar_count,
            'planarity_rate': planar_count / (planar_count + non_planar_count) if planar_count is not None else None
        }
    }
    
    # Save as JSON
    with open(output_dir / f"qm9_{analysis_name}_analysis.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save detailed text report
    with open(output_dir / f"qm9_{analysis_name}_analysis_report.txt", 'w') as f:
        f.write(f"QM9 {analysis_name.title()} Analysis Report\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("DATASET OVERVIEW\n")
        f.write("-" * 20 + "\n")
        f.write(f"Analysis type: {analysis_name}\n")
        f.write(f"Total molecules analyzed: {total_molecules:,}\n")
        f.write(f"Total rings found: {total_rings:,}\n")
        f.write(f"Average rings per molecule: {total_rings/total_molecules:.3f}\n\n")
        
        f.write("RING COUNT ANALYSIS\n")
        f.write("-" * 20 + "\n")
        f.write("Distribution of molecules by ring count:\n")
        for count, num_mols in sorted(ring_count_dist.items()):
            percentage = (num_mols / total_molecules) * 100
            f.write(f"  {count} rings: {num_mols:,} molecules ({percentage:.4f}%)\n")
        
        f.write("\nRING LENGTH ANALYSIS\n")
        f.write("-" * 20 + "\n")
        f.write("Distribution of molecules by maximum ring length:\n")
        for length, num_molecules in sorted(ring_length_dist.items()):
            percentage = (num_molecules / total_molecules) * 100
            if length == 0:
                f.write(f"  0-atom rings (acyclic): {num_molecules:,} molecules ({percentage:.4f}%)\n")
            else:
                f.write(f"  {length}-atom rings: {num_molecules:,} molecules ({percentage:.4f}%)\n")
        
        f.write("\nCONSTRAINT SATISFACTION RATES\n")
        f.write("-" * 30 + "\n")
        f.write("Ring Count Constraints (cumulative per molecule):\n")
        for constraint, (rate, count) in ring_count_rates.items():
            f.write(f"  {constraint}: {rate*100:.4f}%\n")
        
        f.write("\nRing Length Constraints (cumulative per molecule by max ring length):\n")
        for constraint, (rate, count) in ring_length_rates.items():
            f.write(f"  {constraint}: {rate*100:.4f}%\n")
        
        if planar_count is not None:
            f.write(f"\nPLANARITY ANALYSIS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Planar molecules: {planar_count:,}\n")
            f.write(f"Non-planar molecules: {non_planar_count:,}\n")
            f.write(f"Planarity rate: {planar_count/(planar_count+non_planar_count):.3f} ({(planar_count/(planar_count+non_planar_count))*100:.4f}%)\n")
        
        f.write(f"\n\nAnalysis completed on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"✅ Saved {analysis_name} results to {output_dir}/")

def create_comparison_analysis(train_results, val_results, test_results, full_results, output_dir):
    """Create comparison analysis between all dataset splits"""
    print("\n📊 Creating comparison analysis...")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create comparison plots
    plt.figure(figsize=(20, 15))
    
    # Ring count comparison
    plt.subplot(3, 3, 1)
    ring_count_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    train_rates = [train_results['ring_count_rates'][f"≤{i}"][0] for i in ring_count_values]
    val_rates = [val_results['ring_count_rates'][f"≤{i}"][0] for i in ring_count_values]
    test_rates = [test_results['ring_count_rates'][f"≤{i}"][0] for i in ring_count_values]
    full_rates = [full_results['ring_count_rates'][f"≤{i}"][0] for i in ring_count_values]
    x = range(len(ring_count_values))
    
    plt.plot(x, train_rates, 'o-', label='Training', linewidth=2, markersize=8)
    plt.plot(x, val_rates, 's-', label='Validation', linewidth=2, markersize=8)
    plt.plot(x, test_rates, '^-', label='Test', linewidth=2, markersize=8)
    plt.plot(x, full_rates, 'd-', label='Full Dataset', linewidth=2, markersize=8)
    plt.title('Ring Count Constraint Satisfaction Comparison', fontweight='bold')
    plt.xlabel('Maximum Rings')
    plt.ylabel('Satisfaction Rate')
    plt.xticks(x, ring_count_values)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Ring length comparison
    plt.subplot(3, 3, 2)
    ring_length_values = [0, 3, 4, 5, 6, 7, 8, 9]
    train_length_rates = [train_results['ring_length_rates'][f"≤{i}"][0] for i in ring_length_values]
    val_length_rates = [val_results['ring_length_rates'][f"≤{i}"][0] for i in ring_length_values]
    test_length_rates = [test_results['ring_length_rates'][f"≤{i}"][0] for i in ring_length_values]
    full_length_rates = [full_results['ring_length_rates'][f"≤{i}"][0] for i in ring_length_values]
    x = range(len(ring_length_values))
    
    plt.plot(x, train_length_rates, 'o-', label='Training', linewidth=2, markersize=8)
    plt.plot(x, val_length_rates, 's-', label='Validation', linewidth=2, markersize=8)
    plt.plot(x, test_length_rates, '^-', label='Test', linewidth=2, markersize=8)
    plt.plot(x, full_length_rates, 'd-', label='Full Dataset', linewidth=2, markersize=8)
    plt.title('Ring Length Constraint Satisfaction Comparison', fontweight='bold')
    plt.xlabel('Maximum Ring Length')
    plt.ylabel('Satisfaction Rate')
    plt.xticks(x, ring_length_values)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Dataset size comparison
    plt.subplot(3, 3, 3)
    sizes = [train_results['dataset_info']['total_molecules'], 
             val_results['dataset_info']['total_molecules'],
             test_results['dataset_info']['total_molecules'],
             full_results['dataset_info']['total_molecules']]
    labels = ['Training', 'Validation', 'Test', 'Full Dataset']
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'gold']
    
    bars = plt.bar(labels, sizes, color=colors, alpha=0.7)
    plt.title('Dataset Size Comparison', fontweight='bold')
    plt.ylabel('Number of Molecules')
    plt.xticks(rotation=45)
    
    # Add value labels
    for bar, size in zip(bars, sizes):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sizes)*0.01,
                f'{size:,}', ha='center', va='bottom', fontweight='bold')
    
    # Average rings comparison
    plt.subplot(3, 3, 4)
    avg_rings = [train_results['dataset_info']['avg_rings_per_molecule'], 
                 val_results['dataset_info']['avg_rings_per_molecule'],
                 test_results['dataset_info']['avg_rings_per_molecule'],
                 full_results['dataset_info']['avg_rings_per_molecule']]
    
    bars = plt.bar(labels, avg_rings, color=colors, alpha=0.7)
    plt.title('Average Rings per Molecule Comparison', fontweight='bold')
    plt.ylabel('Average Rings')
    plt.xticks(rotation=45)
    
    # Add value labels
    for bar, avg in zip(bars, avg_rings):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(avg_rings)*0.01,
                f'{avg:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Ring count distribution comparison
    plt.subplot(3, 3, 5)
    max_rings = max(max(int(k) for k in train_results['ring_count_distribution'].keys()),
                   max(int(k) for k in val_results['ring_count_distribution'].keys()),
                   max(int(k) for k in test_results['ring_count_distribution'].keys()),
                   max(int(k) for k in full_results['ring_count_distribution'].keys()))
    
    x = range(max_rings + 1)
    train_dist = [train_results['ring_count_distribution'].get(str(i), 0) for i in x]
    val_dist = [val_results['ring_count_distribution'].get(str(i), 0) for i in x]
    test_dist = [test_results['ring_count_distribution'].get(str(i), 0) for i in x]
    full_dist = [full_results['ring_count_distribution'].get(str(i), 0) for i in x]
    
    # Normalize to percentages
    train_sum = sum(train_dist)
    val_sum = sum(val_dist)
    test_sum = sum(test_dist)
    full_sum = sum(full_dist)
    
    train_dist = [v/train_sum*100 if train_sum > 0 else 0 for v in train_dist]
    val_dist = [v/val_sum*100 if val_sum > 0 else 0 for v in val_dist]
    test_dist = [v/test_sum*100 if test_sum > 0 else 0 for v in test_dist]
    full_dist = [v/full_sum*100 if full_sum > 0 else 0 for v in full_dist]
    
    plt.plot(x, train_dist, 'o-', label='Training', linewidth=2, markersize=6)
    plt.plot(x, val_dist, 's-', label='Validation', linewidth=2, markersize=6)
    plt.plot(x, test_dist, '^-', label='Test', linewidth=2, markersize=6)
    plt.plot(x, full_dist, 'd-', label='Full Dataset', linewidth=2, markersize=6)
    plt.title('Ring Count Distribution Comparison (%)', fontweight='bold')
    plt.xlabel('Number of Rings')
    plt.ylabel('Percentage of Molecules')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Ring length distribution comparison
    plt.subplot(3, 3, 6)
    max_length = max(max(int(k) for k in train_results['ring_length_distribution'].keys()),
                    max(int(k) for k in val_results['ring_length_distribution'].keys()),
                    max(int(k) for k in test_results['ring_length_distribution'].keys()),
                    max(int(k) for k in full_results['ring_length_distribution'].keys()))
    
    x = range(0, max_length + 1)  # Include 0 for acyclic molecules
    train_length_dist = [train_results['ring_length_distribution'].get(str(i), 0) for i in x]
    val_length_dist = [val_results['ring_length_distribution'].get(str(i), 0) for i in x]
    test_length_dist = [test_results['ring_length_distribution'].get(str(i), 0) for i in x]
    full_length_dist = [full_results['ring_length_distribution'].get(str(i), 0) for i in x]
    
    # Normalize to percentages
    train_length_sum = sum(train_length_dist)
    val_length_sum = sum(val_length_dist)
    test_length_sum = sum(test_length_dist)
    full_length_sum = sum(full_length_dist)
    
    train_length_dist = [v/train_length_sum*100 if train_length_sum > 0 else 0 for v in train_length_dist]
    val_length_dist = [v/val_length_sum*100 if val_length_sum > 0 else 0 for v in val_length_dist]
    test_length_dist = [v/test_length_sum*100 if test_length_sum > 0 else 0 for v in test_length_dist]
    full_length_dist = [v/full_length_sum*100 if full_length_sum > 0 else 0 for v in full_length_dist]
    
    plt.plot(x, train_length_dist, 'o-', label='Training', linewidth=2, markersize=6)
    plt.plot(x, val_length_dist, 's-', label='Validation', linewidth=2, markersize=6)
    plt.plot(x, test_length_dist, '^-', label='Test', linewidth=2, markersize=6)
    plt.plot(x, full_length_dist, 'd-', label='Full Dataset', linewidth=2, markersize=6)
    plt.title('Ring Length Distribution Comparison (%)', fontweight='bold')
    plt.xlabel('Ring Length (atoms)')
    plt.ylabel('Percentage of Rings')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Split proportions
    plt.subplot(3, 3, 7)
    total_mols = full_results['dataset_info']['total_molecules']
    split_sizes = [train_results['dataset_info']['total_molecules'],
                   val_results['dataset_info']['total_molecules'],
                   test_results['dataset_info']['total_molecules']]
    split_labels = ['Training', 'Validation', 'Test']
    split_colors = ['lightblue', 'lightgreen', 'lightcoral']
    
    wedges, texts, autotexts = plt.pie(split_sizes, labels=split_labels, colors=split_colors,
                                       autopct='%1.1f%%', startangle=90)
    plt.title('Dataset Split Proportions', fontweight='bold')
    
    # Make text more readable
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    # Constraint satisfaction heatmap
    plt.subplot(3, 3, 8)
    constraints = ['≤0', '≤1', '≤2', '≤3', '≤4', '≤5', '≤6', '≤7', '≤8', '≤9', '>9']
    datasets = ['Training', 'Validation', 'Test', 'Full']
    
    # Ring count rates
    rates_matrix = []
    for dataset_results in [train_results, val_results, test_results, full_results]:
        rates = [dataset_results['ring_count_rates'][c][0] for c in constraints]
        rates_matrix.append(rates)
    
    im = plt.imshow(rates_matrix, cmap='YlOrRd', aspect='auto')
    plt.colorbar(im, label='Satisfaction Rate')
    plt.xticks(range(len(constraints)), constraints)
    plt.yticks(range(len(datasets)), datasets)
    plt.title('Ring Count Constraint Satisfaction Heatmap', fontweight='bold')
    plt.xlabel('Ring Count Constraint')
    plt.ylabel('Dataset')
    
    # Add text annotations
    for i in range(len(datasets)):
        for j in range(len(constraints)):
            plt.text(j, i, f'{rates_matrix[i][j]:.2f}', ha='center', va='center', 
                    color='black', fontweight='bold')
    
    # Ring length constraint heatmap
    plt.subplot(3, 3, 9)
    length_constraints = ['≤0', '≤3', '≤4', '≤5', '≤6', '≤7', '≤8', '≤9', '>9']
    
    # Ring length rates
    length_rates_matrix = []
    for dataset_results in [train_results, val_results, test_results, full_results]:
        rates = [dataset_results['ring_length_rates'][c][0] for c in length_constraints]
        length_rates_matrix.append(rates)
    
    im = plt.imshow(length_rates_matrix, cmap='Blues', aspect='auto')
    plt.colorbar(im, label='Satisfaction Rate')
    plt.xticks(range(len(length_constraints)), length_constraints)
    plt.yticks(range(len(datasets)), datasets)
    plt.title('Ring Length Constraint Satisfaction Heatmap', fontweight='bold')
    plt.xlabel('Ring Length Constraint')
    plt.ylabel('Dataset')
    
    # Add text annotations
    for i in range(len(datasets)):
        for j in range(len(length_constraints)):
            plt.text(j, i, f'{length_rates_matrix[i][j]:.2f}', ha='center', va='center', 
                    color='black', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "comprehensive_comparison_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save comprehensive comparison report
    with open(output_dir / "comprehensive_comparison_report.txt", 'w') as f:
        f.write("QM9 Comprehensive Dataset Split Comparison Report\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("DATASET SIZE COMPARISON\n")
        f.write("-" * 30 + "\n")
        f.write(f"Training dataset: {train_results['dataset_info']['total_molecules']:,} molecules\n")
        f.write(f"Validation dataset: {val_results['dataset_info']['total_molecules']:,} molecules\n")
        f.write(f"Test dataset: {test_results['dataset_info']['total_molecules']:,} molecules\n")
        f.write(f"Full dataset: {full_results['dataset_info']['total_molecules']:,} molecules\n")
        f.write(f"Total (sum of splits): {train_results['dataset_info']['total_molecules'] + val_results['dataset_info']['total_molecules'] + test_results['dataset_info']['total_molecules']:,} molecules\n\n")
        
        f.write("DATASET PROPORTIONS\n")
        f.write("-" * 20 + "\n")
        total = full_results['dataset_info']['total_molecules']
        f.write(f"Training: {train_results['dataset_info']['total_molecules']/total:.1%}\n")
        f.write(f"Validation: {val_results['dataset_info']['total_molecules']/total:.1%}\n")
        f.write(f"Test: {test_results['dataset_info']['total_molecules']/total:.1%}\n\n")
        
        f.write("RING COUNT COMPARISON\n")
        f.write("-" * 25 + "\n")
        for i in range(10):
            train_rate = train_results['ring_count_rates'][f"≤{i}"][0]
            val_rate = val_results['ring_count_rates'][f"≤{i}"][0]
            test_rate = test_results['ring_count_rates'][f"≤{i}"][0]
            full_rate = full_results['ring_count_rates'][f"≤{i}"][0]
            f.write(f"≤{i} rings: Train={train_rate:.3f}, Val={val_rate:.3f}, Test={test_rate:.3f}, Full={full_rate:.3f}\n")
        
        # Add >9 category
        train_rate = train_results['ring_count_rates'][">9"][0]
        val_rate = val_results['ring_count_rates'][">9"][0]
        test_rate = test_results['ring_count_rates'][">9"][0]
        full_rate = full_results['ring_count_rates'][">9"][0]
        f.write(f">9 rings: Train={train_rate:.3f}, Val={val_rate:.3f}, Test={test_rate:.3f}, Full={full_rate:.3f}\n")
        
        f.write("\nRING LENGTH COMPARISON\n")
        f.write("-" * 27 + "\n")
        for i in [0, 3, 4, 5, 6, 7, 8, 9]:
            train_rate = train_results['ring_length_rates'][f"≤{i}"][0]
            val_rate = val_results['ring_length_rates'][f"≤{i}"][0]
            test_rate = test_results['ring_length_rates'][f"≤{i}"][0]
            full_rate = full_results['ring_length_rates'][f"≤{i}"][0]
            f.write(f"≤{i} atoms: Train={train_rate:.3f}, Val={val_rate:.3f}, Test={test_rate:.3f}, Full={full_rate:.3f}\n")
        
        # Add >9 category for ring length
        train_rate = train_results['ring_length_rates'][">9"][0]
        val_rate = val_results['ring_length_rates'][">9"][0]
        test_rate = test_results['ring_length_rates'][">9"][0]
        full_rate = full_results['ring_length_rates'][">9"][0]
        f.write(f">9 atoms: Train={train_rate:.3f}, Val={val_rate:.3f}, Test={test_rate:.3f}, Full={full_rate:.3f}\n")
        
        f.write(f"\n\nComprehensive comparison completed on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"✅ Saved comprehensive comparison analysis to {output_dir}/")

def main():
    """Main analysis function"""
    print("🔬 Complete QM9 Dataset Analysis")
    print("=" * 50)
    
    start_time = time.time()
    
    # Create output directory
    output_dir = create_output_dir()
    print(f"📁 Output directory: {output_dir}")
    
    # ===== 1. TRAINING DATASET ANALYSIS =====
    print("\n" + "="*50)
    print("📚 TRAINING DATASET ANALYSIS")
    print("="*50)
    
    train_dataset = load_qm9_dataset("train")
    if train_dataset is None:
        print("❌ Failed to load training dataset. Exiting.")
        return
    
    # Analyze training dataset
    train_ring_count_dist, train_ring_length_dist, train_max_ring_length_per_mol, train_total_rings = analyze_ring_properties(train_dataset, "Training Dataset")
    train_planar_count, train_non_planar_count = analyze_planarity(train_dataset, "Training Dataset")
    train_ring_count_rates, train_ring_length_rates = calculate_constraint_rates(train_ring_count_dist, train_ring_length_dist, train_max_ring_length_per_mol)
    
    # Save training analysis
    train_results = {
        'dataset_info': {
            'total_molecules': len(train_dataset),
            'total_rings': train_total_rings,
            'avg_rings_per_molecule': train_total_rings / len(train_dataset)
        },
        'ring_count_distribution': dict(train_ring_count_dist),
        'ring_length_distribution': dict(train_ring_length_dist),
        'ring_count_rates': train_ring_count_rates,
        'ring_length_rates': train_ring_length_rates
    }
    
    save_results(train_ring_count_dist, train_ring_length_dist, train_ring_count_rates, train_ring_length_rates,
                train_planar_count, train_non_planar_count, train_total_rings, len(train_dataset), 
                output_dir / "train_analysis", "training")
    
    create_plots(train_ring_count_dist, train_ring_length_dist, train_planar_count, train_non_planar_count,
                train_ring_count_rates, train_ring_length_rates, output_dir / "train_analysis", "Training")
    
    # ===== 2. VALIDATION DATASET ANALYSIS =====
    print("\n" + "="*50)
    print("🔍 VALIDATION DATASET ANALYSIS")
    print("="*50)
    
    val_dataset = load_qm9_dataset("val")
    if val_dataset is None:
        print("❌ Failed to load validation dataset. Exiting.")
        return
    
    # Analyze validation dataset
    val_ring_count_dist, val_ring_length_dist, val_max_ring_length_per_mol, val_total_rings = analyze_ring_properties(val_dataset, "Validation Dataset")
    val_planar_count, val_non_planar_count = analyze_planarity(val_dataset, "Validation Dataset")
    val_ring_count_rates, val_ring_length_rates = calculate_constraint_rates(val_ring_count_dist, val_ring_length_dist, val_max_ring_length_per_mol)
    
    # Save validation analysis
    val_results = {
        'dataset_info': {
            'total_molecules': len(val_dataset),
            'total_rings': val_total_rings,
            'avg_rings_per_molecule': val_total_rings / len(val_dataset)
        },
        'ring_count_distribution': dict(val_ring_count_dist),
        'ring_length_distribution': dict(val_ring_length_dist),
        'ring_count_rates': val_ring_count_rates,
        'ring_length_rates': val_ring_length_rates
    }
    
    save_results(val_ring_count_dist, val_ring_length_dist, val_ring_count_rates, val_ring_length_rates,
                val_planar_count, val_non_planar_count, val_total_rings, len(val_dataset), 
                output_dir / "val_analysis", "validation")
    
    create_plots(val_ring_count_dist, val_ring_length_dist, val_planar_count, val_non_planar_count,
                val_ring_count_rates, val_ring_length_rates, output_dir / "val_analysis", "Validation")
    
    # ===== 3. TEST DATASET ANALYSIS =====
    print("\n" + "="*50)
    print("🧪 TEST DATASET ANALYSIS")
    print("="*50)
    
    test_dataset = load_qm9_dataset("test")
    if test_dataset is None:
        print("❌ Failed to load test dataset. Exiting.")
        return
    
    # Analyze test dataset
    test_ring_count_dist, test_ring_length_dist, test_max_ring_length_per_mol, test_total_rings = analyze_ring_properties(test_dataset, "Test Dataset")
    test_planar_count, test_non_planar_count = analyze_planarity(test_dataset, "Test Dataset")
    test_ring_count_rates, test_ring_length_rates = calculate_constraint_rates(test_ring_count_dist, test_ring_length_dist, test_max_ring_length_per_mol)
    
    # Save test analysis
    test_results = {
        'dataset_info': {
            'total_molecules': len(test_dataset),
            'total_rings': test_total_rings,
            'avg_rings_per_molecule': test_total_rings / len(test_dataset)
        },
        'ring_count_distribution': dict(test_ring_count_dist),
        'ring_length_distribution': dict(test_ring_length_dist),
        'ring_count_rates': test_ring_count_rates,
        'ring_length_rates': test_ring_length_rates
    }
    
    save_results(test_ring_count_dist, test_ring_length_dist, test_ring_count_rates, test_ring_length_rates,
                test_planar_count, test_non_planar_count, test_total_rings, len(test_dataset), 
                output_dir / "test_analysis", "test")
    
    create_plots(test_ring_count_dist, test_ring_length_dist, test_planar_count, test_non_planar_count,
                test_ring_count_rates, test_ring_length_rates, output_dir / "test_analysis", "Test")
    
    # ===== 4. FULL DATASET ANALYSIS =====
    print("\n" + "="*50)
    print("🌐 FULL DATASET ANALYSIS")
    print("="*50)
    
    full_datasets = load_full_qm9_datasets()
    if full_datasets is None:
        print("❌ Failed to load full datasets. Exiting.")
        return
    
    # Combine all splits for full dataset analysis
    full_dataset = []
    for split_name, dataset in full_datasets.items():
        full_dataset.extend(dataset)
    
    print(f"📊 Combined dataset: {len(full_dataset)} molecules")
    
    # Analyze full dataset
    full_ring_count_dist, full_ring_length_dist, full_max_ring_length_per_mol, full_total_rings = analyze_ring_properties(full_dataset, "Full Dataset")
    full_planar_count, full_non_planar_count = analyze_planarity(full_dataset, "Full Dataset")
    full_ring_count_rates, full_ring_length_rates = calculate_constraint_rates(full_ring_count_dist, full_ring_length_dist, full_max_ring_length_per_mol)
    
    # Save full dataset analysis
    full_results = {
        'dataset_info': {
            'total_molecules': len(full_dataset),
            'total_rings': full_total_rings,
            'avg_rings_per_molecule': full_total_rings / len(full_dataset)
        },
        'ring_count_distribution': dict(full_ring_count_dist),
        'ring_length_distribution': dict(full_ring_length_dist),
        'ring_count_rates': full_ring_count_rates,
        'ring_length_rates': full_ring_length_rates
    }
    
    save_results(full_ring_count_dist, full_ring_length_dist, full_ring_count_rates, full_ring_length_rates,
                full_planar_count, full_non_planar_count, full_total_rings, len(full_dataset), 
                output_dir / "full_analysis", "full")
    
    create_plots(full_ring_count_dist, full_ring_length_dist, full_planar_count, full_non_planar_count,
                full_ring_count_rates, full_ring_length_rates, output_dir / "full_analysis", "Full")
    
    # ===== 5. COMPREHENSIVE COMPARISON ANALYSIS =====
    print("\n" + "="*50)
    print("📊 COMPREHENSIVE COMPARISON ANALYSIS")
    print("="*50)
    
    create_comparison_analysis(train_results, val_results, test_results, full_results, output_dir / "comparison")
    
    # ===== 6. SUMMARY =====
    print("\n" + "="*50)
    print("📋 ANALYSIS SUMMARY")
    print("="*50)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"✅ Training dataset analysis: {len(train_dataset):,} molecules")
    print(f"✅ Validation dataset analysis: {len(val_dataset):,} molecules")
    print(f"✅ Test dataset analysis: {len(test_dataset):,} molecules")
    print(f"✅ Full dataset analysis: {len(full_dataset):,} molecules")
    print(f"✅ Comprehensive comparison analysis completed")
    
    print(f"\n📁 Results saved to: {output_dir}/")
    print("   ├── train_analysis/ (training dataset analysis)")
    print("   ├── val_analysis/ (validation dataset analysis)")
    print("   ├── test_analysis/ (test dataset analysis)")
    print("   ├── full_analysis/ (full dataset analysis)")
    print("   ├── comparison/ (comprehensive comparison analysis)")
    print("   ├── plots/ (visualization charts)")
    print("   ├── reports/ (detailed analysis reports)")
    print("   └── data/ (JSON data files)")
    print(f"⏱️  Analysis completed in {duration:.1f} seconds")
    print("✅ Complete analysis finished!")

if __name__ == "__main__":
    main() 