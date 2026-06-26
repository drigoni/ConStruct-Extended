#!/usr/bin/env python3
"""
Identify Best Checkpoint from Existing Structure

This script helps you find the best checkpoint from the current confusing folder structure.
Run this to see which checkpoint actually has the lowest NLL value.
"""

import os
import re
from pathlib import Path

def find_best_checkpoint(experiment_name):
    """Find the best checkpoint for a given experiment."""
    checkpoint_dir = f"ConStruct/checkpoints/{experiment_name}"
    
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Experiment directory not found: {checkpoint_dir}")
        return None
    
    print(f"🔍 Analyzing checkpoints for: {experiment_name}")
    print("=" * 60)
    
    # Look for checkpoints in the best folder
    best_dir = os.path.join(checkpoint_dir, "best")
    checkpoints = []
    
    if os.path.exists(best_dir):
        for epoch_dir in os.listdir(best_dir):
            epoch_path = os.path.join(best_dir, epoch_dir)
            if os.path.isdir(epoch_path):
                for ckpt_file in os.listdir(epoch_path):
                    if ckpt_file.endswith('.ckpt'):
                        ckpt_path = os.path.join(epoch_path, ckpt_file)
                        
                        # Parse epoch and NLL from filename
                        epoch_match = re.search(r'epoch=(\d+)', epoch_dir)
                        nll_match = re.search(r'epoch_NLL=([\d.]+)', ckpt_file)
                        
                        if epoch_match and nll_match:
                            epoch = int(epoch_match.group(1))
                            # Remove trailing dot if present
                            nll_str = nll_match.group(1).rstrip('.')
                            nll = float(nll_str)
                            checkpoints.append((ckpt_path, epoch, nll))
    
    if not checkpoints:
        print("❌ No checkpoints found in best folder")
        return None
    
    # Sort by NLL (lower is better)
    checkpoints.sort(key=lambda x: x[2])
    
    print("📊 Checkpoint Ranking (by NLL, lower is better):")
    print("-" * 80)
    for i, (ckpt_path, epoch, nll) in enumerate(checkpoints):
        status = "🥇 BEST" if i == 0 else f"#{i+1}"
        print(f"{status:8} | Epoch {epoch:3d} | NLL: {nll:8.4f} | {os.path.basename(ckpt_path)}")
    print("-" * 80)
    
    # Show the best one
    best_ckpt, best_epoch, best_nll = checkpoints[0]
    print(f"\n✅ Best checkpoint: {best_ckpt}")
    print(f"📈 Best NLL: {best_nll:.4f} at epoch {best_epoch}")
    
    return best_ckpt

def main():
    print("🧪 Checkpoint Identification Tool")
    print("=" * 50)
    
    # List available experiments
    checkpoint_base = "ConStruct/checkpoints"
    if not os.path.exists(checkpoint_base):
        print(f"❌ Checkpoint directory not found: {checkpoint_base}")
        return
    
    experiments = [d for d in os.listdir(checkpoint_base) 
                  if os.path.isdir(os.path.join(checkpoint_base, d))]
    
    print("📁 Available experiments:")
    for i, exp in enumerate(experiments, 1):
        print(f"  {i}. {exp}")
    
    print(f"\n🔍 Enter experiment number (1-{len(experiments)}) or experiment name:")
    choice = input("> ").strip()
    
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(experiments):
            experiment_name = experiments[idx]
        else:
            print("❌ Invalid number")
            return
    else:
        experiment_name = choice
    
    # Find best checkpoint
    best_ckpt = find_best_checkpoint(experiment_name)
    
    if best_ckpt:
        print(f"\n🎯 For testing, use this checkpoint:")
        print(f"   {best_ckpt}")
        
        # Also show the relative path for easy copying
        rel_path = os.path.relpath(best_ckpt, ".")
        print(f"\n📋 Copy-paste friendly path:")
        print(f"   {rel_path}")

if __name__ == "__main__":
    main() 