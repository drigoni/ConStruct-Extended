import os
from collections import Counter
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from torchmetrics import MeanMetric, MeanAbsoluteError, Metric, CatMetric
import torch
import wandb
import torch.nn as nn

from ConStruct.utils import PlaceHolder
import fcd
import numpy as np
import signal
import time
from tabulate import tabulate
from rdkit.Chem import AllChem
import networkx as nx
import pandas as pd

from ConStruct.projector.projector_utils import build_simple_graph_from_edge_tensor
from ConStruct.projector.graph_cycles import enumerate_simple_cycles_unique, count_simple_cycles, max_simple_cycle_length
from ConStruct.projector.is_planar import is_planar

allowed_bonds = {
    "H": {0: 1, 1: 0, -1: 0},
    "C": {0: [3, 4], 1: 3, -1: 3},
    "N": {
        0: [2, 3],
        1: [2, 3, 4],
        -1: 2,
    },  # In QM9, N+ seems to be present in the form NH+ and NH2+
    "O": {0: 2, 1: 3, -1: 1},
    "F": {0: 1, -1: 0},
    "B": 3,
    "Al": 3,
    "Si": 4,
    "P": {0: [3, 5], 1: 4},
    "S": {0: [2, 6], 1: [2, 3], 2: 4, 3: 5, -1: 3},
    "Cl": 1,
    "As": 3,
    "Br": {0: 1, 1: 2},
    "I": 1,
    "Hg": [1, 2],
    "Bi": [3, 5],
    "Se": [2, 4, 6],
}
bond_dict = [
    None,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}

RDLogger.DisableLog("rdApp.*")


class Molecule:
    def __init__(self, graph: PlaceHolder, atom_decoder):
        """atom_decoder: extracted from dataset_infos."""
        self.atom_types = graph.X.long()
        self.bond_types = graph.E.long()
        self.charges = graph.charges.long()
        self.rdkit_mol = self.build_molecule(atom_decoder)
        self.atom_decoder = atom_decoder
        self.num_nodes = len(graph.X)
        self.num_atom_types = len(atom_decoder)
        self.device = self.atom_types.device

    def build_molecule(self, atom_decoder):
        mol = Chem.RWMol()
        for atom, charge in zip(self.atom_types, self.charges):
            if atom == -1:
                continue
            a = Chem.Atom(atom_decoder[int(atom.item())])
            a.SetFormalCharge(charge.item())
            mol.AddAtom(a)
        edge_types = torch.triu(self.bond_types, diagonal=1)
        edge_types[edge_types == -1] = 0
        all_bonds = torch.nonzero(edge_types)
        for i, bond in enumerate(all_bonds):
            if bond[0].item() != bond[1].item():
                mol.AddBond(
                    bond[0].item(),
                    bond[1].item(),
                    bond_dict[edge_types[bond[0], bond[1]].item()],
                )
        try:
            mol = mol.GetMol()
        except Chem.KekulizeException:
            print("Can't kekulize molecule")
            return None
        return mol

    def check_stability(self, debug=False):
        e = self.bond_types.clone()
        e[e == 4] = 1.5
        e[e < 0] = 0
        valencies = torch.sum(e, dim=-1).long()

        n_stable_at = 0
        mol_stable = True
        for i, (atom_type, valency, charge) in enumerate(
            zip(self.atom_types, valencies, self.charges)
        ):
            atom_type = atom_type.item()
            valency = valency.item()
            charge = charge.item()
            possible_bonds = allowed_bonds[self.atom_decoder[atom_type]]
            if type(possible_bonds) == int:
                is_stable = possible_bonds == valency
            elif type(possible_bonds) == dict:
                expected_bonds = (
                    possible_bonds[charge]
                    if charge in possible_bonds.keys()
                    else possible_bonds[0]
                )
                is_stable = (
                    expected_bonds == valency
                    if type(expected_bonds) == int
                    else valency in expected_bonds
                )
            else:
                is_stable = valency in possible_bonds
            if not is_stable:
                mol_stable = False
            if not is_stable and debug:
                print(
                    f"Invalid atom {self.atom_decoder[atom_type]}: valency={valency}, charge={charge}"
                )
                print()
            n_stable_at += int(is_stable)

        return mol_stable, n_stable_at, len(self.atom_types)


class SamplingMolecularMetrics(nn.Module):
    def __init__(self, dataset_infos, test, cfg=None):
        super().__init__()
        self.dataset_infos = dataset_infos
        self.is_molecular = dataset_infos.is_molecular
        self.atom_decoder = dataset_infos.atom_decoder
        self.cfg = cfg

        self.test = test
        self.cfg = cfg
        self.is_molecular = dataset_infos.is_molecular
        self.remove_h = dataset_infos.remove_h
        # FIX: Canonicalize training SMILES to match generated SMILES format
        if dataset_infos.train_smiles is not None:
            # Canonicalize training SMILES to match the format used in compute_validity
            canonical_train_smiles = set()
            for smiles in dataset_infos.train_smiles:
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                        canonical_train_smiles.add(canonical_smiles)
                except:
                    # Skip invalid SMILES
                    continue
            self.train_smiles = canonical_train_smiles

        else:
            self.train_smiles = set()
        
        self.test_smiles = (
            dataset_infos.test_smiles if test else dataset_infos.val_smiles
        )
        self.test_fcd_stats = (
            dataset_infos.test_fcd_stats if test else dataset_infos.val_fcd_stats
        )
        
        # Timing tracking
        self.total_sampling_time = 0.0
        self.avg_sampling_time = 0.0
        self.num_batches = 0
        
        # Projection timing tracking
        self.total_projection_time = 0.0
        self.projection_times = []  # List to store individual projection times
        self.projection_calls = 0
        
        # Wall clock timing
        import time
        self.start_time = time.time()

        self.atom_stable = MeanMetric()
        self.mol_stable = MeanMetric()


        
        # Retrieve dataset smiles only for qm9 currently.
        # self.train_smiles = set(dataset_infos.train_smiles)  # REMOVED DUPLICATE
        self.validity_metric = MeanMetric()
        # self.uniqueness_metric = UniquenessMetric()
        self.novelty_metric = MeanMetric(nan_strategy="ignore")

        self.charge_w1 = MeanMetric()
        self.valency_w1 = MeanMetric()

        # for fcd
        self.val_smiles = list(
            dataset_infos.test_smiles if test else dataset_infos.val_smiles
        )
        self.val_fcd_mu, self.val_fcd_sigma = (
            dataset_infos.test_fcd_stats if test else dataset_infos.val_fcd_stats
        )

    def reset(self):
        for metric in [
            self.atom_stable,
            self.mol_stable,
            self.validity_metric,  # self.uniqueness_metric,
            self.novelty_metric,
            self.atom_stable,
            self.mol_stable,
            self.charge_w1,
            self.valency_w1,
        ]:
            metric.reset()

    def compute_validity(self, generated):
        """generated: list of couples (positions, atom_types)"""
        valid = []
        all_smiles = []
        error_message = Counter()
        for mol in generated:
            rdmol = mol.rdkit_mol
            if rdmol is not None:
                try:
                    mol_frags = Chem.rdmolops.GetMolFrags(
                        rdmol, asMols=True, sanitizeFrags=True
                    )
                    
                    # below code handles disconnected molecules by selecting the largest valid molecular fragment.
                    if len(mol_frags) > 1:
                        error_message[4] += 1  # count the number of disconnected molecules
                    largest_mol = max(
                        mol_frags, default=mol, key=lambda m: m.GetNumAtoms()
                    )
                    Chem.SanitizeMol(largest_mol)
                    smiles = Chem.MolToSmiles(largest_mol, canonical=True)
                    valid.append(smiles)
                    all_smiles.append(smiles)
                    error_message[-1] += 1
                except Chem.rdchem.AtomValenceException:
                    error_message[1] += 1
                    all_smiles.append("error")
                except Chem.rdchem.KekulizeException:
                    error_message[2] += 1
                    all_smiles.append("error")
                except Chem.rdchem.AtomKekulizeException or ValueError:
                    error_message[3] += 1
                    all_smiles.append("error")
        print(
            f"Error messages: AtomValence {error_message[1]}, Kekulize {error_message[2]}, other {error_message[3]}, "
            f" -- No error {error_message[-1]}"
        )
        self.validity_metric.update(
            value=len(valid) / len(generated), weight=len(generated)
        )
        return valid, all_smiles, error_message

    def evaluate(self, generated):
        """generated: list of pairs (positions: n x 3, atom_types: n [int])
        the positions and atom types should already be masked."""

        valid_mols, all_smiles, error_message = self.compute_validity(generated)
        validity = self.validity_metric.compute().item()

        # self.uniqueness_metric.update(valid_mols)
        if len(valid_mols) > 0:
            uniqueness = len(set(valid_mols)) / len(valid_mols)
        else:
            uniqueness = 0.0
        unique = list(set(valid_mols))

        if self.train_smiles is not None and len(unique) > 0:
            novel = [s for s in unique if s not in self.train_smiles]
            self.novelty_metric.update(
                value=len(novel) / len(unique), weight=len(unique)
            )
            novelty = self.novelty_metric.compute().item()
        else:
            print("No valid molecules")
            novelty = 0.0
        # num_molecules = int(self.validity_metric.weight.item())
        # print(f"Validity over {num_molecules} molecules: {validity * 100 :.2f}%")

        key = "val_sampling" if not self.test else "test_sampling"
        dic = {
            f"{key}/Validity": validity * 100,
            f"{key}/Uniqueness": uniqueness * 100 if uniqueness != 0 else 0,
            f"{key}/Novelty": novelty * 100 if novelty != 0 else 0,
        }

        return all_smiles, dic

    def forward(self, generated_graphs: list[PlaceHolder], current_epoch, local_rank, computed_metrics=None):
        """Compute sampling metrics and track constraint violations."""
        molecules = []

        for batch in generated_graphs:
            graphs = batch.split()
            for graph in graphs:
                molecule = Molecule(graph, atom_decoder=self.dataset_infos.atom_decoder)
                molecules.append(molecule)

        if not self.dataset_infos.remove_h:
            print(f"Analyzing molecule stability on {local_rank}...")
            for i, mol in enumerate(molecules):
                mol_stable, at_stable, num_bonds = mol.check_stability()
                self.mol_stable.update(value=mol_stable)
                self.atom_stable.update(value=at_stable / num_bonds, weight=num_bonds)

            stability_dict = {
                "mol_stable": self.mol_stable.compute().item(),
                "atm_stable": self.atom_stable.compute().item(),
            }
            if local_rank == 0:
                print("Stability metrics:", stability_dict)
                if wandb.run:
                    wandb.log(stability_dict, commit=False)

        # Validity, uniqueness, novelty
        all_generated_smiles, metrics = self.evaluate(molecules)
        
        # No chemical validation metrics in initial commit - removed to match exactly

        # Calculate FCD BEFORE table generation
        to_log_fcd = self.compute_fcd(generated_smiles=all_generated_smiles)
        metrics.update(to_log_fcd)

        if len(all_generated_smiles) > 0 and local_rank == 0:
            print("Some generated smiles: " + " ".join(all_generated_smiles[:10]))
            # --- Constraint check integration ---
            constraint_type = getattr(self.dataset_infos, "constraint_type", None)
            constraint_value = getattr(self.dataset_infos, "constraint_value", None)
            # Fallback: try to infer from self.cfg if available
            if constraint_type is None and hasattr(self, "cfg") and self.cfg is not None:
                constraint_type = getattr(self.cfg.model, "rev_proj", None)
                # Map old names to explicit ones
                if constraint_type == "ring_length_at_most":
                    constraint_value = getattr(self.cfg.model, "max_ring_length", None)
                elif constraint_type == "ring_length_at_least":
                    constraint_value = getattr(self.cfg.model, "min_ring_length", None)
                elif constraint_type == "ring_count_at_most":
                    constraint_value = getattr(self.cfg.model, "max_rings", None)
                elif constraint_type == "ring_count_at_least":
                    constraint_value = getattr(self.cfg.model, "min_rings", None)
            # List of constraint types that are enforced
            enforced_types = ["ring_count_at_most", "ring_count_at_least", "ring_length_at_most", "ring_length_at_least", "planar", "tree", "lobster"]
            boolean_constraints = ["planar", "tree", "lobster"]
            # Use computed_metrics if available, otherwise fall back to local metrics
            if computed_metrics is not None:
                # Merge so locally computed keys (like FCD) override or complement global ones (like Disconnected)
                metrics_to_pass = {**computed_metrics, **metrics}
            else:
                metrics_to_pass = metrics
            
            # Generate new structured tables for both val and test splits
            from ConStruct.reporting.report_builder import (
                detect_constraint_kind, constraint_caption,
                collect_core, collect_structural,
                chemistry_from_smiles, collect_chemistry, collect_timing,
                render_table_md, render_table_console, core_definitions_md
            )
            
            split = "test" if self.test else "val"
            N_total = len(all_generated_smiles)
            
            # meta for titles
            experiment_name = getattr(self.cfg.general, 'name', 'experiment')
            dataset_name = "QM9"  # hardcoded for now, could be made configurable
            diffusion_steps = getattr(self.cfg.model, 'diffusion_steps', None)
            eval_meta = {"sampler": "reverse"}  # or infer if you have faster sampling flag
            
            # constraint caption
            kind = detect_constraint_kind(metrics_to_pass, split, self.cfg)
            constraint_meta = {
                "max_rings": getattr(self.cfg.model, 'max_rings', None),
                "max_ring_length": getattr(self.cfg.model, 'max_ring_length', None),
                "min_rings": getattr(self.cfg.model, 'min_rings', None),
                "min_ring_length": getattr(self.cfg.model, 'min_ring_length', None),
            }
            constraint_str = constraint_caption(kind, constraint_meta)
            if kind != "none" and not bool(
                getattr(self.cfg.model, "use_projection", True)
            ):
                constraint_str += " (projection disabled; evaluation target only)"
            
            # New (STRUCTURAL, projector-consistent)
            ring_count_counts, ring_length_counts = self.collect_structural_distributions_from_graphs(generated_graphs)
            N_struct = sum(ring_count_counts)  # equals number of molecules

            # core/structural metrics (removed alignment)
            core_rows = collect_core(split, metrics_to_pass, N_total, self.cfg)
            struct_rows = collect_structural(
                split,
                metrics_to_pass,
                N_struct,  # denominator = all generated graphs (structural)
                ring_count_counts,              # per-molecule ring-count histogram
                ring_length_counts,             # per-molecule MAX ring-length histogram (with 0=acyclic)
                max_rings=getattr(self.cfg.model, 'max_rings', None),
                max_ring_length=getattr(self.cfg.model, 'max_ring_length', None),
                min_rings=getattr(self.cfg.model, 'min_rings', None),
                min_ring_length=getattr(self.cfg.model, 'min_ring_length', None),
                cfg=self.cfg,  # Pass configuration for constraint detection
            )
            
            # Collect actual timing data with proper defaults
            timing_dict = {
                "sampling_sec": self.total_sampling_time if self.total_sampling_time > 0 else None,  # total wall seconds spent in sampling for this split
                "sampling_batches": self.num_batches if self.num_batches > 0 else None,  # number of sampling batches used
                "proj_ms_mean": None,  # mean ms per projector call
                "proj_share_pct": None,  # % of sampling time spent in projector
                "epochs_to_best": None,  # int if available
                "early_stop_epoch": None,  # int if available
                "wall_clock_hhmm": None,  # overall run wall clock as "HH:MM" if available
            }
            
            # Get actual projection timing statistics
            proj_mean_ms, proj_share_pct = self.get_projection_stats()
            if proj_mean_ms is not None and proj_mean_ms > 0:
                timing_dict["proj_ms_mean"] = proj_mean_ms
            if proj_share_pct is not None and proj_share_pct > 0:
                timing_dict["proj_share_pct"] = proj_share_pct
            
            # Try to get epoch information from the model if available
            if hasattr(self, 'cfg') and self.cfg and hasattr(self.cfg.model, 'max_epochs'):
                timing_dict["epochs_to_best"] = self.cfg.model.max_epochs
            
            # Try to get early stopping information from the model if available
            if hasattr(self, 'model') and hasattr(self.model, 'best_val_nll'):
                # This indicates the model has early stopping tracking
                timing_dict["early_stop_epoch"] = "ES enabled"
            
            # Try to get wall clock time if available
            import time
            if hasattr(self, 'start_time'):
                elapsed_time = time.time() - self.start_time
                hours = int(elapsed_time // 3600)
                minutes = int((elapsed_time % 3600) // 60)
                timing_dict["wall_clock_hhmm"] = f"{hours:02d}:{minutes:02d}"
            
            # Check if this is a no-constraint experiment and set appropriate defaults
            if (
                hasattr(self, "cfg")
                and self.cfg
                and (
                    not bool(getattr(self.cfg.model, "use_projection", True))
                    or getattr(self.cfg.model, "rev_proj", None) in (None, "")
                )
            ):
                # No constraint experiment - set projection metrics to indicate no projection
                timing_dict["proj_ms_mean"] = 0.0  # No projection time
                timing_dict["proj_share_pct"] = 0.0  # No projection share
            
            # Collect device information
            import torch
            device_meta = {
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                "torch": torch.__version__,
                "cuda": torch.version.cuda if hasattr(torch.version, 'cuda') else "NA"
            }
            
            # Timing rows (appendix)
            timing_rows = collect_timing(timing_dict, N_total, device_meta)
            
            # ALSO PRINT TABLES TO CONSOLE FOR IMMEDIATE VISIBILITY
            
            print("\n" + "="*80)
            print("📊 CONSTRAINT SATISFACTION RESULTS")
            print("="*80)
            
            # Add definitions with bullet points
            print("\n📋 **Definitions**:")
            print("• **FCD** — Fréchet ChemNet Distance computed on valid canonical SMILES; lower is better")
            print("• **Unique/Novel/Valid** — proportions computed over all generated molecules (not only valid)")
            print("• **Disconnected** — share of graphs with >1 connected component")
            print("• **Property satisfied** — share of generated graphs meeting the configured structural target")
            print("• **V.U.N.** — product of Valid × Unique × Novel (in [0,100]%)")
            print()
            
            # Display the metric payload independently of the configured logger.
            print("🔍 **SAMPLING METRICS**:")
            print("=" * 60)
            for key, value in metrics_to_pass.items():
                if key.startswith(f"{split}_sampling/"):
                    print(f"  {key}: {value}")
            print("=" * 60)
            print()
            
            # Core metrics table
            print("\n🎯 CORE METRICS:")
            print("=" * 80)
            core_table = render_table_console(f"{experiment_name} — {split.upper()} — {dataset_name}", 
                                            f"Constraint: {constraint_str} · N={N_total:,} · diffusion_steps={diffusion_steps if diffusion_steps is not None else 'NA'} · sampler={eval_meta.get('sampler','reverse')}", 
                                            core_rows)
            print(core_table)
            
            # Structural metrics table
            print("\n🏗️ STRUCTURAL CONSTRAINT SATISFACTION:")
            print("=" * 80)
            struct_table = render_table_console(f"{experiment_name} — {split.upper()} — {dataset_name} — Structural", 
                                              f"Constraint: {constraint_str} · N={N_total:,} · diffusion_steps={diffusion_steps if diffusion_steps is not None else 'NA'} · sampler={eval_meta.get('sampler','reverse')}", 
                                              struct_rows)
            print(struct_table)
            

            
            # Timing metrics table (always show, even with NA values)
            print("\n⏱️ TIMING METRICS:")
            print("=" * 80)
            timing_table = render_table_console(f"{experiment_name} — {split.upper()} — {dataset_name} — Timing", 
                                              f"Constraint: {constraint_str} · N={N_total:,} · diffusion_steps={diffusion_steps if diffusion_steps is not None else 'NA'} · sampler={eval_meta.get('sampler','reverse')}", 
                                              timing_rows)
            print(timing_table)
            
            # Optional Chemistry appendix (kept as separate file if you want it)
            chem_stats = chemistry_from_smiles(all_generated_smiles or [])
            chem_rows = collect_chemistry(chem_stats)
            if chem_rows:
                # Print chemistry table to console
                print("\n🧪 CHEMISTRY METRICS:")
                print("=" * 80)
                chem_table = render_table_console(f"{experiment_name} — {split.upper()} — {dataset_name} — Chemistry", 
                                                f"Constraint: {constraint_str} · N={N_total:,} · diffusion_steps={diffusion_steps if diffusion_steps is not None else 'NA'} · sampler={eval_meta.get('sampler','reverse')}", 
                                                chem_rows)
                print(chem_table)
                
                # Chemistry metrics displayed in console only

        # Save in any case in the graphs folder
        os.makedirs("graphs", exist_ok=True)
        textfile = open(
            f"graphs/valid_unique_molecules_e{current_epoch}_GR{local_rank}.txt", "w"
        )
        textfile.writelines(all_generated_smiles)
        textfile.close()
        # Save in the root folder if test_model
        if self.test:
            filename = f"final_smiles_GR{local_rank}_{0}.txt"
            for i in range(2, 10):
                if os.path.exists(filename):
                    filename = f"final_smiles_GR{local_rank}_{i}.txt"
                else:
                    break
            with open(filename, "w") as fp:
                for smiles in all_generated_smiles:
                    # write each item on a new line
                    fp.write("%s\n" % smiles)
                print(f"All smiles saved on rank {local_rank}")
        # Compute statistics
        stat = (
            self.dataset_infos.statistics["test"]
            if self.test
            else self.dataset_infos.statistics["val"]
        )

        key = "val_sampling" if not self.test else "test_sampling"
        if self.is_molecular:
            # the molecule validity is calculated with charge information
            charge_w1, charge_w1_per_class = charge_distance(
                molecules, stat.charge_types, stat.atom_types, self.dataset_infos
            )
            self.charge_w1(charge_w1)
            metrics[f"{key}/ChargeW1"] = self.charge_w1.compute().item()
            valency_w1, valency_w1_per_class = valency_distance(
                molecules,
                stat.valencies,
                stat.atom_types,
                self.dataset_infos.atom_encoder,
            )
            self.valency_w1(valency_w1)
            # TODO: (line below) torch lightning stalls for multi-gpu sampling is number of samples to generate is <=2
            metrics[f"{key}/ValencyW1"] = self.valency_w1.compute().item()

        # if local_rank == 0:
        #     print(f"Sampling metrics", {k: round(val, 3) for k, val in metrics.items()})
        # if local_rank == 0:
        #     print(f"Molecular metrics computed.")

        # Track constraint violations
        constraint_violations = self._track_constraint_violations(generated_graphs, local_rank)
        
        # Compare projector vs evaluation logic for debugging
        constraint_type = None
        constraint_value = None
        
        # Try to get constraint info from config
        if hasattr(self, 'cfg') and self.cfg is not None:
            if hasattr(self.cfg.model, 'rev_proj'):
                if self.cfg.model.rev_proj == 'ring_length_at_most':
                    constraint_type = 'ring_length_at_most'
                    constraint_value = getattr(self.cfg.model, 'max_ring_length', None)
        
        if constraint_type == "ring_length_at_most" and constraint_value is not None:
            self._compare_projector_vs_evaluation(generated_graphs, constraint_type, constraint_value)
        
        return metrics

    def _track_constraint_violations(self, generated_graphs: list[PlaceHolder], local_rank: int):
        """Track constraint violations for generated graphs."""
        violations = {
            'ring_count': [],
            'ring_length': [],
            'planarity': []
        }
        
        # Detect constraint type from config
        constraint_type = None
        constraint_value = None
        
        if hasattr(self, 'cfg') and self.cfg is not None:
            if hasattr(self.cfg.model, 'rev_proj'):
                if self.cfg.model.rev_proj == 'ring_count_at_most':
                    constraint_type = 'ring_count_at_most'
                    constraint_value = getattr(self.cfg.model, 'max_rings', None)
                elif self.cfg.model.rev_proj == 'ring_length_at_most':
                    constraint_type = 'ring_length_at_most'
                    constraint_value = getattr(self.cfg.model, 'max_ring_length', None)
                elif self.cfg.model.rev_proj == 'ring_count_at_least':
                    constraint_type = 'ring_count_at_least'
                    constraint_value = getattr(self.cfg.model, 'min_rings', None)
                elif self.cfg.model.rev_proj == 'ring_length_at_least':
                    constraint_type = 'ring_length_at_least'
                    constraint_value = getattr(self.cfg.model, 'min_ring_length', None)
                elif self.cfg.model.rev_proj == 'planar':
                    constraint_type = 'planar'
                    constraint_value = None
        
        print(f"[VIOLATION TRACKING] Detected constraint: {constraint_type}, value: {constraint_value}")
        
        # Use the SAME graph construction method as projectors for consistency
        from ConStruct.projector.projector_utils import build_simple_graph_from_edge_tensor
        
        # Check each constraint type - ONLY the enforced constraint
        for batch_idx, batch in enumerate(generated_graphs):
            for graph_idx, (edge_mat, mask) in enumerate(zip(batch.E, batch.node_mask)):
                
                # Use the EXACT same graph construction as projectors
                nx_graph = build_simple_graph_from_edge_tensor(edge_mat, mask)
                
                # Check ring count constraint - ONLY if it's the enforced constraint
                if constraint_type == "ring_count_at_most" and constraint_value is not None:
                    from ConStruct.projector.is_ring.is_ring_count_at_most import has_at_most_n_rings
                    if not has_at_most_n_rings(nx_graph, constraint_value):
                        violations['ring_count'].append({
                            'index': f"{batch_idx}_{graph_idx}",
                            'max_allowed': constraint_value,
                            'actual': count_simple_cycles(nx_graph),
                            'source': f'rank{local_rank}'
                        })

                if constraint_type == "ring_count_at_least" and constraint_value is not None:
                    from ConStruct.projector.is_ring.is_ring_count_at_least import has_at_least_n_rings
                    if not has_at_least_n_rings(nx_graph, constraint_value):
                        violations['ring_count'].append({
                            'index': f"{batch_idx}_{graph_idx}",
                            'min_allowed': constraint_value,
                            'actual': count_simple_cycles(nx_graph),
                            'source': f'rank{local_rank}'
                        })
                
                # Check ring length constraint - ONLY if it's the enforced constraint
                if constraint_type == "ring_length_at_most" and constraint_value is not None:
                    from ConStruct.projector.is_ring.is_ring_length_at_most import has_rings_of_length_at_most
                    if not has_rings_of_length_at_most(nx_graph, constraint_value):
                        violations['ring_length'].append({
                            'index': f"{batch_idx}_{graph_idx}",
                            'max_allowed': constraint_value,
                            'actual': max_simple_cycle_length(nx_graph),
                            'source': f'rank{local_rank}'
                        })

                if constraint_type == "ring_length_at_least" and constraint_value is not None:
                    from ConStruct.projector.is_ring.is_ring_length_at_least import has_rings_of_length_at_least
                    if not has_rings_of_length_at_least(nx_graph, constraint_value):
                        violations['ring_length'].append({
                            'index': f"{batch_idx}_{graph_idx}",
                            'min_allowed': constraint_value,
                            'actual': max_simple_cycle_length(nx_graph),
                            'source': f'rank{local_rank}'
                        })
                
                # Check planarity constraint - ONLY if it's the enforced constraint
                if constraint_type == "planar":
                    if not is_planar(nx_graph):
                        violations['planarity'].append({
                            'index': f"{batch_idx}_{graph_idx}",
                            'source': f'rank{local_rank}'
                        })
        
        # Save violation reports
        self._save_violation_reports(violations, local_rank)

    def _save_violation_reports(self, violations, local_rank):
        """Save detailed violation reports to files."""
        # Use a single timestamp for all files to ensure consistency
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Get the enforced constraint type from config
        constraint_type = None
        if hasattr(self, 'cfg') and self.cfg is not None:
            if hasattr(self.cfg.model, 'rev_proj'):
                constraint_type = self.cfg.model.rev_proj
        
        # Only save violations for the enforced constraint
        enforced_violations = {}
        if constraint_type in {'ring_count_at_most', 'ring_count_at_least'}:
            enforced_violations['ring_count'] = violations['ring_count']
        elif constraint_type in {'ring_length_at_most', 'ring_length_at_least'}:
            enforced_violations['ring_length'] = violations['ring_length']
        elif constraint_type == 'planar':
            enforced_violations['planarity'] = violations['planarity']
        
        # Debug: Show what's being tracked
        print(f"[VIOLATION TRACKING] Enforced constraint: {constraint_type}")
        print(f"[VIOLATION TRACKING] Tracking violations for: {list(enforced_violations.keys())}")
        print(f"[VIOLATION TRACKING] Ignoring violations for: {[k for k in violations.keys() if k not in enforced_violations]}")
        
        # Save detailed violation reports
        for violation_type, violation_list in enforced_violations.items():
            if violation_list:
                # Save to CSV file
                csv_filename = f"violations_{violation_type}_rank{local_rank}_{timestamp}.csv"
                try:
                    import pandas as pd
                    df = pd.DataFrame(violation_list)
                    df.to_csv(csv_filename, index=False)
                    print(f"[VIOLATIONS] Saved {len(violation_list)} {violation_type} violations to {csv_filename}")
                except Exception as e:
                    print(f"[WARNING] Failed to save {violation_type} violations to CSV: {e}")
        
        # Save summary report with consistent timestamp
        summary_filename = f"violation_summary_rank{local_rank}_{timestamp}.txt"
        try:
            with open(summary_filename, 'w') as f:
                f.write(f"Constraint Violation Summary - Rank {local_rank}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Enforced Constraint: {constraint_type}\n")
                
                # Get constraint value
                constraint_value = None
                if hasattr(self, 'cfg') and self.cfg is not None:
                    if constraint_type == 'ring_count_at_most':
                        constraint_value = getattr(self.cfg.model, 'max_rings', None)
                    elif constraint_type == 'ring_length_at_most':
                        constraint_value = getattr(self.cfg.model, 'max_ring_length', None)
                    elif constraint_type == 'ring_count_at_least':
                        constraint_value = getattr(self.cfg.model, 'min_rings', None)
                    elif constraint_type == 'ring_length_at_least':
                        constraint_value = getattr(self.cfg.model, 'min_ring_length', None)
                
                if constraint_value is not None:
                    f.write(f"Constraint Value: {constraint_value}\n")
                f.write("\n")
                
                # Write violation counts
                for violation_type, violation_list in enforced_violations.items():
                    f.write(f"{violation_type.upper()} VIOLATIONS: {len(violation_list)}\n")
                
                f.write("\n")
            
            print(f"[VIOLATIONS] Summary saved to {summary_filename}")
        except Exception as e:
            print(f"[WARNING] Failed to save violation summary: {e}")

    def _compare_projector_vs_evaluation(self, generated_graphs, constraint_type, constraint_value):
        """Compare projector logic with evaluation logic to identify discrepancies."""
        print(f"\n[DEBUG] Comparing projector vs evaluation for {constraint_type} ≤ {constraint_value}")
        
        discrepancies = []
        total_molecules = 0
        
        for batch in generated_graphs:
            for edge_mat, mask in zip(batch.E, batch.node_mask):
                # Use unified graph construction helper
                from ConStruct.projector.projector_utils import build_simple_graph_from_edge_tensor
                nx_graph = build_simple_graph_from_edge_tensor(edge_mat, mask)
                
                total_molecules += 1
                
                try:
                    # Get evaluation metrics
                    from ConStruct.projector.graph_cycles import enumerate_simple_cycles_unique, max_simple_cycle_length
                    cycles = list(enumerate_simple_cycles_unique(nx_graph))
                    max_ring_length = max_simple_cycle_length(nx_graph)
                    
                    # Get projector decision
                    if constraint_type == "ring_length_at_most":
                        from ConStruct.projector.is_ring.is_ring_length_at_most import has_rings_of_length_at_most
                        projector_allows = has_rings_of_length_at_most(nx_graph, constraint_value)
                        evaluation_violates = max_ring_length > constraint_value
                        
                        if projector_allows != (not evaluation_violates):
                            discrepancies.append({
                                'max_ring_length': max_ring_length,
                                'constraint_value': constraint_value,
                                'projector_allows': projector_allows,
                                'evaluation_violates': evaluation_violates,
                                'cycles': cycles
                            })
                    
                except Exception as e:
                    print(f"[WARNING] Failed to analyze graph: {e}")
                    continue
        
        # Report findings
        print(f"[DEBUG] Analyzed {total_molecules} molecules")
        print(f"[DEBUG] Found {len(discrepancies)} projector-evaluation discrepancies")
        
        if discrepancies:
            print(f"[DEBUG] Discrepancy details:")
            for i, disc in enumerate(discrepancies[:5]):  # Show first 5
                print(f"  {i+1}. Max ring length: {disc['max_ring_length']}, Constraint: {disc['constraint_value']}")
                print(f"     Projector allows: {disc['projector_allows']}, Evaluation violates: {disc['evaluation_violates']}")
                print(f"     Cycles: {[len(c) for c in disc['cycles']]}")
        
        return discrepancies

    def compute_fcd(self, generated_smiles):
        fcd_model = fcd.load_ref_model()
        
        # Use single-threaded canonicalization to avoid multiprocessing conflicts
        # in SLURM/PyTorch Lightning environments
        # This fixes the hanging issue on second seed runs
        canonical_smiles_list = []
        for smile in generated_smiles:
            try:
                mol = Chem.MolFromSmiles(smile)
                if mol is not None:
                    canonical_smile = Chem.MolToSmiles(mol, canonical=True)
                    canonical_smiles_list.append(canonical_smile)
            except:
                continue
        
        generated_smiles = [smile for smile in canonical_smiles_list if smile is not None]

        if len(generated_smiles) <= 1:
            print("Not enough (<=1) valid smiles for FCD computation.")
            fcd_score = -1
        else:
            try:
                gen_activations = fcd.get_predictions(fcd_model, generated_smiles)
                gen_mu = np.mean(gen_activations, axis=0)
                gen_sigma = np.cov(gen_activations.T)
                target_mu = self.val_fcd_mu
                target_sigma = self.val_fcd_sigma
                try:
                    fcd_score = fcd.calculate_frechet_distance(
                        mu1=gen_mu,
                        sigma1=gen_sigma,
                        mu2=target_mu,
                        sigma2=target_sigma,
                    )
                except ValueError as e:
                    eps = 1e-6
                    print(f"Error in FCD computation: {e}. Increasing eps to {eps}")
                    eps_sigma = eps * np.eye(gen_sigma.shape[0])
                    gen_sigma = gen_sigma + eps_sigma
                    target_sigma = self.val_fcd_sigma + eps_sigma
                    fcd_score = fcd.calculate_frechet_distance(
                        mu1=gen_mu,
                        sigma1=gen_sigma,
                        mu2=target_mu,
                        sigma2=target_sigma,
                    )
            except Exception as e:
                print(f"FCD calculation failed: {e}")
                fcd_score = -1

        key = "val_sampling" if not self.test else "test_sampling"
        return {f"{key}/fcd score": fcd_score}

    def record_sampling_time(self, sampling_time: float):
        """Record sampling time for timing metrics."""
        self.total_sampling_time += sampling_time
        self.num_batches += 1
        self.avg_sampling_time = self.total_sampling_time / self.num_batches
    
    def record_projection_time(self, projection_time: float):
        """Record projection time for timing metrics."""
        self.total_projection_time += projection_time
        self.projection_times.append(projection_time)
        self.projection_calls += 1
    
    def get_projection_stats(self):
        """Get projection timing statistics."""
        if self.projection_calls == 0:
            return None, None
        
        mean_projection_time = self.total_projection_time / self.projection_calls
        projection_share = (self.total_projection_time / self.total_sampling_time * 100) if self.total_sampling_time > 0 else 0
        
        return mean_projection_time * 1000, projection_share  # Convert to ms
    
    def collect_structural_distributions_from_graphs(self, generated_graphs: list[PlaceHolder]):
        """
        Returns:
            ring_count_counts: list sized 10 → bins [0,1,2,3,4,5,6,7,8,9+]
            ring_length_counts: list sized 12 → bins:
                [0 (acyclic), 3,4,5,6,7,8,9,10,11,12, >12]
        """
        from ConStruct.projector.projector_utils import build_simple_graph_from_edge_tensor
        import networkx as nx

        # Ring-count histogram: 0..8, 9+ lumped
        ring_count_counts = [0] * 10

        # Max-ring-length histogram:
        # index 0 = acyclic
        # index 1..10 => lengths 3..12
        # index 11 => >12
        ring_length_counts = [0] * 12

        for batch in generated_graphs:
            for edge_mat, mask in zip(batch.E, batch.node_mask):
                g = build_simple_graph_from_edge_tensor(edge_mat, mask)
                try:
                    from ConStruct.projector.graph_cycles import enumerate_simple_cycles_unique
                    cycles = list(enumerate_simple_cycles_unique(g))
                except Exception:
                    # Count as unknown → treat as violating bucket for safety
                    # but here we place into " >12 " max-length bin to keep totals consistent
                    ring_count_counts[-1] += 1
                    ring_length_counts[-1] += 1
                    continue

                # Ring count
                r = len(cycles)
                ring_count_counts[r if r < 9 else 9] += 1

                # Max ring length (per molecule)
                if not cycles:
                    ring_length_counts[0] += 1  # acyclic
                else:
                    Lmax = max(len(c) for c in cycles)
                    if 3 <= Lmax <= 12:
                        idx = (Lmax - 3) + 1  # 3→1, 12→10
                        ring_length_counts[idx] += 1
                    else:
                        ring_length_counts[-1] += 1  # >12

        return ring_count_counts, ring_length_counts

    def collect_ring_distributions(self, generated_graphs: list[PlaceHolder]) -> Tuple[List[int], List[int]]:
        """
        Collect ring count and ring length distributions from generated graphs using structural methods.
        
        Args:
            generated_graphs: List of PlaceHolder objects containing generated graphs
            
        Returns:
            Tuple of (ring_count_counts, ring_length_counts) where each is a list of counts
        """
        import networkx as nx
        from ConStruct.projector.projector_utils import build_simple_graph_from_edge_tensor
        
        # Initialize distribution counters
        ring_count_counts = [0] * 10  # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9+ rings
        ring_length_counts = [0] * 10  # 0, 3, 4, 5, 6, 7, 8, 9, 10, 11+ atoms per ring (0 for acyclic)
        
        for batch in generated_graphs:
            for edge_mat, mask in zip(batch.E, batch.node_mask):
                try:
                    # Use unified graph construction helper
                    nx_graph = build_simple_graph_from_edge_tensor(edge_mat, mask)
                    
                    # Get all simple cycles (structural analysis; all simple rings)
                    cycles = list(enumerate_simple_cycles_unique(nx_graph))
                    
                    # Count rings (ring count)
                    num_rings = len(cycles)
                    if num_rings < len(ring_count_counts):
                        ring_count_counts[num_rings] += 1
                    else:
                        ring_count_counts[-1] += 1  # 9+ rings
                    
                    # Analyze ring lengths
                    if cycles:
                        # Graph has cycles - analyze each cycle length
                        for cycle in cycles:
                            ring_length = len(cycle)
                            if ring_length >= 3:  # Valid ring length
                                idx = ring_length - 2  # Convert to 0-based index (3->1, 4->2, etc.)
                                if idx < len(ring_length_counts):
                                    ring_length_counts[idx] += 1
                                else:
                                    ring_length_counts[-1] += 1  # 11+ atoms
                    else:
                        # Graph is acyclic - count as length 0
                        ring_length_counts[0] += 1
                            
                except Exception:
                    # Conservative: treat failure as acyclic (no rings)
                    ring_count_counts[0] += 1
                    ring_length_counts[0] += 1
                    continue
        
        return ring_count_counts, ring_length_counts
    
    def _calculate_avg_molecular_weight(self, smiles_list: List[str]) -> float:
        """Calculate average molecular weight for a list of SMILES."""
        if not smiles_list:
            return 0.0
        
        total_weight = 0.0
        valid_count = 0
        
        for smiles in smiles_list:
            if smiles and smiles != "error":
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        # Try to sanitize the molecule first
                        try:
                            Chem.SanitizeMol(mol)
                        except:
                            pass  # Continue even if sanitization fails
                        
                        # Try different methods to get molecular weight
                        weight = 0.0
                        try:
                            # Method 1: Use Descriptors.MolWt (proper import)
                            weight = Descriptors.MolWt(mol)
                        except Exception as e:
                            try:
                                # Method 2: Manual calculation as fallback
                                weight = 0.0
                                for atom in mol.GetAtoms():
                                    weight += atom.GetMass()
                            except Exception as e2:
                                continue
                        
                        if weight > 0:  # Only count if we got a valid weight
                            total_weight += weight
                            valid_count += 1
                except Exception as e:
                    continue
        
        return total_weight / valid_count if valid_count > 0 else 0.0
    
    def _calculate_avg_atom_count(self, smiles_list: List[str]) -> float:
        """Calculate average number of atoms for a list of SMILES."""
        if not smiles_list:
            return 0.0
        
        total_atoms = 0
        valid_count = 0
        
        for smiles in smiles_list:
            if smiles and smiles != "error":
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        total_atoms += mol.GetNumAtoms()
                        valid_count += 1
                except:
                    continue
        
        return total_atoms / valid_count if valid_count > 0 else 0.0
    
    def _calculate_avg_bond_count(self, smiles_list: List[str]) -> float:
        """Calculate average number of bonds for a list of SMILES."""
        if not smiles_list:
            return 0.0
        
        total_bonds = 0
        valid_count = 0
        
        for smiles in smiles_list:
            if smiles and smiles != "error":
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        total_bonds += mol.GetNumBonds()
                        valid_count += 1
                except:
                    continue
        
        return total_bonds / valid_count if valid_count > 0 else 0.0
    
    def _calculate_avg_ring_count(self, smiles_list: List[str]) -> float:
        """Calculate average ring count for a list of SMILES."""
        if not smiles_list:
            return 0.0
        
        total_rings = 0
        valid_count = 0
        
        for smiles in smiles_list:
            if smiles and smiles != "error":
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        ring_info = mol.GetRingInfo()
                        total_rings += ring_info.NumRings()
                        valid_count += 1
                except:
                    continue
        
        return total_rings / valid_count if valid_count > 0 else 0.0
    
    def _calculate_avg_ring_length(self, smiles_list: List[str]) -> float:
        """Calculate average ring length for a list of SMILES."""
        if not smiles_list:
            return 0.0
        
        total_ring_length = 0
        total_rings = 0
        valid_count = 0
        
        for smiles in smiles_list:
            if smiles and smiles != "error":
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        ring_info = mol.GetRingInfo()
                        rings = ring_info.AtomRings()
                        for ring in rings:
                            total_ring_length += len(ring)
                            total_rings += 1
                        valid_count += 1
                except:
                    continue
        
        return total_ring_length / total_rings if total_rings > 0 else 0.0
    
    def _calculate_avg_valency(self, smiles_list: List[str]) -> float:
        """Calculate average valency for a list of SMILES."""
        if not smiles_list:
            return 0.0
        
        total_valency = 0
        total_atoms = 0
        valid_count = 0
        
        for smiles in smiles_list:
            if smiles and smiles != "error":
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        for atom in mol.GetAtoms():
                            total_valency += atom.GetTotalValence()
                            total_atoms += 1
                        valid_count += 1
                except:
                    continue
        
        return total_valency / total_atoms if total_atoms > 0 else 0.0
    
    # Note: Novelty and uniqueness calculations are only done on valid molecules in the evaluate() method
    # These helper methods were incorrectly calculating on all molecules and have been removed


def charge_distance(molecules, target, atom_types_probabilities, dataset_infos):
    device = molecules[0].bond_types.device
    generated_distribution = torch.zeros_like(target).to(device)
    for molecule in molecules:
        for atom_type in range(target.shape[0]):
            mask = molecule.atom_types == atom_type
            if mask.sum() > 0:
                at_charges = dataset_infos.one_hot_charges(molecule.charges[mask])
                generated_distribution[atom_type] += at_charges.sum(dim=0)

    s = generated_distribution.sum(dim=1, keepdim=True)
    s[s == 0] = 1
    generated_distribution = generated_distribution / s

    cs_generated = torch.cumsum(generated_distribution, dim=1)
    cs_target = torch.cumsum(target, dim=1).to(device)

    w1_per_class = torch.sum(torch.abs(cs_generated - cs_target), dim=1)

    w1 = torch.sum(w1_per_class * atom_types_probabilities.to(device)).item()
    return w1, w1_per_class


def valency_distance(
    molecules, target_valencies, atom_types_probabilities, atom_encoder
):
    # Build a dict for the generated molecules that is similar to the target one
    num_atom_types = len(atom_types_probabilities)
    generated_valencies = {i: Counter() for i in range(num_atom_types)}
    for molecule in molecules:
        edge_types = molecule.bond_types
        edge_types[edge_types == 4] = 1.5
        valencies = torch.sum(edge_types, dim=0)
        for atom, val in zip(molecule.atom_types, valencies):
            generated_valencies[atom.item()][val.item()] += 1

    # Convert the valencies to a tensor of shape (num_atom_types, max_valency)
    max_valency_target = max(
        max(vals.keys()) if len(vals) > 0 else -1 for vals in target_valencies.values()
    )
    max_valency_generated = max(
        max(vals.keys()) if len(vals) > 0 else -1
        for vals in generated_valencies.values()
    )
    max_valency = int(max(max_valency_target, max_valency_generated))

    valencies_target_tensor = torch.zeros(num_atom_types, max_valency + 1)
    for atom_type, valencies in target_valencies.items():
        for valency, count in valencies.items():
            valencies_target_tensor[int(atom_encoder[atom_type]), int(valency)] = count

    valencies_generated_tensor = torch.zeros(num_atom_types, max_valency + 1)
    for atom_type, valencies in generated_valencies.items():
        for valency, count in valencies.items():
            valencies_generated_tensor[int(atom_type), int(valency)] = count

    # Normalize the distributions
    s1 = torch.sum(valencies_target_tensor, dim=1, keepdim=True)
    s1[s1 == 0] = 1
    valencies_target_tensor = valencies_target_tensor / s1

    s2 = torch.sum(valencies_generated_tensor, dim=1, keepdim=True)
    s2[s2 == 0] = 1
    valencies_generated_tensor = valencies_generated_tensor / s2

    cs_target = torch.cumsum(valencies_target_tensor, dim=1)
    cs_generated = torch.cumsum(valencies_generated_tensor, dim=1)

    w1_per_class = torch.sum(torch.abs(cs_target - cs_generated), dim=1)

    # print('debugging for molecular_metrics - valency_distance')
    # print('cs_target', cs_target)
    # print('cs_generated', cs_generated)

    total_w1 = torch.sum(w1_per_class * atom_types_probabilities).item()
    return total_w1, w1_per_class


class GeneratedNDistribution(Metric):
    full_state_update = False

    def __init__(self, max_n):
        super().__init__()
        self.add_state(
            "n_dist",
            default=torch.zeros(max_n + 1, dtype=torch.float),
            dist_reduce_fx="sum",
        )

    def update(self, molecules):
        for molecule in molecules:
            atom_types, _ = molecule
            n = atom_types.shape[0]
            self.n_dist[n] += 1

    def compute(self):
        return self.n_dist / torch.sum(self.n_dist)


class GeneratedNodesDistribution(Metric):
    full_state_update = False

    def __init__(self, num_atom_types):
        super().__init__()
        self.add_state(
            "node_dist",
            default=torch.zeros(num_atom_types, dtype=torch.float),
            dist_reduce_fx="sum",
        )

    def update(self, molecules):
        for molecule in molecules:
            atom_types, _ = molecule

            for atom_type in atom_types:
                assert (
                    int(atom_type) != -1
                ), "Mask error, the molecules should already be masked at the right shape"
                self.node_dist[int(atom_type)] += 1

    def compute(self):
        return self.node_dist / torch.sum(self.node_dist)


class GeneratedEdgesDistribution(Metric):
    full_state_update = False

    def __init__(self, num_edge_types):
        super().__init__()
        self.add_state(
            "edge_dist",
            default=torch.zeros(num_edge_types, dtype=torch.float),
            dist_reduce_fx="sum",
        )

    def update(self, molecules):
        for molecule in molecules:
            _, edge_types = molecule
            mask = torch.ones_like(edge_types)
            mask = torch.triu(mask, diagonal=1).bool()
            edge_types = edge_types[mask]
            unique_edge_types, counts = torch.unique(edge_types, return_counts=True)
            for type, count in zip(unique_edge_types, counts):
                self.edge_dist[type] += count

    def compute(self):
        return self.edge_dist / torch.sum(self.edge_dist)


class MeanNumberEdge(Metric):
    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state("total_edge", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, molecules, weight=1.0) -> None:
        for molecule in molecules:
            _, edge_types = molecule
            triu_edge_types = torch.triu(edge_types, diagonal=1)
            bonds = torch.nonzero(triu_edge_types)
            self.total_edge += len(bonds)
        self.total_samples += len(molecules)

    def compute(self):
        return self.total_edge / self.total_samples


class ValencyDistribution(Metric):
    full_state_update = False

    def __init__(self, max_n):
        super().__init__()
        self.add_state(
            "edgepernode_dist",
            default=torch.zeros(3 * max_n - 2, dtype=torch.float),
            dist_reduce_fx="sum",
        )

    def update(self, molecules) -> None:
        for molecule in molecules:
            _, edge_types = molecule
            edge_types[edge_types == 4] = 1.5
            valencies = torch.sum(edge_types, dim=0)
            unique, counts = torch.unique(valencies, return_counts=True)
            for valency, count in zip(unique, counts):
                self.edgepernode_dist[valency] += count

    def compute(self):
        return self.edgepernode_dist / torch.sum(self.edgepernode_dist)


class HistogramsMAE(MeanAbsoluteError):
    def __init__(self, target_histogram, **kwargs):
        """Compute the distance between histograms."""
        super().__init__(**kwargs)
        assert (target_histogram.sum() - 1).abs() < 1e-3
        self.target_histogram = target_histogram

    def update(self, pred):
        pred = pred / pred.sum()
        self.target_histogram = self.target_histogram.type_as(pred)
        super().update(pred, self.target_histogram)


#
# class UniquenessMetric(Metric):
#     is_differentiable = False
#     higher_is_better = True
#     full_state_update = True
#     def __init__(self):
#         """ Check if the number of unique molecules by concatenating the smiles. """
#         super().__init__(compute_on_cpu=True)
#         # add the smiles as a state
#         self.add_state("smiles", default=[], dist_reduce_fx=None)
#
#     def update(self, smiles_list):
#         self.smiles.extend(smiles_list)
#
#     def compute(self):
#         print(f"Computing uniqueness over {len(self.smiles)} smiles")
#         if len(self.smiles) == 0:
#             return 0.0
#         return len(set(self.smiles)) / len(self.smiles)


def smiles_from_generated_samples_file(generated_samples_file, atom_decoder):
    """ """
    smiles_list = []
    with open(generated_samples_file, "r") as f:
        # Repeat until the end of the file
        while True:
            # Check if we reached the end of the file
            line = f.readline()
            print("First line", line)
            if not line:
                break
            else:
                N = int(line.split("=")[1])

            # Extract X (labels or coordinates of the nodes)
            f.readline()
            X = list(map(int, f.readline().split()))
            X = torch.tensor(X)[:N]

            # Extract charges
            f.readline()
            charges = list(map(int, f.readline().split()))
            charges = torch.tensor(charges)
            N_before_mask = len(charges)
            charges = charges[:N]

            f.readline()
            E = []
            for i in range(N_before_mask):
                E.append(list(map(int, f.readline().split())))
            E = torch.tensor(E)[:N, :N]

            graph = PlaceHolder(X=X, E=E, charges=charges, y=None)
            f.readline()
            mol = Molecule(graph, atom_decoder)
            rdkit_mol = mol.rdkit_mol
            # Try to convert to smiles
            try:
                smiles = Chem.MolToSmiles(rdkit_mol)
            except:
                smiles = None
            print(smiles)
            smiles_list.append(smiles)

        # Save the smiles list to file
        with open(generated_samples_file.split(".")[0] + ".smiles", "w") as f:
            for smiles in smiles_list:
                f.write(smiles + "\n")
    return smiles


def check_ring_constraints(smiles_list, constraint_type, constraint_value, logger=print, timeout_seconds=30):
    total = 0
    passed = 0
    
    # Collect distribution data for better analysis
    ring_counts = []
    ring_lengths = []
    
    # Use a more robust approach without signal handling
    import time
    from concurrent.futures import ThreadPoolExecutor, TimeoutError
    
    def check_single_molecule(smi):
        """Check a single molecule for constraint satisfaction without signal handling."""
        if smi is None or smi == "error":
            return None
            
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return None
                
            if constraint_type == "ring_length_at_most":
                try:
                    ring_info = mol.GetRingInfo()
                    max_len = max((len(r) for r in ring_info.AtomRings()), default=0)
                    ring_lengths.append(max_len)
                    return max_len <= constraint_value
                except Exception as e:
                    logger(f"[WARNING] Ring length calculation failed for SMILES: {smi}, error: {e}")
                    return None
            elif constraint_type == "ring_length_at_least":
                try:
                    ring_info = mol.GetRingInfo()
                    max_len = max((len(r) for r in ring_info.AtomRings()), default=0)
                    ring_lengths.append(max_len)
                    return max_len >= constraint_value
                except Exception as e:
                    logger(f"[WARNING] Ring length calculation failed for SMILES: {smi}, error: {e}")
                    return None
                    
            elif constraint_type == "ring_count_at_most":
                try:
                    ring_info = mol.GetRingInfo()
                    n_rings = ring_info.NumRings() if ring_info else 0
                    ring_counts.append(n_rings)
                    return n_rings <= constraint_value
                except Exception as e:
                    logger(f"[WARNING] Ring count calculation failed for SMILES: {smi}, error: {e}")
                    return None
            elif constraint_type == "ring_count_at_least":
                try:
                    ring_info = mol.GetRingInfo()
                    n_rings = ring_info.NumRings() if ring_info else 0
                    ring_counts.append(n_rings)
                    return n_rings >= constraint_value
                except Exception as e:
                    logger(f"[WARNING] Ring count calculation failed for SMILES: {smi}, error: {e}")
                    return None
            else:
                return None
                
        except Exception as e:
            logger(f"[WARNING] RDKit parsing failed for SMILES: {smi}, error: {e}")
            return None
    
    # Process molecules in batches to avoid overwhelming the system
    batch_size = 50
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i + batch_size]
        
        # Use ThreadPoolExecutor for timeout handling without signals
        with ThreadPoolExecutor(max_workers=4) as executor:
            try:
                # Submit all molecules in the batch
                future_to_smi = {executor.submit(check_single_molecule, smi): smi for smi in batch}
                
                # Process results with timeout
                for future in future_to_smi:
                    try:
                        result = future.result(timeout=timeout_seconds)
                        if result is not None:
                            total += 1
                            if result:
                                passed += 1
                    except TimeoutError:
                        smi = future_to_smi[future]
                        logger(f"[WARNING] Constraint check timed out for SMILES: {smi}")
                        continue
                    except Exception as e:
                        smi = future_to_smi[future]
                        logger(f"[WARNING] Constraint check failed for SMILES: {smi}, error: {e}")
                        continue
                        
            except Exception as e:
                logger(f"[WARNING] Batch processing failed: {e}")
                continue
    
    # Determine the correct operator based on constraint type
    if constraint_type in ["ring_count_at_least", "ring_length_at_least"]:
        operator = "≥"
    else:
        operator = "≤"
    
    # Calculate satisfaction rate
    satisfaction_rate = (passed / total * 100) if total > 0 else 0.0
    
    # Enhanced output with distribution information
    # logger(f"[Constraint Check] {passed}/{total} molecules satisfy {constraint_type} {operator} {constraint_value} ({satisfaction_rate:.1f}%)")
    
    # Add distribution information for better understanding
    # if ring_counts and constraint_type.startswith("ring_count"):
    #     from collections import Counter
    #     count_dist = Counter(ring_counts)
    #     # Format distribution in a readable way
    #     dist_str = ", ".join([f"{count} rings: {freq} molecules" for count, freq in sorted(count_dist.items())])
    #     logger(f"[Distribution] Ring counts: {dist_str}")
    # elif ring_lengths and constraint_type.startswith("ring_length"):
    #     from collections import Counter
    #     length_dist = Counter(ring_lengths)
    #     # Format distribution in a readable way
    #     dist_str = ", ".join([f"{length}-atom rings: {freq} rings" for length, freq in sorted(length_dist.items())])
    #     logger(f"[Distribution] Ring lengths: {dist_str}")
    
    # Return results for potential use
    return {
        'constraint_type': constraint_type,
        'constraint_value': constraint_value,
        'total_molecules': total,
        'satisfied_molecules': passed,
        'satisfaction_rate': satisfaction_rate / 100.0,  # Return as decimal
        'ring_counts': ring_counts,
        'ring_lengths': ring_lengths
    }


def analyze_all_molecules_constraints(all_generated_smiles: List[str], constraint_type: str, constraint_value: int) -> Dict:
    """
    Analyze constraint satisfaction for ALL molecules (including invalid ones).
    This provides structural constraint analysis following ConStruct philosophy.
    
    Args:
        all_generated_smiles: List of all generated SMILES (including None and "error")
        constraint_type: Type of constraint
        constraint_value: Value of the constraint
        
    Returns:
        Dictionary with structural constraint analysis results
    """
    total_molecules = len(all_generated_smiles)
    valid_molecules = len([s for s in all_generated_smiles if s and s != "error"])
    
    # Analyze ALL molecules for structural constraints
    structural_ring_counts = []
    structural_ring_lengths = []
    structural_planar_count = 0
    structural_non_planar_count = 0
    
    for smiles in all_generated_smiles:
        if smiles and smiles != "error":
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    # Ring count analysis
                    ring_info = mol.GetRingInfo()
                    n_rings = ring_info.NumRings() if ring_info else 0
                    structural_ring_counts.append(n_rings)
                    
                    # Ring length analysis
                    if ring_info and ring_info.NumRings() > 0:
                        for ring in ring_info.AtomRings():
                            structural_ring_lengths.append(len(ring))
                    
                    # Planarity analysis
                    try:
                        # Generate 3D conformer for planarity check
                        AllChem.EmbedMolecule(mol)
                        AllChem.MMFFOptimizeMolecule(mol)
                        conf = mol.GetConformer()
                        coords = conf.GetPositions()
                        
                        if len(coords) >= 3:
                            # Calculate the normal vector of the plane formed by first 3 atoms
                            v1 = coords[1] - coords[0]
                            v2 = coords[2] - coords[0]
                            normal = np.cross(v1, v2)
                            normal = normal / np.linalg.norm(normal)
                            
                            # Check if all other atoms are close to this plane
                            planar = True
                            for i in range(3, len(coords)):
                                v = coords[i] - coords[0]
                                distance = abs(np.dot(v, normal))
                                if distance > 0.5:  # Threshold for planarity
                                    planar = False
                                    break
                            
                            if planar:
                                structural_planar_count += 1
                            else:
                                structural_non_planar_count += 1
                        else:
                            structural_non_planar_count += 1
                    except:
                        structural_non_planar_count += 1
            except:
                pass
    
    # Calculate structural constraint rates
    structural_ring_count_rates = {}
    for max_rings in range(7):  # 0 to 6 rings
        satisfied = sum(1 for count in structural_ring_counts if count <= max_rings)
        rate = satisfied / total_molecules if total_molecules > 0 else 0.0
        structural_ring_count_rates[f"≤{max_rings}"] = rate
    
    # Calculate structural ring length rates
    structural_ring_length_rates = {}
    for max_length in range(3, 9):  # 3 to 8 atoms
        satisfied = sum(1 for length in structural_ring_lengths if length <= max_length)
        rate = satisfied / len(structural_ring_lengths) if structural_ring_lengths else 0.0
        structural_ring_length_rates[f"≤{max_length}"] = rate
    
    # Calculate planarity rate
    structural_planarity_rate = structural_planar_count / total_molecules if total_molecules > 0 else 0.0
    
    return {
        'total_molecules': total_molecules,
        'valid_molecules': valid_molecules,
        'structural_ring_counts': structural_ring_counts,
        'structural_ring_lengths': structural_ring_lengths,
        'structural_planar_count': structural_planar_count,
        'structural_non_planar_count': structural_non_planar_count,
        'structural_ring_count_rates': structural_ring_count_rates,
        'structural_ring_length_rates': structural_ring_length_rates,
        'structural_planarity_rate': structural_planarity_rate
    }

def check_ring_constraints_all_molecules(smiles_list, constraint_type, constraint_value, logger=print, timeout_seconds=30):
    """
    Check ring constraints for ALL molecules (including invalid ones).
    This provides structural constraint analysis.
    """
    total = len(smiles_list)
    passed = 0
    
    # Collect distribution data for structural analysis
    ring_counts = []
    ring_lengths = []
    
    def check_single_molecule(smi):
        """Check a single molecule for constraint satisfaction."""
        if smi is None or smi == "error":
            return None
            
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return None
                
            if constraint_type == "ring_length_at_most":
                try:
                    ring_info = mol.GetRingInfo()
                    max_len = max((len(r) for r in ring_info.AtomRings()), default=0)
                    ring_lengths.append(max_len)
                    return max_len <= constraint_value
                except Exception as e:
                    logger(f"[WARNING] Ring length calculation failed for SMILES: {smi}, error: {e}")
                    return None
            elif constraint_type == "ring_length_at_least":
                try:
                    ring_info = mol.GetRingInfo()
                    max_len = max((len(r) for r in ring_info.AtomRings()), default=0)
                    ring_lengths.append(max_len)
                    return max_len >= constraint_value
                except Exception as e:
                    logger(f"[WARNING] Ring length calculation failed for SMILES: {smi}, error: {e}")
                    return None
                    
            elif constraint_type == "ring_count_at_most":
                try:
                    ring_info = mol.GetRingInfo()
                    n_rings = ring_info.NumRings() if ring_info else 0
                    ring_counts.append(n_rings)
                    return n_rings <= constraint_value
                except Exception as e:
                    logger(f"[WARNING] Ring count calculation failed for SMILES: {smi}, error: {e}")
                    return None
            elif constraint_type == "ring_count_at_least":
                try:
                    ring_info = mol.GetRingInfo()
                    n_rings = ring_info.NumRings() if ring_info else 0
                    ring_counts.append(n_rings)
                    return n_rings >= constraint_value
                except Exception as e:
                    logger(f"[WARNING] Ring count calculation failed for SMILES: {smi}, error: {e}")
                    return None
            else:
                return None
                
        except Exception as e:
            logger(f"[WARNING] RDKit parsing failed for SMILES: {smi}, error: {e}")
            return None
    
    # Process molecules in batches
    batch_size = 50
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i + batch_size]
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            try:
                future_to_smi = {executor.submit(check_single_molecule, smi): smi for smi in batch}
                
                for future in future_to_smi:
                    try:
                        result = future.result(timeout=timeout_seconds)
                        if result is not None:
                            if result:
                                passed += 1
                    except TimeoutError:
                        smi = future_to_smi[future]
                        logger(f"[WARNING] Constraint check timed out for SMILES: {smi}")
                        continue
                    except Exception as e:
                        smi = future_to_smi[future]
                        logger(f"[WARNING] Constraint check failed for SMILES: {smi}, error: {e}")
                        continue
                        
            except Exception as e:
                logger(f"[WARNING] Batch processing failed: {e}")
                continue
    
    # Determine the correct operator based on constraint type
    if constraint_type in ["ring_count_at_least", "ring_length_at_least"]:
        operator = "≥"
    else:
        operator = "≤"
    
    # Calculate satisfaction rate (structural - based on ALL molecules)
    satisfaction_rate = (passed / total * 100) if total > 0 else 0.0
    
    # Enhanced output with distribution information
    logger(f"[Structural Constraint Check] {passed}/{total} molecules satisfy {constraint_type} {operator} {constraint_value} ({satisfaction_rate}%)")
    
    # Add distribution information
    if ring_counts and constraint_type.startswith("ring_count"):
        from collections import Counter
        count_dist = Counter(ring_counts)
        dist_str = ", ".join([f"{count} rings: {freq} molecules" for count, freq in sorted(count_dist.items())])
        logger(f"[Structural Distribution] Ring counts: {dist_str}")
    elif ring_lengths and constraint_type.startswith("ring_length"):
        from collections import Counter
        length_dist = Counter(ring_lengths)
        dist_str = ", ".join([f"{length}-atom rings: {freq} rings" for length, freq in sorted(length_dist.items())])
        logger(f"[Structural Distribution] Ring lengths: {dist_str}")
    
    return {
        'constraint_type': constraint_type,
        'constraint_value': constraint_value,
        'total_molecules': total,
        'satisfied_molecules': passed,
        'satisfaction_rate': satisfaction_rate / 100.0,
        'ring_counts': ring_counts,
        'ring_lengths': ring_lengths
    }


def check_ring_constraints_all_molecules_from_graphs(generated_graphs, constraint_type, constraint_value, logger=print):
    """
    Check ring constraints for ALL molecules using graph-based structural analysis.
    This provides structural constraint analysis using the same graph construction as projectors.
    
    Args:
        generated_graphs: List of generated graph batches (same as WandB)
        constraint_type: Type of constraint ('ring_count_at_most', 'ring_length_at_most', etc.)
        constraint_value: Value of the constraint
        logger: Logger function
        
    Returns:
        Dictionary with structural constraint analysis results
    """
    total_molecules = 0
    satisfied_molecules = 0
    
    # Collect distribution data for structural analysis
    ring_counts = []
    ring_lengths = []
    
    # Use the same approach as WandB: direct graph analysis
    for batch in generated_graphs:
        for edge_mat, mask in zip(batch.E, batch.node_mask):
            # Use unified graph construction helper
            from ConStruct.projector.projector_utils import build_simple_graph_from_edge_tensor
            nx_graph = build_simple_graph_from_edge_tensor(edge_mat, mask)
            
            total_molecules += 1
            
            try:
                # Use all simple cycles for structural constraint
                from ConStruct.projector.graph_cycles import enumerate_simple_cycles_unique, max_simple_cycle_length
                cycles = list(enumerate_simple_cycles_unique(nx_graph))
                ring_count = len(cycles)
                ring_counts.append(ring_count)
                
                max_ring_length = max_simple_cycle_length(nx_graph)
                ring_lengths.append(max_ring_length)
                
                # Check constraint satisfaction
                if constraint_type == "ring_count_at_most":
                    satisfies_constraint = ring_count <= constraint_value
                elif constraint_type == "ring_length_at_most":
                    satisfies_constraint = max_ring_length <= constraint_value
                elif constraint_type == "ring_count_at_least":
                    satisfies_constraint = ring_count >= constraint_value
                elif constraint_type == "ring_length_at_least":
                    satisfies_constraint = max_ring_length >= constraint_value
                else:
                    satisfies_constraint = False
                
                if satisfies_constraint:
                    satisfied_molecules += 1
                    
            except Exception as e:
                # If structural analysis fails, assume constraint not satisfied
                ring_counts.append(0)
                ring_lengths.append(0)
                pass
    
    # Calculate satisfaction rate (structural - based on ALL molecules)
    satisfaction_rate = (satisfied_molecules / total_molecules) if total_molecules > 0 else 0.0
    
    # Determine the correct operator based on constraint type
    if constraint_type in ["ring_count_at_least", "ring_length_at_least"]:
        operator = "≥"
    else:
        operator = "≤"
    
    # Enhanced output with distribution information
    logger(f"[Structural Constraint Check (Graphs)] {satisfied_molecules}/{total_molecules} molecules satisfy {constraint_type} {operator} {constraint_value} ({satisfaction_rate*100}%)")
    
    return {
        'constraint_type': constraint_type,
        'constraint_value': constraint_value,
        'total_molecules': total_molecules,
        'satisfied_molecules': satisfied_molecules,
        'satisfaction_rate': satisfaction_rate,
        'ring_counts': ring_counts,
        'ring_lengths': ring_lengths
    }


def check_planarity_constraint(generated_graphs, logger=print):
    """
    Check planarity constraint for valid molecules only.
    Uses the same graph-based approach as WandB logging.
    
    Args:
        generated_graphs: List of generated graph batches (same as WandB)
        logger: Logger function
        
    Returns:
        Dictionary with planarity results
    """
    satisfied_molecules = 0
    total_molecules = 0
    
    # Use the same approach as WandB: direct graph analysis
    for batch in generated_graphs:
        for edge_mat, mask in zip(batch.E, batch.node_mask):
            # Use unified graph construction helper
            from ConStruct.projector.projector_utils import build_simple_graph_from_edge_tensor
            nx_graph = build_simple_graph_from_edge_tensor(edge_mat, mask)
            
            # Use the same is_planar function as WandB
            try:
                from ConStruct.projector.is_planar import is_planar
                is_planar_result = is_planar(nx_graph)
                total_molecules += 1
                if is_planar_result:
                    satisfied_molecules += 1
            except Exception as e:
                # If planarity check fails, assume non-planar
                total_molecules += 1
                pass
    
    satisfaction_rate = (satisfied_molecules / total_molecules) if total_molecules > 0 else 0.0
    
    return {
        'satisfied_molecules': satisfied_molecules,
        'total_molecules': total_molecules,
        'satisfaction_rate': satisfaction_rate
    }


def check_planarity_constraint_all_molecules(generated_graphs, logger=print):
    """
    Check planarity constraint for ALL molecules (including invalid ones).
    Uses the same graph-based approach as WandB logging.
    
    Args:
        generated_graphs: List of generated graph batches (same as WandB)
        logger: Logger function
        
    Returns:
        Dictionary with planarity results
    """
    satisfied_molecules = 0
    total_molecules = 0
    
    # Use the same approach as WandB: direct graph analysis
    for batch in generated_graphs:
        for edge_mat, mask in zip(batch.E, batch.node_mask):
            # Use unified graph construction helper
            from ConStruct.projector.projector_utils import build_simple_graph_from_edge_tensor
            nx_graph = build_simple_graph_from_edge_tensor(edge_mat, mask)
            
            # Use the same is_planar function as WandB
            try:
                from ConStruct.projector.is_planar import is_planar
                is_planar_result = is_planar(nx_graph)
                total_molecules += 1
                if is_planar_result:
                    satisfied_molecules += 1
            except Exception as e:
                # If planarity check fails, assume non-planar
                total_molecules += 1
                pass
    
    satisfaction_rate = (satisfied_molecules / total_molecules) if total_molecules > 0 else 0.0
    
    return {
        'satisfied_molecules': satisfied_molecules,
        'total_molecules': total_molecules,
        'satisfaction_rate': satisfaction_rate
    }


if __name__ == "__main__":
    file = ""
    atom_decoder = ["C", "N", "O", "F", "B", "Br", "Cl", "I", "P", "S", "Se", "Si"]

    smiles_list = smiles_from_generated_samples_file(file, atom_decoder)
    print(smiles_list)
