from collections import Counter
from typing import List

import torch.nn as nn
import networkx as nx
import numpy as np
import scipy.sparse as sp
import wandb
from torchmetrics import MeanMetric, MaxMetric, Metric, MeanAbsoluteError
import torch
from torch import Tensor
import fcd

from ConStruct.utils import PlaceHolder
from ConStruct.metrics.metrics_utils import (
    counter_to_tensor,
    wasserstein1d,
    total_variation1d,
)

from ConStruct.projector.projector_utils import has_lobster_components
from ConStruct.projector.graph_cycles import count_simple_cycles, max_simple_cycle_length
from ConStruct.projector.is_planar import is_planar


class SamplingMetrics(nn.Module):
    def __init__(self, dataset_infos, test, train_loader=None, val_loader=None, cfg=None):
        super().__init__()
        self.dataset_infos = dataset_infos
        self.test = test
        self.cfg = cfg
        self.stat = (
            self.dataset_infos.statistics["test"]
            if self.test
            else self.dataset_infos.statistics["val"]
        )


        # Add back the disconnected metric (from original implementation)
        self.disconnected = MeanMetric()
        self.mean_components = MeanMetric()
        self.max_components = MaxMetric()
        self.num_nodes_w1 = MeanMetric()
        self.node_types_tv = MeanMetric()
        self.edge_types_tv = MeanMetric()
        self.mean_planarity = MeanMetric()
        self.mean_no_cycles = MeanMetric()
        self.mean_lobster_components = MeanMetric()
        
        # Add ring constraint metrics (structural)
        self.mean_ring_count_satisfaction = MeanMetric()
        self.mean_ring_length_satisfaction = MeanMetric()
        
        self.deg_histogram = DegreeHistogramMetric(self.stat)

        if dataset_infos.is_molecular:
            from ConStruct.metrics.sampling_molecular_metrics import (
                SamplingMolecularMetrics,
            )

            self.domain_metrics = SamplingMolecularMetrics(
                dataset_infos=dataset_infos, test=test, cfg=self.cfg
            )
        elif dataset_infos.is_tls:
            from ConStruct.metrics.sampling_tls_metrics import TLSSamplingMetrics

            self.domain_metrics = TLSSamplingMetrics(
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                tls_type=dataset_infos.dataset_name,
            )
        else:
            from ConStruct.metrics.spectre_utils import (
                Comm20SamplingMetrics,
                PlanarSamplingMetrics,
                GridSamplingMetrics,
                TreeSamplingMetrics,
                LobsterSamplingMetrics,
                SBMSamplingMetrics,
                EgoSamplingMetrics,
                ProteinSamplingMetrics,
            )

            if dataset_infos.dataset_name == "comm-20":
                self.domain_metrics = Comm20SamplingMetrics(
                    train_dataloader=train_loader, val_dataloader=val_loader
                )
            elif dataset_infos.dataset_name in [
                "planar",
                "enzymes",  # not used
            ]:
                self.domain_metrics = PlanarSamplingMetrics(
                    train_dataloader=train_loader, val_dataloader=val_loader
                )
            elif dataset_infos.dataset_name == "grid":
                self.domain_metrics = GridSamplingMetrics(
                    train_dataloader=train_loader, val_dataloader=val_loader
                )
            elif dataset_infos.dataset_name == "tree":
                self.domain_metrics = TreeSamplingMetrics(
                    train_dataloader=train_loader, val_dataloader=val_loader
                )
            elif dataset_infos.dataset_name == "lobster":
                self.domain_metrics = LobsterSamplingMetrics(
                    train_dataloader=train_loader, val_dataloader=val_loader
                )
            elif dataset_infos.dataset_name == "sbm":
                self.domain_metrics = SBMSamplingMetrics(
                    train_dataloader=train_loader, val_dataloader=val_loader
                )
            elif dataset_infos.dataset_name == "protein":
                self.domain_metrics = ProteinSamplingMetrics(
                    train_dataloader=train_loader, val_dataloader=val_loader
                )
            elif dataset_infos.dataset_name == "ego":
                self.domain_metrics = EgoSamplingMetrics(
                    train_dataloader=train_loader, val_dataloader=val_loader
                )
            else:
                raise ValueError(
                    f"Dataset {dataset_infos.dataset_name} not implemented"
                )

    def reset(self):
        for metric in [
            self.disconnected,  # Add back disconnected metric
            self.mean_components,
            self.max_components,
            self.num_nodes_w1,
            self.node_types_tv,
            self.edge_types_tv,
            self.mean_planarity,
            self.mean_no_cycles,
            self.mean_lobster_components,
            self.mean_ring_count_satisfaction,
            self.mean_ring_length_satisfaction,
            self.deg_histogram,
        ]:
            metric.reset()
        self.domain_metrics.reset()

    def record_sampling_time(self, sampling_time: float):
        """Record sampling time for timing metrics."""
        if hasattr(self.domain_metrics, 'record_sampling_time'):
            self.domain_metrics.record_sampling_time(sampling_time)

    def record_projection_time(self, projection_time: float):
        """Record projection time for timing metrics."""
        if hasattr(self.domain_metrics, 'record_projection_time'):
            self.domain_metrics.record_projection_time(projection_time)

    def compute_all_metrics(
        self, generated_graphs: list, current_epoch, local_rank, log_to_wandb=True
    ):
        """Compare statistics of the generated data with statistics of the val/test set"""

        # Number of nodes
        self.num_nodes_w1(number_nodes_distance(generated_graphs, self.stat.num_nodes))

        # Node types
        node_type_tv, node_tv_per_class = node_types_distance(
            generated_graphs, self.stat.atom_types, save_histogram=self.test
        )
        self.node_types_tv(node_type_tv)

        # Edge types
        edge_types_tv, edge_tv_per_class = bond_types_distance(
            generated_graphs, self.stat.bond_types, save_histogram=self.test
        )
        self.edge_types_tv(edge_types_tv)

        # Components
        device = self.mean_components.device  # Use mean_components.device as it's safer
        connected_comp = connected_components(generated_graphs).to(device)
        self.disconnected(connected_comp > 1)  # Add back disconnected calculation
        self.mean_components(connected_comp)
        self.max_components(connected_comp)

        # Planarity
        planarity_ratios = planarity_ratio(generated_graphs).to(device)
        self.mean_planarity(planarity_ratios)

        # No cycles
        no_cycles_ratios = no_cycles_ratio(generated_graphs).to(device)
        self.mean_no_cycles(no_cycles_ratios)

        # Lobster components
        lobster_components_ratios = lobster_components_ratio(generated_graphs).to(
            device
        )
        self.mean_lobster_components(lobster_components_ratios)

        # Ring constraint satisfaction (structural)
        ring_constraint_info = None
        if hasattr(self, 'cfg') and self.cfg and hasattr(self.cfg.model, 'rev_proj'):
            if self.cfg.model.rev_proj == 'ring_count_at_most':
                max_rings = getattr(self.cfg.model, 'max_rings', 3)
                ring_count_ratios = ring_count_satisfaction_ratio(generated_graphs, max_rings).to(device)
                self.mean_ring_count_satisfaction(ring_count_ratios)
                ring_constraint_info = ('ring_count_at_most', max_rings)
            
            elif self.cfg.model.rev_proj == 'ring_length_at_most':
                max_ring_length = getattr(self.cfg.model, 'max_ring_length', 6)
                ring_length_ratios = ring_length_satisfaction_ratio(generated_graphs, max_ring_length).to(device)
                self.mean_ring_length_satisfaction(ring_length_ratios)
                ring_constraint_info = ('ring_length_at_most', max_ring_length)

            elif self.cfg.model.rev_proj == 'ring_count_at_least':
                min_rings = getattr(self.cfg.model, 'min_rings', 1)
                ring_count_ratios = ring_count_at_least_satisfaction_ratio(
                    generated_graphs, min_rings
                ).to(device)
                self.mean_ring_count_satisfaction(ring_count_ratios)
                ring_constraint_info = ('ring_count_at_least', min_rings)

            elif self.cfg.model.rev_proj == 'ring_length_at_least':
                min_ring_length = getattr(self.cfg.model, 'min_ring_length', 3)
                ring_length_ratios = ring_length_at_least_satisfaction_ratio(
                    generated_graphs, min_ring_length
                ).to(device)
                self.mean_ring_length_satisfaction(ring_length_ratios)
                ring_constraint_info = ('ring_length_at_least', min_ring_length)

        # Degree distributions
        self.deg_histogram(generated_graphs)

        # Log
        key = "val_sampling" if not self.test else "test_sampling"
        (
            generated_deg_hist,
            diff_deg_hist,
            abs_diff_deg_hist,
        ) = self.deg_histogram.get_hists_to_log(target_hist=self.stat.degree_hist[0])
        histogram_values = {
            f"{key}/generated_deg_hist": {
                "counts": generated_deg_hist[0].tolist(),
                "bins": generated_deg_hist[1].tolist(),
            },
            f"{key}/diff_deg_hist": {
                "counts": diff_deg_hist[0].tolist(),
                "bins": diff_deg_hist[1].tolist(),
            },
            f"{key}/abs_diff_deg_hist": {
                "counts": abs_diff_deg_hist[0].tolist(),
                "bins": abs_diff_deg_hist[1].tolist(),
            },
        }
        to_log = {
            f"{key}/NumNodesW1": self.num_nodes_w1.compute().item(),
            f"{key}/NodeTypesTV": self.node_types_tv.compute().item(),
            f"{key}/EdgeTypesTV": self.edge_types_tv.compute().item(),
            f"{key}/Disconnected": self.disconnected.compute().item() * 100,
            f"{key}/MeanComponents": self.mean_components.compute().item(),
            f"{key}/MaxComponents": self.max_components.compute().item(),
            f"{key}/planarity": self.mean_planarity.compute().item(),
            f"{key}/no_cycles": self.mean_no_cycles.compute().item(),
            f"{key}/lobster_components": self.mean_lobster_components.compute().item(),
            **histogram_values,
        }

        # Add ring constraint metrics to WandB logging (using stored info)
        if ring_constraint_info:
            constraint_type, constraint_value = ring_constraint_info
            if constraint_type in {'ring_count_at_most', 'ring_count_at_least'}:
                satisfaction_rate = self.mean_ring_count_satisfaction.compute().item()
                to_log[f"{key}/ring_count_satisfaction"] = satisfaction_rate
                to_log[f"{key}/ring_count_violation"] = 1 - satisfaction_rate
            elif constraint_type in {'ring_length_at_most', 'ring_length_at_least'}:
                satisfaction_rate = self.mean_ring_length_satisfaction.compute().item()
                to_log[f"{key}/ring_length_satisfaction"] = satisfaction_rate
                to_log[f"{key}/ring_length_violation"] = 1 - satisfaction_rate

        if self.domain_metrics is not None:
            # Compute domain metrics normally
            log_domain_metrics = self.domain_metrics.forward(
                generated_graphs, current_epoch, local_rank, computed_metrics=to_log
            )
            to_log.update(log_domain_metrics)
            ratios = self.compute_ratios_to_ref(
                reference_metrics=(
                    self.dataset_infos.test_reference_metrics
                    if self.test
                    else self.dataset_infos.val_reference_metrics
                ),
                generated_metrics=log_domain_metrics,
            )
            to_log.update(ratios)

        if log_to_wandb and wandb.run:
            wandb_payload = dict(to_log)
            wandb_payload[f"{key}/generated_deg_hist"] = wandb.Histogram(
                np_histogram=generated_deg_hist
            )
            wandb_payload[f"{key}/diff_deg_hist"] = wandb.Histogram(
                np_histogram=diff_deg_hist
            )
            wandb_payload[f"{key}/abs_diff_deg_hist"] = wandb.Histogram(
                np_histogram=abs_diff_deg_hist
            )
            wandb.log(wandb_payload, commit=False)
        if local_rank == 0:
            printable_metrics = {}
            for metric_name, value in to_log.items():
                if isinstance(value, dict) and "counts" in value:
                    printable_metrics[metric_name] = [
                        round(element, 3) for element in value["counts"]
                    ]
                else:
                    try:
                        printable_metrics[metric_name] = round(value, 3)
                    except TypeError:
                        printable_metrics[metric_name] = value
            print("Sampling metrics", printable_metrics)

        return to_log, edge_tv_per_class

    def compute_ratios_to_ref(self, reference_metrics, generated_metrics):
        def compute_ratio(generated, reference):
            # Protect against division by 0 (when ref is 0)
            return (
                (generated / reference)
                if reference != 0
                else (generated + 1e-8) / (reference + 1e-8)
            )

        ratios = {}
        log_key = "test" if self.test else "val"
        if self.dataset_infos.is_molecular:
            for key in ["fcd score"]:  # TODO: Add more metrics?
                ratios[f"{log_key}_ratio/{key}"] = compute_ratio(
                    generated=generated_metrics[f"{log_key}_sampling/{key}"],
                    reference=reference_metrics[f"val_sampling/{key}"],
                )
            ratios[f"{log_key}_ratio/average"] = sum(ratios.values()) / len(ratios)
        elif self.dataset_infos.is_tls:
            # TLS entries
            tls_ratios = {}
            metrics_to_ratio = [
                metric
                for metric in generated_metrics
                if (
                    "tls_metric" in metric
                    and "frac" not in metric
                    and "mean_tls_validity" not in metric
                )
            ]
            for key in metrics_to_ratio:
                tls_ratios[f"{log_key}_ratio/{key.split('/')[-1]}"] = compute_ratio(
                    generated=generated_metrics[key],
                    reference=reference_metrics[key],
                )
            tls_ratios[f"{log_key}_ratio/average_tls"] = sum(tls_ratios.values()) / len(
                tls_ratios
            )
            # Spectre entries
            spectre_ratios = {}
            for key, val in generated_metrics.items():
                if key in self.domain_metrics.metrics_list:
                    spectre_ratios[f"{log_key}_ratio/{key}"] = compute_ratio(
                        generated=generated_metrics[key],
                        reference=reference_metrics[key],
                    )
            spectre_ratios[f"{log_key}_ratio/average_spectre"] = sum(
                spectre_ratios.values()
            ) / len(spectre_ratios)
            # Merge
            ratios = {**tls_ratios, **spectre_ratios}
        else:  # spectre datasets
            for key, val in generated_metrics.items():
                if key in self.domain_metrics.metrics_list:
                    ratios[f"{log_key}_ratio/{key}"] = compute_ratio(
                        generated=generated_metrics[key],
                        reference=reference_metrics[key],
                    )
            ratios[f"{log_key}_ratio/average"] = sum(ratios.values()) / len(ratios)
        return ratios


def number_nodes_distance(generated_graphs: List[PlaceHolder], dataset_counts):
    """each element of generated graphs is a batch of graphs."""
    max_number_nodes = max(dataset_counts.keys())
    reference_n = torch.zeros(max_number_nodes + 1, device=generated_graphs[0].X.device)
    for n, count in dataset_counts.items():
        reference_n[n] = count

    c = Counter()
    for batch in generated_graphs:
        num_nodes = torch.sum(batch.node_mask, dim=1)
        c.update(num_nodes.tolist())

    generated_n = counter_to_tensor(c).to(reference_n.device)
    return wasserstein1d(generated_n, reference_n)


def node_types_distance(
    generated_graphs: List[PlaceHolder], target, save_histogram=False
):
    device = generated_graphs[0].X.device
    target = target.to(device)
    generated_distribution = torch.zeros_like(target)

    for batch in generated_graphs:
        unique, counts = torch.unique(batch.X[batch.X >= 0], return_counts=True)
        for u, c in zip(unique.to(device), counts.to(device)):
            generated_distribution[u] += c

    if save_histogram:
        np.save("generated_atom_types.npy", generated_distribution.cpu().numpy())
    return total_variation1d(generated_distribution, target)


def bond_types_distance(
    generated_graphs: List[PlaceHolder], target, save_histogram=False
):
    device = generated_graphs[0].X.device
    target = target.to(device)
    generated_distribution = torch.zeros_like(target)

    for batch in generated_graphs:
        unique, counts = torch.unique(batch.E[batch.E >= 0], return_counts=True)
        for u, c in zip(unique.to(device), counts.to(device)):
            generated_distribution[u] += c

    if save_histogram:
        np.save("generated_bond_types.npy", generated_distribution.cpu().numpy())
    tv, tv_per_class = total_variation1d(generated_distribution, target.to(device))
    return tv, tv_per_class


def connected_components(generated_graphs: List[PlaceHolder]):
    all_num_components = []
    for batch in generated_graphs:
        for edge_mat, mask in zip(batch.E, batch.node_mask):
            # Use unified graph construction helper
            from ConStruct.projector.projector_utils import build_simple_graph_from_edge_tensor
            nx_graph = build_simple_graph_from_edge_tensor(edge_mat, mask)
            num_components = nx.number_connected_components(nx_graph)
            all_num_components.append(num_components)
    all_num_components = torch.tensor(
        all_num_components, device=generated_graphs[0].X.device
    )
    return all_num_components


# TODO: all these metrics have similar structure, just apply a different check function. Abstract this.
def planarity_ratio(generated_graphs: List[PlaceHolder]):
    planarity_ratios = []
    for batch in generated_graphs:
        for edge_mat, mask in zip(batch.E, batch.node_mask):
            # Use unified graph construction helper
            from ConStruct.projector.projector_utils import build_simple_graph_from_edge_tensor
            nx_graph = build_simple_graph_from_edge_tensor(edge_mat, mask)
            is_planar_result = is_planar(nx_graph)
            planarity_ratios.append(int(is_planar_result))
    planarity_ratios = torch.tensor(
        planarity_ratios, device=generated_graphs[0].X.device
    )
    return planarity_ratios


def no_cycles_ratio(generated_graphs: List[PlaceHolder]):
    no_cycles_list = []
    for batch in generated_graphs:
        for edge_mat, mask in zip(batch.E, batch.node_mask):
            # Use unified graph construction helper
            from ConStruct.projector.projector_utils import build_simple_graph_from_edge_tensor
            nx_graph = build_simple_graph_from_edge_tensor(edge_mat, mask)
            no_cycles_list.append(nx.is_forest(nx_graph))
    no_cycles_tg = torch.tensor(no_cycles_list, device=generated_graphs[0].X.device)
    return no_cycles_tg


from ConStruct.projector.projector_utils import has_lobster_components


def lobster_components_ratio(generated_graphs: List[PlaceHolder]):
    lobster_components_list = []
    for batch in generated_graphs:
        for edge_mat, mask in zip(batch.E, batch.node_mask):
            # Use unified graph construction helper
            from ConStruct.projector.projector_utils import build_simple_graph_from_edge_tensor
            nx_graph = build_simple_graph_from_edge_tensor(edge_mat, mask)
            lobster_components_list.append(has_lobster_components(nx_graph))
    lobster_components_tg = torch.tensor(
        lobster_components_list, device=generated_graphs[0].X.device
    )
    return lobster_components_tg


def ring_count_satisfaction_ratio(generated_graphs: List[PlaceHolder], max_rings: int):
    """
    Calculate the ratio of graphs that satisfy ring count constraint (structural).
    
    Args:
        generated_graphs: List of generated graph batches
        max_rings: Maximum allowed ring count (constraint value)
    
    Returns:
        Tensor of satisfaction ratios (1 for satisfied, 0 for not satisfied)
    """
    satisfaction_ratios = []
    
    for batch in generated_graphs:
        for edge_mat, mask in zip(batch.E, batch.node_mask):
            # Use unified graph construction helper
            from ConStruct.projector.projector_utils import build_simple_graph_from_edge_tensor
            nx_graph = build_simple_graph_from_edge_tensor(edge_mat, mask)
            
            # Check if graph satisfies ring count constraint (structural; all simple rings)
            try:
                ring_count = count_simple_cycles(nx_graph)
                satisfies_constraint = int(ring_count <= max_rings)
            except Exception:
                # Conservative fallback: treat failure as not satisfied
                satisfies_constraint = 0
            satisfaction_ratios.append(satisfies_constraint)
    
    satisfaction_ratios = torch.tensor(
        satisfaction_ratios, device=generated_graphs[0].X.device
    )
    return satisfaction_ratios


def ring_count_at_least_satisfaction_ratio(
    generated_graphs: List[PlaceHolder], min_rings: int
):
    """Return one indicator per graph for the structural ring-count lower bound."""
    from ConStruct.projector.is_ring.is_ring_count_at_least import has_at_least_n_rings
    from ConStruct.projector.projector_utils import build_simple_graph_from_edge_tensor

    satisfied = []
    for batch in generated_graphs:
        for edge_mat, mask in zip(batch.E, batch.node_mask):
            try:
                graph = build_simple_graph_from_edge_tensor(edge_mat, mask)
                satisfied.append(int(has_at_least_n_rings(graph, min_rings)))
            except Exception:
                satisfied.append(0)
    return torch.tensor(satisfied, device=generated_graphs[0].X.device)


def ring_length_at_least_satisfaction_ratio(
    generated_graphs: List[PlaceHolder], min_ring_length: int
):
    """Return one indicator per graph for maximum simple-cycle length >= L."""
    from ConStruct.projector.is_ring.is_ring_length_at_least import (
        has_rings_of_length_at_least,
    )
    from ConStruct.projector.projector_utils import build_simple_graph_from_edge_tensor

    satisfied = []
    for batch in generated_graphs:
        for edge_mat, mask in zip(batch.E, batch.node_mask):
            try:
                graph = build_simple_graph_from_edge_tensor(edge_mat, mask)
                satisfied.append(
                    int(has_rings_of_length_at_least(graph, min_ring_length))
                )
            except Exception:
                satisfied.append(0)
    return torch.tensor(satisfied, device=generated_graphs[0].X.device)


def ring_length_satisfaction_ratio(generated_graphs: List[PlaceHolder], max_ring_length: int):
    """
    Calculate the ratio of graphs that satisfy max ring length constraint (structural).
    
    Args:
        generated_graphs: List of generated graph batches
        max_ring_length: Maximum allowed ring length (constraint value)
    
    Returns:
        Tensor of satisfaction ratios (1 for satisfied, 0 for not satisfied)
    """
    satisfaction_ratios = []
    
    for batch in generated_graphs:
        for edge_mat, mask in zip(batch.E, batch.node_mask):
            # Use unified graph construction helper
            from ConStruct.projector.projector_utils import build_simple_graph_from_edge_tensor
            nx_graph = build_simple_graph_from_edge_tensor(edge_mat, mask)
            
            # Check if graph satisfies max ring length constraint (structural; all simple rings)
            try:
                max_cycle_length = max_simple_cycle_length(nx_graph)
                satisfies_constraint = int(max_cycle_length <= max_ring_length)
            except Exception:
                # Conservative fallback: treat failure as not satisfied
                satisfies_constraint = 0
            satisfaction_ratios.append(satisfies_constraint)
    
    satisfaction_ratios = torch.tensor(
        satisfaction_ratios, device=generated_graphs[0].X.device
    )
    return satisfaction_ratios


class DegreeHistogramMetric(Metric):
    def __init__(self, dataset_stats, upper_margin=5):
        super().__init__()
        self.max_possible_degree = max(dataset_stats.num_nodes.keys()) - 1
        max_target_degree = dataset_stats.degree_hist[1][-1]
        self.num_bins = (
            min(
                max_target_degree + upper_margin + 1,  # +1 for last bin
                self.max_possible_degree,
            )
            + 1  # +1 for degree 0
        )  # ,
        self.add_state(
            "histogram",
            default=torch.zeros(self.num_bins),
            dist_reduce_fx="sum",
        )

    def update(self, local_generated_graphs) -> None:
        local_hist = self.compute_deg_histogram(local_generated_graphs)
        self.histogram += local_hist

    def compute(self):
        """Return normalized histogram.

        Since it is in compute, self.histogram will be synced (reduced, i.e., summed) across all GPUs, and thus we can normalize it.
        """
        return self.histogram / self.histogram.sum()

    def compute_deg_histogram(self, generated_graphs: List[PlaceHolder]):
        generated_degrees = []
        for batch in generated_graphs:
            for edge_mat, mask in zip(batch.E, batch.node_mask):
                # Use unified graph construction helper
                from ConStruct.projector.projector_utils import build_simple_graph_from_edge_tensor
                nx_graph = build_simple_graph_from_edge_tensor(edge_mat, mask)
                degrees = [nx_graph.degree(node) for node in nx_graph.nodes()]
                generated_degrees.extend(degrees)

        generated_hist = self.get_histogram(generated_degrees)

        return generated_hist

    def get_histogram(self, generated_degrees):
        generated_degrees = torch.tensor(
            generated_degrees, dtype=torch.float32, device=self.histogram.device
        )
        # torch.histogram does not work with CUDA. This function does
        gen_hist = torch.histc(
            generated_degrees,
            self.max_possible_degree + 1,
            min=0,
            max=self.max_possible_degree,
        )
        large_degrees_sum = torch.sum(gen_hist[self.num_bins - 1 :])
        gen_hist = torch.cat(
            (gen_hist[: self.num_bins - 1], large_degrees_sum.unsqueeze(0))
        )

        # Not normalized because different local histograms will be summed later
        return gen_hist

    def get_hists_to_log(self, target_hist):
        self_normalized_hist = self.compute()  # get hist synced across GPUs

        # Output numpy objects to easen interface with wandb
        self_normalized_hist = self_normalized_hist.cpu().numpy()
        # pad shorter histogram
        if len(target_hist) < len(self_normalized_hist):
            padding = (0, len(self_normalized_hist) - len(target_hist))
            target_hist = np.pad(target_hist, padding)
        else:
            padding = (0, len(target_hist) - len(self_normalized_hist))
            self_normalized_hist = np.pad(self_normalized_hist, padding)

        # Output numpy objects to easen interface with wandb
        diff_hist = self_normalized_hist - target_hist
        assert diff_hist.sum() < 1e-3  # Assert both are normalized to the same value
        abs_diff_hist = np.abs(diff_hist)

        # Ensure bin_edges has length histogram_length + 1
        histogram_length = len(self_normalized_hist)
        bin_edges = np.arange(histogram_length + 1)
        generated_hist_to_log = self_normalized_hist, bin_edges
        diff_hist_to_log = diff_hist, bin_edges
        abs_diff_hist_to_log = abs_diff_hist, bin_edges

        # Assert target hist sums to 1
        return generated_hist_to_log, diff_hist_to_log, abs_diff_hist_to_log


class HistogramsMAE(MeanAbsoluteError):
    def __init__(self, target_histogram, **kwargs):
        """Compute the distance between histograms."""
        super().__init__(**kwargs)
        assert (target_histogram.sum() - 1).abs() < 1e-3
        self.target_histogram = target_histogram

    def update(self, pred):
        pred = pred / pred.sum()
        target = self.target_histogram.to(pred.device)
        super().update(pred, target)


class CEPerClass(Metric):
    full_state_update = True

    def __init__(self, class_id):
        super().__init__()
        self.class_id = class_id
        self.add_state("total_ce", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.softmax = torch.nn.Softmax(dim=-1)
        self.binary_cross_entropy = torch.nn.BCELoss(reduction="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets.
        Args:
            preds: Predictions from model   (bs, n, d) or (bs, n, n, d)
            target: Ground truth values     (bs, n, d) or (bs, n, n, d)
        """
        target = target.reshape(-1, target.shape[-1])
        mask = (target != 0.0).any(dim=-1)

        prob = self.softmax(preds)[..., self.class_id]
        prob = prob.flatten()[mask]

        target = target[:, self.class_id]
        target = target[mask]

        output = self.binary_cross_entropy(prob, target)
        self.total_ce += output
        self.total_samples += prob.numel()

    def compute(self):
        return self.total_ce / self.total_samples


class MeanNumberEdge(Metric):
    full_state_update = True

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
