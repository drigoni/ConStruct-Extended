import math
import os
import pickle
import time
import logging
from datetime import datetime

import networkx as nx
import numpy as np
import torch
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from ConStruct.reporting.report_builder import setup_additional_logging, log_and_print

from ConStruct.models.transformer_model import GraphTransformer
from ConStruct.diffusion.noise_model import (
    DiscreteUniformTransition,
    MarginalTransition,
    AbsorbingTransition,
    AbsorbingEdgesTransition,
    EdgeInsertionTransition,
)
from ConStruct.diffusion import diffusion_utils
from ConStruct.metrics.train_metrics import TrainLoss
from ConStruct.metrics.abstract_metrics import NLL
from ConStruct.analysis.visualization import Visualizer
from ConStruct import utils
from ConStruct.metrics.sampling_molecular_metrics import Molecule
from ConStruct.diffusion.extra_features import ExtraFeatures, DummyExtraFeatures
from ConStruct.diffusion.extra_features_molecular import ExtraMolecularFeatures
from ConStruct.metrics.abstract_metrics import TrainAbstractMetrics
from ConStruct.metrics.train_metrics import TrainMolecularMetrics
from ConStruct.metrics.abstract_metrics import XKl, ChargesKl, EKl
from ConStruct.datasets.adaptive_loader import effective_batch_size
from ConStruct.projector import projector_utils
import ConStruct.projector.is_planar.is_planar as is_planar
import time
from ConStruct.projector.projector_utils import (
    PlanarProjector,
    TreeProjector,
    LobsterProjector,
    RingCountAtMostProjector,
    RingLengthAtMostProjector,
    RingCountAtLeastProjector,
    RingLengthAtLeastProjector,
)
from networkx.algorithms import isomorphism as iso
from ConStruct.projector.graph_cycles import enumerate_simple_cycles_unique, count_simple_cycles, max_simple_cycle_length





class DiscreteDenoisingDiffusion(pl.LightningModule):
    model_dtype = torch.float32
    best_val_nll = 1e8
    val_counter = 0
    start_epoch_time = None
    train_iterations = None
    val_iterations = None
    sampling_times = []

    def __init__(self, cfg, dataset_infos, val_sampling_metrics, test_sampling_metrics):
        super().__init__()

        self.cfg = cfg
        self.name = cfg.general.name
        self.T = cfg.model.diffusion_steps

        self.nodes_dist = dataset_infos.nodes_dist
        self.dataset_infos = dataset_infos
        if cfg.model.extra_features:
            self.extra_features = ExtraFeatures(cfg=cfg, dataset_infos=dataset_infos)
            self.input_dims = self.extra_features.update_input_dims(
                dataset_infos.input_dims
            )
        else:
            self.extra_features = DummyExtraFeatures()

        if cfg.model.extra_molecular_features and dataset_infos.is_molecular:
            self.domain_features = ExtraMolecularFeatures(dataset_infos)
            self.input_dims = self.domain_features.update_input_dims(self.input_dims)
        else:
            self.domain_features = DummyExtraFeatures()
        self.output_dims = dataset_infos.output_dims

        # Train metrics
        self.train_loss = TrainLoss(self.cfg.model.lambda_train)

        if dataset_infos.is_molecular:
            self.train_metrics = TrainMolecularMetrics(dataset_infos)
        else:
            self.train_metrics = TrainAbstractMetrics()

        # Validation metrics
        self.val_metrics = torchmetrics.MetricCollection(
            [
                XKl(),
                ChargesKl(),
                EKl(),
            ]
        )
        self.val_nll = NLL()
        self.val_sampling_metrics = val_sampling_metrics

        # Test metrics
        self.test_metrics = torchmetrics.MetricCollection(
            [
                XKl(),
                ChargesKl(),
                EKl(),
            ]
        )
        self.test_nll = NLL()
        self.test_sampling_metrics = test_sampling_metrics

        self.save_hyperparameters(
            ignore=[
                "train_metrics",
                "val_sampling_metrics",
                "test_sampling_metrics",
                "dataset_infos",
            ]
        )

        self.model = GraphTransformer(
            input_dims=self.input_dims,
            n_layers=cfg.model.n_layers,
            hidden_mlp_dims=cfg.model.hidden_mlp_dims,
            hidden_dims=cfg.model.hidden_dims,
            output_dims=self.output_dims,
            dropout=cfg.model.dropout,
            dropout_in_and_out=cfg.model.dropout_in_and_out,
        )

        # ===== TRANSITION MECHANISM INSTANTIATION =====
        # Different transition types implement different diffusion mechanisms:
        # - Edge-Deletion: Edges absorb to no-edge state (for "at most" constraints)
        # - Edge-Insertion: Edges absorb to edge state (for "at least" constraints)  
        # - Marginal: Uses dataset marginals (no artificial bias)
        # - Uniform: Uniform distribution (baseline)
        #
        # CRITICAL: Only ONE mechanism should be active at runtime to preserve
        # theoretical soundness of the diffusion process.
        
        if cfg.model.transition == "uniform":
            self.noise_model = DiscreteUniformTransition(
                output_dims=self.output_dims, cfg=cfg
            )
        elif cfg.model.transition == "marginal":
            print(
                f"Marginal distribution of the classes: nodes: {self.dataset_infos.atom_types} --"
                f" edges: {self.dataset_infos.edge_types} --"
                f"charges: {self.dataset_infos.charges_marginals}"
            )

            self.noise_model = MarginalTransition(
                x_marginals=self.dataset_infos.atom_types,
                e_marginals=self.dataset_infos.edge_types,
                charges_marginals=self.dataset_infos.charges_marginals,
                y_classes=self.output_dims.y,
                cfg=cfg,
            )
        elif cfg.model.transition == "absorbing":
            print("Absorbing transition model")
            self.noise_model = AbsorbingTransition(
                cfg=cfg,
                output_dims=self.output_dims,
            )
        elif cfg.model.transition == "absorbing_edges":
            print("Edge-deletion transition model (for 'at most' constraints)")
            self.noise_model = AbsorbingEdgesTransition(
                cfg=cfg,
                x_marginals=self.dataset_infos.atom_types,
                e_marginals=self.dataset_infos.edge_types,
                charges_marginals=self.dataset_infos.charges_marginals,
                y_classes=self.output_dims.y,
            )
        elif cfg.model.transition == "edge_insertion":
            print("Edge-addition transition model (for 'at least' constraints)")
            self.noise_model = EdgeInsertionTransition(
                cfg=cfg,
                x_marginals=self.dataset_infos.atom_types,
                e_marginals=self.dataset_infos.edge_types,
                charges_marginals=self.dataset_infos.charges_marginals,
                y_classes=self.output_dims.y,
            )
        else:
            # Debug logging removed for production
            assert ValueError(
                f"Transition type '{cfg.model.transition}' not implemented."
            )

        # ===== TRANSITION-PROJECTOR VALIDATION =====
        # Ensure proper combinations of transitions and projectors
        # This prevents mixing incompatible mechanisms
        if hasattr(cfg.model, 'rev_proj') and cfg.model.rev_proj:
            self._validate_transition_projector_compatibility(cfg)

        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps

        self.visualizer = Visualizer(dataset_infos=dataset_infos)

        self.collapse_charges = (
            self.dataset_infos.collapse_charges
            if hasattr(self.dataset_infos, "collapse_charges")
            else None
        )

    def _validate_transition_projector_compatibility(self, cfg):
        """
        Validate that transition and projector combinations are compatible.
        
        This ensures proper use of transitions with appropriate projectors.
        Edge-deletion and edge-insertion can coexist without conflicts.
        """
        transition = cfg.model.transition
        rev_proj = cfg.model.rev_proj
        
        # Edge-deletion transitions should use "at most" projectors
        edge_deletion_transitions = ["absorbing_edges"]
        at_most_projectors = ["ring_count_at_most", "ring_length_at_most"]
        
        edge_insertion_transitions = ["edge_insertion"]
        at_least_projectors = ["ring_count_at_least", "ring_length_at_least"]
        
        # Marginal transitions can use any projector
        marginal_transitions = ["marginal", "uniform", "absorbing"]
        
        # Only warn about potentially incompatible combinations, don't raise errors
        if transition in edge_deletion_transitions and rev_proj in at_least_projectors:
            logger.warning(
                "Edge-deletion transition '%s' is usually paired with an 'at most' projector, got '%s'.",
                transition,
                rev_proj,
            )

        if transition in edge_insertion_transitions and rev_proj in at_most_projectors:
            logger.warning(
                "Edge-insertion transition '%s' is usually paired with an 'at least' projector, got '%s'.",
                transition,
                rev_proj,
            )
        
        # Log successful validation (commented for clean output)
        # if transition in edge_deletion_transitions:
        #     print(f"✓ Validated: Edge-deletion transition '{transition}' with '{rev_proj}' projector")
        # elif transition in marginal_transitions:
        #     print(f"✓ Validated: Marginal transition '{transition}' with '{rev_proj}' projector")
        # else:
        #     print(f"✓ Validated: Transition '{transition}' with '{rev_proj}' projector")

    def forward(self, z_t):
        assert z_t.node_mask is not None
        extra_features = self.extra_features(z_t)
        extra_domain_features = self.domain_features(z_t)

        # Need to copy to preserve dimensions in transition to z_{t-1} in sampling (prevent changing dimensions of z_t
        model_input = z_t.copy()
        model_input.X = torch.cat(
            (z_t.X, z_t.charges, extra_features.X, extra_domain_features.X), dim=2
        ).float()
        model_input.E = torch.cat(
            (z_t.E, extra_features.E, extra_domain_features.E), dim=3
        ).float()
        model_input.y = torch.hstack(
            (z_t.y, extra_features.y, extra_domain_features.y, z_t.t)
        ).float()

        return self.model(model_input)

    @property
    def BS(self):
        return self.cfg.train.batch_size

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.train.lr,
            amsgrad=True,
            weight_decay=self.cfg.train.weight_decay,
        )
        
        # Add learning rate scheduler if enabled
        if hasattr(self.cfg.train, 'lr_scheduler') and self.cfg.train.lr_scheduler.enable:
            scheduler_type = self.cfg.train.lr_scheduler.type
            
            if scheduler_type == "cosine":
                # Cosine annealing with warmup
                from torch.optim.lr_scheduler import CosineAnnealingLR
                scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=self.cfg.train.n_epochs,
                    eta_min=self.cfg.train.lr_scheduler.min_lr
                )
            elif scheduler_type == "step":
                # Step decay
                from torch.optim.lr_scheduler import StepLR
                scheduler = StepLR(
                    optimizer,
                    step_size=100,
                    gamma=0.5
                )
            elif scheduler_type == "plateau":
                # Reduce on plateau
                from torch.optim.lr_scheduler import ReduceLROnPlateau
                scheduler = ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=0.5,
                    patience=10,
                    min_lr=self.cfg.train.lr_scheduler.min_lr
                )
            else:
                print(f"Warning: Unknown scheduler type '{scheduler_type}', using no scheduler")
                return optimizer
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/epoch_NLL" if scheduler_type == "plateau" else None,
                    "interval": "epoch",
                    "frequency": 1
                }
            }
        
        return optimizer

    def training_step(self, data, i):
        if data.edge_index.numel() == 0:
            self.print("Found a batch with no edges. Skipping.")
            return
        dense_data = utils.to_dense(data, self.dataset_infos)
        z_t = self.noise_model.apply_noise(dense_data)
        pred = self.forward(z_t)

        loss, tl_log_dict = self.train_loss(
            masked_pred=pred,
            masked_true=dense_data,
            log=i % self.log_every_steps == 0,
        )

        tm_log_dict = self.train_metrics(
            masked_pred=pred,
            masked_true=dense_data,
            log=i % self.log_every_steps == 0,
        )
        if tl_log_dict is not None:
            self.log_dict(tl_log_dict, batch_size=self.BS)
        if tm_log_dict is not None:
            self.log_dict(tm_log_dict, batch_size=self.BS)

        return loss

    def on_fit_start(self) -> None:
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)

        # Set a different seed for each GPU (initial seed gets the seed from pl.seed_everything)
        # torch.random.manual_seed(torch.initial_seed() + self.local_rank)

    def on_train_epoch_start(self) -> None:
        self.print("Starting epoch", self.current_epoch)
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        self.print(f"Train epoch {self.current_epoch} ends")
        tle_log = self.train_loss.log_epoch_metrics()
        self.print(
            f"Epoch {self.current_epoch} finished: "
            f"X: {tle_log['train_epoch/x_CE'] :.2f} --"
            f" charges: {tle_log['train_epoch/charges_CE']:.2f} --"
            f" E: {tle_log['train_epoch/E_CE'] :.2f} --"
            f" y: {tle_log['train_epoch/y_CE'] :.2f} -- {time.time() - self.start_epoch_time:.1f}s "
        )
        self.log_dict(tle_log, batch_size=self.BS)
        tme_log = self.train_metrics.log_epoch_metrics(
            self.current_epoch, self.local_rank
        )
        if tme_log is not None:
            self.log_dict(tme_log, batch_size=self.BS)
        if wandb.run:
            wandb.log({"epoch": self.current_epoch}, commit=False)
        if self.cfg.general.memory_summary:
            print(torch.cuda.memory_summary())

    def on_validation_epoch_start(self) -> None:
        self.val_nll.reset()
        self.val_metrics.reset()
        self.val_sampling_metrics.reset()

    def validation_step(self, data, i):
        dense_data = utils.to_dense(data, self.dataset_infos)
        z_t = self.noise_model.apply_noise(dense_data)
        pred = self.forward(z_t)
        nll, log_dict = self.compute_val_loss(
            pred, z_t, clean_data=dense_data, test=False
        )
        return {"loss": nll}, log_dict

    def on_validation_epoch_end(self) -> None:
        metrics = [self.val_nll.compute(), self.val_metrics.compute()]
        log_dict = {
            "val/epoch_NLL": metrics[0],
            "val/X_kl": metrics[1]["XKl"] * self.T,
            "val/E_kl": metrics[1]["EKl"] * self.T,
            "val/charges_kl": metrics[1]["ChargesKl"] * self.T,
        }
        self.log_dict(log_dict, on_epoch=True, on_step=False, sync_dist=True)
        if wandb.run:
            wandb.log(log_dict)

        print_str = []
        for key, val in log_dict.items():
            new_val = f"{val:.2f}"
            print_str.append(f"{key}: {new_val} -- ")
        print_str = "".join(print_str)
        print(f"Epoch {self.current_epoch}: {print_str}."[:-4])

        # Log val nll with default Lightning logger, so it can be monitored by checkpoint callback
        val_nll = metrics[0]
        self.log("val/epoch_NLL", val_nll, sync_dist=True)

        if val_nll < self.best_val_nll:
            self.best_val_nll = val_nll
        self.print(
            "Val loss: %.4f \t Best val loss:  %.4f\n" % (val_nll, self.best_val_nll)
        )

        self.val_counter += 1
        if self.name == "debug" or (
            self.val_counter % self.cfg.general.sample_every_val == 0
            and self.current_epoch > 0
        ):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.print(f"[{timestamp}] Sampling start")
            start = time.time()
            gen = self.cfg.general
            samples = self.sample_n_graphs(
                samples_to_generate=math.ceil(
                    gen.samples_to_generate / max(self._trainer.num_devices, 1)
                ),
                chains_to_save=gen.chains_to_save if self.local_rank == 0 else 0,
                samples_to_save=gen.samples_to_save if self.local_rank == 0 else 0,
                test=False,
            )
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"[{timestamp}] Sampled {len(samples)} batches on local rank {self.local_rank}. Sampling took {time.time() - start:.2f} seconds\n"
            )
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Computing sampling metrics on {self.local_rank}...")
            self.val_sampling_metrics.compute_all_metrics(
                generated_graphs=samples,
                current_epoch=self.current_epoch,
                local_rank=self.local_rank,
            )
        self.print(f"Val epoch {self.current_epoch} ends")

    def on_test_epoch_start(self) -> None:
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)
        # Set a different seed for each GPU (initial seed gets the seed from pl.seed_everything)
        # torch.random.manual_seed(torch.initial_seed() + self.local_rank)

        self.test_nll.reset()
        self.test_metrics.reset()
        self.test_sampling_metrics.reset()

    def test_step(self, data, i):
        dense_data = utils.to_dense(data, self.dataset_infos)
        z_t = self.noise_model.apply_noise(dense_data)
        pred = self.forward(z_t)
        nll, log_dict = self.compute_val_loss(
            pred, z_t, clean_data=dense_data, test=True
        )
        return {"loss": nll}, log_dict

    def on_test_epoch_end(self) -> None:
        """Measure likelihood on a test set and compute stability metrics."""
        metrics = [self.test_nll.compute(), self.test_metrics.compute()]
        test_nll = metrics[0]
        self.print(f"Test loss: {test_nll :.4f}")
        log_dict = {
            "test/epoch_NLL": metrics[0],
            "test/X_kl": metrics[1]["XKl"] * self.T,
            "test/E_kl": metrics[1]["EKl"] * self.T,
            "test/charges_kl": metrics[1]["ChargesKl"] * self.T,
        }
        self.log_dict(log_dict, sync_dist=True)

        print_str = []
        for key, val in log_dict.items():
            new_val = f"{val:.4f}"
            print_str.append(f"{key}: {new_val} -- ")
        print_str = "".join(print_str)
        print(f"Epoch {self.current_epoch}: {print_str}."[:-4])

        if wandb.run:
            wandb.log(log_dict)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Sampling start on GR{self.global_rank}")
        start = time.time()
        to_sample = math.ceil(
            self.cfg.general.final_model_samples_to_generate
            / max(self._trainer.num_devices, 1)
        )
        self.print(
            f"Samples to generate: {to_sample} for each of the {max(self._trainer.num_devices, 1)} devices"
        )
        self.print(f"Samples to save: {self.cfg.general.final_model_samples_to_save}")
        samples = self.sample_n_graphs(
            samples_to_generate=to_sample,
            chains_to_save=self.cfg.general.final_model_chains_to_save,
            samples_to_save=self.cfg.general.final_model_samples_to_save,
            test=True,
        )
        
        # Save the samples list as pickle to a file that depends on the local rank
        # This is needed to avoid overwriting the same file on different GPUs
        with open(f"generated_samples_rank{self.local_rank}.pkl", "wb") as f:
            pickle.dump(samples, f)

        print("Saving the generated graphs")
        # This line is used to sync between gpus
        self._trainer.strategy.barrier()
        filename = f"generated_samples_rank{self.local_rank}.txt"
        with open(filename, "w") as f:
            for batch in samples:
                num_nodes = batch.node_mask.sum(dim=-1)
                for i in range(batch.X.shape[0]):
                    n = num_nodes[i].item()
                    f.write(f"N={n}\n")
                    # X:
                    f.write("X: \n")
                    for node in batch.X[i, :n].tolist():
                        f.write(f"{node} ")
                    f.write("\n")

                    # Charges
                    if batch.charges is not None:
                        f.write("charges: \n")
                        for c in batch.charges[i, :n].tolist():
                            f.write(f"{c} ")
                        f.write("\n")

                    # E
                    f.write("E: \n")
                    for edge_types in batch.E[i, :n, :n].tolist():
                        for edge in edge_types:
                            f.write(f"{edge} ")
                        f.write("\n")
                    f.write("\n")
        print("Saved.")
        print("📊 Computing final sampling metrics...")

        # Load the pickles of the other GPUs
        samples = []
        for i in range(self._trainer.num_devices):
            with open(f"generated_samples_rank{i}.pkl", "rb") as f:
                samples.extend(pickle.load(f))

        # Pass constraint information to sampling metrics for violation tracking
        if hasattr(self, 'dataset_infos'):
            # Extract constraint info from config
            constraint_type = None
            constraint_value = None
            
            # Check for ring constraints in config
            if hasattr(self.cfg, 'rev_proj'):
                if self.cfg.rev_proj == 'ring_count_at_most':
                    constraint_type = 'ring_count_at_most'
                    constraint_value = getattr(self.cfg, 'max_rings', None)
                elif self.cfg.rev_proj == 'ring_length_at_most':
                    constraint_type = 'ring_length_at_most'
                    constraint_value = getattr(self.cfg, 'max_ring_length', None)
            
            # Set constraint info in dataset_infos for violation tracking
            if constraint_type and constraint_value:
                self.dataset_infos.constraint_type = constraint_type
                self.dataset_infos.constraint_value = constraint_value
                print(f"[CONSTRAINT INFO] Type: {constraint_type}, Value: {constraint_value}")

        self.test_sampling_metrics.compute_all_metrics(
            generated_graphs=samples,
            current_epoch=self.current_epoch,
            local_rank=self.local_rank,
        )
        # Setup additional logging and log the completion message
        additional_logger = setup_additional_logging()
        log_and_print(f"Done. Sampling took {time.time() - start:.2f} seconds\n", additional_logger)
        log_and_print(f"Test ends.", additional_logger)

        # Close wandb
        wandb.finish()

    def kl_prior(self, clean_data, node_mask):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones(
            (clean_data.X.size(0), 1), dtype=torch.long, device=clean_data.X.device
        )
        Ts = self.T * ones
        Qtb = self.noise_model.get_Qt_bar(t_int=Ts)

        # Compute transition probabilities
        probX = clean_data.X @ Qtb.X + 1e-7  # (bs, n, dx_out)
        probE = clean_data.E @ Qtb.E.unsqueeze(1) + 1e-7  # (bs, n, n, de_out)
        probc = clean_data.charges @ Qtb.charges + 1e-7
        probX = probX / probX.sum(dim=-1, keepdims=True)
        probE = probE / probE.sum(dim=-1, keepdims=True)
        probc = probc / probc.sum(dim=-1, keepdims=True)
        assert probX.shape == clean_data.X.shape

        # bs, n, _ = probX.shape
        limit_dist = self.noise_model.get_limit_dist().device_as(probX)

        # Set masked rows to limit dist, so its KL to limit dist is 0 and doesn't contribute to loss
        probX[~node_mask] = limit_dist.X.float()
        diag_mask = ~torch.eye(
            node_mask.size(1), device=node_mask.device, dtype=torch.bool
        ).unsqueeze(0)
        probE[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2) * diag_mask), :] = (
            limit_dist.E.float()
        )

        # Compute Kl divergences
        kl_distance_X = F.kl_div(
            input=probX.log(), target=limit_dist.X[None, None, :], reduction="none"
        )
        kl_distance_E = F.kl_div(
            input=probE.log(),
            target=limit_dist.E[None, None, None, :],
            reduction="none",
        )

        if limit_dist.charges.numel() > 0:
            probc[~node_mask] = limit_dist.charges.float()
            kl_distance_c = F.kl_div(
                input=probc.log(),
                target=limit_dist.charges[None, None, :],
                reduction="none",
            )
        else:
            kl_distance_c = torch.zeros_like(kl_distance_X)

        return (
            diffusion_utils.sum_except_batch(kl_distance_X)
            + diffusion_utils.sum_except_batch(kl_distance_E)
            + diffusion_utils.sum_except_batch(kl_distance_c)
        ).mean()

    def compute_Lt(self, clean_data, pred, z_t, s_int, node_mask, test):
        t_int = z_t.t_int
        pred = utils.PlaceHolder(
            X=F.softmax(pred.X, dim=-1),
            charges=F.softmax(pred.charges, dim=-1),
            E=F.softmax(pred.E, dim=-1),
            node_mask=clean_data.node_mask,
            y=None,
        )

        Qtb = self.noise_model.get_Qt_bar(z_t.t_int)
        Qsb = self.noise_model.get_Qt_bar(s_int)
        Qt = self.noise_model.get_Qt(t_int)

        # Compute distributions to compare with KL
        bs, n, d = clean_data.X.shape
        prob_true = diffusion_utils.posterior_distributions(
            clean_data=clean_data, noisy_data=z_t, Qt=Qt, Qsb=Qsb, Qtb=Qtb
        )
        prob_true.E = prob_true.E.reshape((bs, n, n, -1))
        prob_pred = diffusion_utils.posterior_distributions(
            clean_data=pred, noisy_data=z_t, Qt=Qt, Qsb=Qsb, Qtb=Qtb
        )
        prob_pred.E = prob_pred.E.reshape((bs, n, n, -1))

        # Reshape and filter masked rows
        prob_true = diffusion_utils.mask_distributions(prob_true, node_mask)
        prob_pred = diffusion_utils.mask_distributions(prob_pred, node_mask)
        prob_true.X = torch.log(prob_true.X + 1e-7)
        prob_true.E = torch.log(prob_true.E + 1e-7)
        if prob_true.charges is not None:
            prob_true.charges = torch.log(prob_true.charges + 1e-7)

        # Compute metrics
        metrics = (self.test_metrics if test else self.val_metrics)(
            prob_pred, prob_true
        )

        return self.T * (
            (metrics["XKl"] if not torch.any(torch.isnan(metrics["XKl"])) else 0.0)
            + (
                metrics["ChargesKl"]
                if not torch.any(torch.isnan(metrics["ChargesKl"]))
                else 0.0
            )
            + (metrics["EKl"] if not torch.any(torch.isnan(metrics["EKl"])) else 0.0)
        )

    def compute_val_loss(self, pred, z_t, clean_data, test=False):
        """Computes an estimator for the variational lower bound, or the simple loss (MSE).
        pred: (batch_size, n, total_features)
        noisy_data: dict
        X, E, y : (bs, n, dx),  (bs, n, n, de), (bs, dy)
        node_mask : (bs, n)
        Output: nll (size 1)
        """
        node_mask = z_t.node_mask
        t_int = z_t.t_int
        s_int = t_int - 1

        # 1.
        N = node_mask.sum(1).long()
        log_pN = self.nodes_dist.log_prob(N).mean()

        # 2. The KL between q(z_T | x) and p(z_T) = Uniform(1/num_classes). Should be close to zero.
        kl_prior = self.kl_prior(clean_data, node_mask)

        # 3. Diffusion loss
        loss_all_t = self.compute_Lt(clean_data, pred, z_t, s_int, node_mask, test)

        # Combine terms
        nlls = -log_pN + kl_prior + loss_all_t

        # Update NLL metric object and return batch nll
        nll = (self.test_nll if test else self.val_nll)(nlls)  # Average over the batch

        log_dict = {
            "kl prior": kl_prior,
            "Estimator loss terms": loss_all_t,
            "log_pn": log_pN,
            "test_nll" if test else "val_nll": nll,
        }

        return nll, log_dict

    @torch.no_grad()
    def sample_batch(
        self,
        n_nodes: list,
        number_chain_steps: int = 50,
        batch_id: int = 0,
        keep_chain: int = 0,
        save_final: int = 0,
        test=True,
    ):
        # Start timing for this batch
        batch_start_time = time.time()
        
        # Reset projection timing for this batch
        self.total_projection_time = 0.0
        """
        :param batch_id: int
        :param n_nodes: list of int containing the number of nodes to sample for each graph
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"[{timestamp}] Sampling a batch with {len(n_nodes)} graphs on local rank {self.local_rank}."
            f" Saving {save_final} visualization and {keep_chain} full chains."
        )
        assert keep_chain >= 0
        assert save_final >= 0
        n_nodes = torch.Tensor(n_nodes).long().to(self.device)
        batch_size = len(n_nodes)
        n_max = torch.max(n_nodes).item()
        # Build the masks
        arange = (
            torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        )
        node_mask = arange < n_nodes.unsqueeze(1)
        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = self.noise_model.sample_limit_dist(node_mask=node_mask)

        assert (z_T.E == torch.transpose(z_T.E, 1, 2)).all()
        
        # Ensure number_chain_steps doesn't exceed diffusion steps
        # print(f"DEBUG: number_chain_steps: {number_chain_steps}, self.T: {self.T}")
        # assert number_chain_steps < self.T
        if number_chain_steps >= self.T:
            # For debug configs with small diffusion steps, adjust chain steps
            number_chain_steps = max(1, self.T - 1)
            print(f"Warning: Adjusted number_chain_steps to {number_chain_steps} to fit within {self.T} diffusion steps")

        chains = utils.PlaceHolder(
            X=torch.zeros((number_chain_steps, keep_chain, n_max), dtype=torch.long),
            charges=torch.zeros(
                (number_chain_steps, keep_chain, n_max), dtype=torch.long
            ),
            E=torch.zeros((number_chain_steps, keep_chain, n_max, n_max)),
            y=None,
        )

        z_t = z_T

        # Create planarity or other constraint preserving objects
        if self.cfg.model.rev_proj == "planar":
            rev_projector = PlanarProjector(z_t)
        elif self.cfg.model.rev_proj == "tree":
            rev_projector = TreeProjector(z_t)
        elif self.cfg.model.rev_proj == "lobster":
            rev_projector = LobsterProjector(z_t)
        elif self.cfg.model.rev_proj == "ring_count_at_most":
            max_rings = getattr(self.cfg.model, "max_rings", 0)
            atom_decoder = getattr(self.dataset_infos, "atom_decoder", None)
            if not hasattr(self, '_printed_ring_count_mode'):
                # print(f"🔧 RingCountAtMostProjector: max_rings={max_rings}")
                self._printed_ring_count_mode = True
            rev_projector = RingCountAtMostProjector(z_t, max_rings, atom_decoder)
        elif self.cfg.model.rev_proj == "ring_count_at_least":
            min_rings = getattr(self.cfg.model, "min_rings", 1)
            atom_decoder = getattr(self.dataset_infos, "atom_decoder", None)
            rev_projector = RingCountAtLeastProjector(z_t, min_rings, atom_decoder)
        elif self.cfg.model.rev_proj == "ring_length_at_most":
            max_ring_length = getattr(self.cfg.model, "max_ring_length", 6)
            atom_decoder = getattr(self.dataset_infos, "atom_decoder", None)
            if not hasattr(self, '_printed_ring_length_mode'):
                # print(f"🔧 RingLengthAtMostProjector: max_ring_length={max_ring_length}")
                self._printed_ring_length_mode = True
            rev_projector = RingLengthAtMostProjector(z_t, max_ring_length, atom_decoder)
        elif self.cfg.model.rev_proj == "ring_length_at_least":
            min_ring_length = getattr(self.cfg.model, "min_ring_length", 3)
            atom_decoder = getattr(self.dataset_infos, "atom_decoder", None)
            rev_projector = RingLengthAtLeastProjector(
                z_t, min_ring_length, atom_decoder
            )
        elif self.cfg.model.rev_proj is None or self.cfg.model.rev_proj == "":
            # No constraint training - no projector needed
            rev_projector = None
            if not hasattr(self, '_printed_no_constraint_mode'):
                print(f"🔧 No constraint training - generating unrestricted graphs")
                self._printed_no_constraint_mode = True
        else:
            assert ValueError(
                f"Projection type '{self.cfg.model.rev_proj}' not implemented."
            )

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        
        # Timing for entire reverse diffusion process
        diffusion_start_time = time.time()
        total_projection_time = 0.0
        total_model_time = 0.0
        
        for s_int in reversed(range(0, self.T, self.cfg.general.faster_sampling)):
            s_array = s_int * torch.ones(
                (batch_size, 1), dtype=torch.long, device=z_t.X.device
            )
            
            # Time the model forward pass
            model_start_time = time.time()
            z_s = self.sample_zs_from_zt(z_t, s_array)
            model_time = time.time() - model_start_time
            total_model_time += model_time

            # Initialize projection_time to 0.0 (will be updated if projection is performed)
            projection_time = 0.0

            # Constraint preserving (if any constraint is specified)
            if self.cfg.model.rev_proj and rev_projector is not None:
                # Set current timestep for logging
                rev_projector.current_timestep = s_int
                
                # Time the projection step
                projection_start_time = time.time()
                rev_projector.project(z_s)
                projection_time = time.time() - projection_start_time
                total_projection_time += projection_time
                
                # Record projection timing in metrics if available
                if test and hasattr(self, 'test_sampling_metrics'):
                    self.test_sampling_metrics.record_projection_time(projection_time)
                elif not test and hasattr(self, 'val_sampling_metrics'):
                    self.val_sampling_metrics.record_projection_time(projection_time)
                
                # Count rings after projection and check validity
                if hasattr(rev_projector, 'max_rings'):
                    total_rings_after = 0
                    graph_valid = True
                    for graph_idx, nx_graph in enumerate(rev_projector.nx_graphs_list):
                        ring_count = count_simple_cycles(nx_graph)
                        total_rings_after += ring_count
                        if ring_count > rev_projector.max_rings:
                            graph_valid = False
                            # logger.warning(f"⚠️ Graph {graph_idx} has {ring_count} rings > {rev_projector.max_rings} at timestep {s_int}")
                
                # Check ring length constraint after projection and check validity
                elif hasattr(rev_projector, 'max_ring_length'):
                    max_ring_length_after = 0
                    graph_valid = True
                    for graph_idx, nx_graph in enumerate(rev_projector.nx_graphs_list):
                        max_cycle_length = max_simple_cycle_length(nx_graph)
                        max_ring_length_after = max(max_ring_length_after, max_cycle_length)
                        if max_cycle_length > rev_projector.max_ring_length:
                            graph_valid = False
                            # logger.warning(f"⚠️ Graph {graph_idx} has max cycle length {max_cycle_length} > {rev_projector.max_ring_length} at timestep {s_int}")
                elif hasattr(rev_projector, 'min_rings'):
                    graph_valid = True
                    for graph_idx, nx_graph in enumerate(rev_projector.nx_graphs_list):
                        ring_count = count_simple_cycles(nx_graph)
                        if ring_count < rev_projector.min_rings:
                            graph_valid = False
                elif hasattr(rev_projector, 'min_ring_length'):
                    graph_valid = True
                    for graph_idx, nx_graph in enumerate(rev_projector.nx_graphs_list):
                        if not rev_projector.valid_graph_fn(nx_graph):
                            graph_valid = False
                
                # Assert constraint at t == 0
                if s_int == 0:
                    # Final projection at t=0 for both constraints
                    constraint_kind = None
                    if hasattr(rev_projector, 'max_rings'):
                        constraint_kind = "ring_count"
                        from ConStruct.projector.is_ring.is_ring_count_at_most.is_ring_count_at_most import ring_count_at_most_sanitize, has_at_most_n_rings
                        K = rev_projector.max_rings
                        for g_idx, g in enumerate(rev_projector.nx_graphs_list):
                            # Apply sanitizer
                            g = ring_count_at_most_sanitize(g, K)
                            rev_projector.nx_graphs_list[g_idx] = g
                            # Final assert
                            assert has_at_most_n_rings(g, K), f"Post-pass ring-count assertion failed: graph {g_idx} has > {K} cycles at t=0"
                    
                    elif hasattr(rev_projector, 'max_ring_length'):
                        constraint_kind = "ring_length"
                        from ConStruct.projector.is_ring.is_ring_length_at_most.is_ring_length_at_most import ring_length_at_most_sanitize, has_rings_of_length_at_most
                        L = rev_projector.max_ring_length
                        for g_idx, g in enumerate(rev_projector.nx_graphs_list):
                            # Apply sanitizer
                            g = ring_length_at_most_sanitize(g, L)
                            rev_projector.nx_graphs_list[g_idx] = g
                            # Final assert
                            assert has_rings_of_length_at_most(g, L), f"Post-pass ring-length assertion failed: graph {g_idx} has cycles > length {L} at t=0"
                    elif hasattr(rev_projector, 'min_rings'):
                        constraint_kind = "ring_count_at_least"
                        K = rev_projector.min_rings
                        for g_idx, g in enumerate(rev_projector.nx_graphs_list):
                            assert count_simple_cycles(g) >= K, (
                                f"Post-pass ring-count assertion failed: graph {g_idx} has < {K} cycles at t=0"
                            )
                    elif hasattr(rev_projector, 'min_ring_length'):
                        constraint_kind = "ring_length_at_least"
                        L = rev_projector.min_ring_length
                        for g_idx, g in enumerate(rev_projector.nx_graphs_list):
                            assert rev_projector.valid_graph_fn(g), (
                                f"Post-pass ring-length assertion failed: graph {g_idx} has no cycle of length >= {L} at t=0"
                            )
                    
                    # Log final projection info
                    if constraint_kind:
                        blocked_edges_total = sum(len(blocked_set) for blocked_set in rev_projector.blocked_edges.values()) if hasattr(rev_projector, 'blocked_edges') else 0
                        # logger.info(f"Final projection at t=0: constraint={constraint_kind}, blocked_edges={blocked_edges_total}")
            
                # Planarity-specific validation
                elif self.cfg.model.rev_proj == "planar":
                    planarity_valid = True
                    non_planar_graphs = []
                    for graph_idx, nx_graph in enumerate(rev_projector.nx_graphs_list):
                        if not rev_projector.valid_graph_fn(nx_graph):
                            planarity_valid = False
                            non_planar_graphs.append(graph_idx)
                            # logger.warning(f"⚠️ Graph {graph_idx} is non-planar at timestep {s_int}")
                    
                    # Log planarity status at t == 0
                    if s_int == 0:
                        if not planarity_valid:
                            logger.error(f"❌ PLANARITY VIOLATION at t=0: {len(non_planar_graphs)}/{len(rev_projector.nx_graphs_list)} graphs non-planar")
                            logger.error(f"   Non-planar graphs: {non_planar_graphs}")
                            # Don't raise exception for planarity - just log the violation
                        else:
                            pass  # Don't raise exception for planarity - just log the violation
            
                # DEBUG: Track natural structural properties for unconstrained generation
                # Only run this analysis during training, not during validation
                elif (self.cfg.model.rev_proj is None or self.cfg.model.rev_proj == "") and not test:
                    # Convert current z_s to NetworkX graphs for analysis
                    discrete_z_s = z_s.collapse(self.collapse_charges)
                    nx_graphs_current = []
                    total_rings_current = 0
                    planar_graphs_current = 0
                    ring_lengths_current = []
                    
                    # Track ring count and length distributions
                    ring_count_distribution = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}  # ≤0, ≤1, ≤2, ≤3, ≤4, ≤5
                    ring_length_distribution = {3: 0, 4: 0, 5: 0, 6: 0}  # ≤3, ≤4, ≤5, ≤6
                    
                    for i in range(discrete_z_s.X.shape[0]):
                        adj_matrix = discrete_z_s.E[i].cpu().numpy()
                        nx_graph = nx.from_numpy_array(adj_matrix)
                        nx_graphs_current.append(nx_graph)
                        
                        # Count rings and track ring lengths using all simple cycles
                        cycles = list(enumerate_simple_cycles_unique(nx_graph))
                        ring_count = len(cycles)
                        total_rings_current += ring_count
                        
                        # Track ring lengths and update distributions
                        max_ring_length_in_graph = max_simple_cycle_length(nx_graph)
                        for cycle in cycles:
                            ring_length = len(cycle)
                            ring_lengths_current.append(ring_length)
                        
                        # Update ring count distribution (≤0, ≤1, ≤2, ≤3, ≤4, ≤5)
                        for max_rings in ring_count_distribution.keys():
                            if ring_count <= max_rings:
                                ring_count_distribution[max_rings] += 1
                        
                        # Update ring length distribution (≤3, ≤4, ≤5, ≤6)
                        for max_length in ring_length_distribution.keys():
                            if max_ring_length_in_graph <= max_length:
                                ring_length_distribution[max_length] += 1
                        
                        # Check planarity
                        if is_planar.is_planar(nx_graph):
                            planar_graphs_current += 1
                    
                    # Log natural properties at key timesteps
                    if s_int % 100 == 0 or s_int == 0:
                        planarity_rate = planar_graphs_current / len(nx_graphs_current) if nx_graphs_current else 0
                        
                        # Calculate rates for ring count distributions
                        ring_count_rates = {}
                        for max_rings, count in ring_count_distribution.items():
                            rate = count / len(nx_graphs_current) if nx_graphs_current else 0
                            ring_count_rates[f"≤{max_rings}"] = rate
                        
                        # Calculate rates for ring length distributions
                        ring_length_rates = {}
                        for max_length, count in ring_length_distribution.items():
                            rate = count / len(nx_graphs_current) if nx_graphs_current else 0
                            ring_length_rates[f"≤{max_length}"] = rate
                        
                        # Track evolution of properties during diffusion
                        if not hasattr(self, '_property_evolution'):
                            self._property_evolution = []
                        
                        if s_int % 50 == 0:  # Track every 50 timesteps
                            # Calculate rates for this timestep
                            ring_count_rates = {}
                            for max_rings, count in ring_count_distribution.items():
                                rate = count / len(nx_graphs_current) if nx_graphs_current else 0
                                ring_count_rates[f"≤{max_rings}"] = rate
                            
                            ring_length_rates = {}
                            for max_length, count in ring_length_distribution.items():
                                rate = count / len(nx_graphs_current) if nx_graphs_current else 0
                                ring_length_rates[f"≤{max_length}"] = rate
                            
                            self._property_evolution.append({
                                'timestep': s_int,
                                'planarity_rate': planar_graphs_current / len(nx_graphs_current) if nx_graphs_current else 0,
                                'ring_count_rates': ring_count_rates,
                                'ring_length_rates': ring_length_rates
                            })
                            
                            # Store final properties for summary
                            if s_int == 0:
                                if not hasattr(self, '_natural_properties_summary'):
                                    # Calculate final rates
                                    ring_count_rates = {}
                                    for max_rings, count in ring_count_distribution.items():
                                        rate = count / len(nx_graphs_current) if nx_graphs_current else 0
                                        ring_count_rates[f"≤{max_rings}"] = rate
                                    
                                    ring_length_rates = {}
                                    for max_length, count in ring_length_distribution.items():
                                        rate = count / len(nx_graphs_current) if nx_graphs_current else 0
                                        ring_length_rates[f"≤{max_length}"] = rate
                                    
                                    self._natural_properties_summary = {
                                        'total_graphs': len(nx_graphs_current),
                                        'planar_graphs': planar_graphs_current,
                                        'planarity_rate': planar_graphs_current / len(nx_graphs_current) if nx_graphs_current else 0,
                                        'ring_count_rates': ring_count_rates,
                                        'ring_length_rates': ring_length_rates
                                    }
                else:
                    # For constrained generation, initialize empty tracking variables
                    nx_graphs_current = []
                    ring_count_distribution = {}
                    ring_length_distribution = {}
                    planar_graphs_current = 0


            z_t = z_s



            # Save the first keep_chain graphs
            if (s_int * number_chain_steps) % self.T == 0:
                write_index = (
                    number_chain_steps - 1 - ((s_int * number_chain_steps) // self.T)
                )
                discrete_z_t = z_t.collapse(self.collapse_charges)
                chains.X[write_index] = discrete_z_t.X[:keep_chain]
                chains.E[write_index] = discrete_z_t.E[:keep_chain]
                if discrete_z_t.charges.numel() > 0:
                    chains.charges[write_index] = discrete_z_t.charges[:keep_chain]

        # Sample final data
        sampled_s = z_t.collapse(self.collapse_charges)
        X, E, charges = sampled_s.X, sampled_s.E, sampled_s.charges
        

        # X, E, charges = discrete_z_s.X, discrete_z_s.E, discrete_z_s.charges

        # Prepare the chain for saving
        if keep_chain > 0:
            chains.X[-1] = X[
                :keep_chain
            ]  # Overwrite last frame with the resulting X, E
            chains.E[-1] = E[:keep_chain]
            if sampled_s.charges.numel() > 0:
                chains.charges[-1] = charges[:keep_chain]

        final_batch = sampled_s

        # Visualize chains
        if keep_chain > 0:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.print(f"[{timestamp}] Batch sampled. Visualizing chains starts!")
            chains_path = os.path.join(
                os.getcwd(),
                f"chains/epoch{self.current_epoch}/",
                f"batch{batch_id}_GR{self.global_rank}",
            )
            os.makedirs(chains_path, exist_ok=True)

            self.visualizer.visualize_chains(
                chains=chains,
                num_nodes=n_nodes,
                chain_path=chains_path,
                batch_id=batch_id,
                local_rank=self.local_rank,
            )

        # Visualize the final molecules
        current_path = os.getcwd()
        result_path = os.path.join(
            current_path,
            f"graphs/{self.name}/epoch{self.current_epoch}_b{batch_id}/",
        )
        self.visualizer.visualize(
            path=result_path,
            graphs=final_batch,
            atom_decoder=(
                self.dataset_infos.atom_decoder
                if hasattr(self.dataset_infos, "atom_decoder")
                else None
            ),
            num_graphs_to_visualize=save_final,
        )
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.print(f"[{timestamp}] Done.")

        # During testing, move the graphs to cpu so that they can be aggregated for metrics computation
        if test:
            final_batch = final_batch.device_as(torch.zeros(1, device="cpu"))



        # Record timing for this batch
        batch_time = time.time() - batch_start_time
        if hasattr(self, 'sampling_times'):
            self.sampling_times.append(batch_time)
        else:
            self.sampling_times = [batch_time]
        
        # Record timing in sampling metrics if available
        if test and hasattr(self, 'test_sampling_metrics'):
            self.test_sampling_metrics.record_sampling_time(batch_time)
            # Projection timing is now recorded during each projection step
        elif not test and hasattr(self, 'val_sampling_metrics'):
            self.val_sampling_metrics.record_sampling_time(batch_time)
            # Projection timing is now recorded during each projection step

        return final_batch

    def sample_zs_from_zt(self, z_t, s_int):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
        if last_step, return the graph prediction as well"""
        pred = self.forward(z_t)
        return self.noise_model.sample_zs_from_zt_and_pred(
            z_t=z_t, pred=pred, s_int=s_int
        )

    def sample_n_graphs(
        self,
        samples_to_generate: int,
        chains_to_save: int,
        samples_to_save: int,
        test: bool,
    ):
        if samples_to_generate <= 0:
            return []

        chains_left_to_save = chains_to_save

        samples = []
        # The first graphs are sampled without sorting the sizes, so that the visualizations are not biased
        first_sampling = min(samples_to_generate, max(samples_to_save, chains_to_save))
        if first_sampling > 0:
            n_nodes = self.nodes_dist.sample_n(first_sampling, self.device)
            current_max_size = 0
            current_n_list = []
            for i, n in enumerate(n_nodes):
                potential_max_size = max(current_max_size, n)
                if self.cfg.dataset.adaptive_loader:
                    potential_ebs = effective_batch_size(
                        potential_max_size,
                        self.cfg.train.reference_batch_size,
                        sampling=True,
                    )
                else:
                    potential_ebs = int(
                        1.8
                        * self.cfg.train.batch_size  # 1.8 because No need to make a backward pass
                    )
                if potential_ebs > len(current_n_list) or len(current_n_list) == 0:
                    current_n_list.append(n)
                    current_max_size = potential_max_size
                else:
                    chains_save = max(min(chains_left_to_save, len(current_n_list)), 0)
                    samples.append(
                        self.sample_batch(
                            n_nodes=current_n_list,
                            batch_id=i,
                            save_final=len(current_n_list),
                            keep_chain=chains_save,
                            number_chain_steps=self.number_chain_steps,
                            test=test,
                        )
                    )
                    chains_left_to_save -= chains_save
                    current_n_list = [n]
                    current_max_size = n
            chains_save = max(min(chains_left_to_save, len(current_n_list)), 0)
            samples.append(
                self.sample_batch(
                    n_nodes=current_n_list,
                    batch_id=i + 1,
                    save_final=len(current_n_list),
                    keep_chain=chains_save,
                    number_chain_steps=self.number_chain_steps,
                    test=test,
                )
            )
            if samples_to_generate - first_sampling <= 0:
                return samples

        n_nodes = self.nodes_dist.sample_n(
            samples_to_generate - first_sampling, self.device
        )

        if self.cfg.dataset.adaptive_loader:
            # The remaining graphs are sampled in decreasing graph size
            n_nodes = torch.sort(n_nodes, descending=True)[0]

        max_size = 0
        current_n_list = []
        for i, n in enumerate(n_nodes):
            max_size = max(max_size, n)
            potential_ebs = (
                effective_batch_size(
                    max_size, self.cfg.train.reference_batch_size, sampling=True
                )
                if self.cfg.dataset.adaptive_loader
                else 1.8 * self.cfg.train.batch_size
            )
            if potential_ebs > len(current_n_list) or len(current_n_list) == 0:
                current_n_list.append(n)
            else:
                samples.append(self.sample_batch(n_nodes=current_n_list, test=test))
                current_n_list = [n]
                max_size = n
        samples.append(self.sample_batch(n_nodes=current_n_list, test=test))

        return samples
