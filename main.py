try:
    import graph_tool as gt
except ModuleNotFoundError:
    print("Graph tool not found, non molecular datasets cannot be used")
import os
import pathlib
import warnings

import hydra
import pytorch_lightning as pl
import torch

from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from lightning_fabric.utilities.warnings import PossibleUserWarning

from ConStruct import utils
from ConStruct.diffusion_model_discrete import DiscreteDenoisingDiffusion
from ConStruct.baseline import BaselineModel
from ConStruct.metrics.sampling_metrics import SamplingMetrics

torch.cuda.empty_cache()
warnings.filterwarnings("ignore", category=PossibleUserWarning)


def validate_sampling_only_config(cfg: DictConfig) -> None:
    """Fail fast on ambiguous or unsupported sampling-only configurations."""
    if not getattr(cfg.general, "sampling_only", False):
        return
    if not cfg.general.test_only:
        raise ValueError(
            "general.sampling_only=true requires general.test_only to point to a checkpoint."
        )
    if cfg.general.resume:
        raise ValueError(
            "general.resume cannot be used with general.sampling_only=true; "
            "use general.test_only for the checkpoint."
        )
    if cfg.general.evaluate_all_checkpoints:
        raise ValueError(
            "general.evaluate_all_checkpoints=true is incompatible with sampling-only mode; "
            "run one checkpoint per invocation."
        )
    if hasattr(cfg.model, "is_baseline"):
        raise ValueError(
            "Sampling-only checkpoint mode is not supported for baseline models."
        )


def validate_early_stopping_config(cfg: DictConfig) -> None:
    """Ensure a sampling metric exists at every early-stopping check."""
    if (
        cfg.general.test_only
        or not cfg.train.early_stopping.enable
        or not str(cfg.train.early_stopping.monitor).startswith("val_sampling/")
    ):
        return
    if cfg.general.sample_every_val != 1:
        raise ValueError(
            "Early stopping on a val_sampling/* metric requires "
            "general.sample_every_val=1 so the monitored metric is available "
            "at every validation check."
        )
    if cfg.general.samples_to_generate <= 0:
        raise ValueError(
            "Early stopping on a val_sampling/* metric requires a positive "
            "general.samples_to_generate."
        )


@hydra.main(version_base="1.3", config_path="./configs", config_name="config")
def main(cfg: DictConfig):
    validate_sampling_only_config(cfg)
    validate_early_stopping_config(cfg)
    pl.seed_everything(cfg.train.seed)
    dataset_config = cfg.dataset

    if dataset_config.name in [
        "sbm",
        "comm-20",
        "planar",
        "ego",
        "grid",
        "enzymes",
        "lobster",
        "tree",
    ]:
        from ConStruct.datasets.spectre_dataset import (
            SpectreGraphDataModule,
            SpectreDatasetInfos,
        )

        datamodule = SpectreGraphDataModule(cfg)
        dataset_infos = SpectreDatasetInfos(datamodule)

    elif dataset_config["name"] == "protein":
        from ConStruct.datasets import protein_dataset

        datamodule = protein_dataset.ProteinDataModule(cfg)
        dataset_infos = protein_dataset.ProteinInfos(datamodule=datamodule)

    elif dataset_config.name in ["qm9", "qm9_filtered"]:
        from ConStruct.datasets.qm9_dataset import QM9DataModule, QM9Infos

        datamodule = QM9DataModule(cfg)
        dataset_infos = QM9Infos(datamodule, cfg)
    elif dataset_config.name == "guacamol":
        from ConStruct.datasets.guacamol_dataset import GuacamolDataModule, GuacamolInfos

        datamodule = GuacamolDataModule(cfg)
        dataset_infos = GuacamolInfos(datamodule, cfg)
    elif dataset_config.name == "moses":
        from ConStruct.datasets.moses_dataset import MosesDataModule, MosesInfos

        datamodule = MosesDataModule(cfg)
        dataset_infos = MosesInfos(datamodule, cfg)
    elif dataset_config.name in ["low_tls", "high_tls"]:
        from ConStruct.datasets.tls_dataset import TLSDataModule, TLSInfos

        datamodule = TLSDataModule(cfg)
        dataset_infos = TLSInfos(datamodule)
    else:
        raise NotImplementedError("Unknown dataset {}".format(cfg.dataset))

    val_sampling_metrics = SamplingMetrics(
        dataset_infos,
        test=False,
        train_loader=datamodule.train_dataloader(),
        val_loader=datamodule.val_dataloader(),
        cfg=cfg,
    )
    test_sampling_metrics = SamplingMetrics(
        dataset_infos,
        test=True,
        train_loader=datamodule.train_dataloader(),
        val_loader=datamodule.test_dataloader(),
        cfg=cfg,
    )

    if not hasattr(cfg.model, "is_baseline"):
        model = DiscreteDenoisingDiffusion(
            cfg=cfg,
            dataset_infos=dataset_infos,
            val_sampling_metrics=val_sampling_metrics,
            test_sampling_metrics=test_sampling_metrics,
        )

        # need to ignore metrics because otherwise ddp tries to sync them
        params_to_ignore = [
            "module.model.dataset_infos",
            "module.model.val_sampling_metrics",
            "module.model.test_sampling_metrics",
        ]
        torch.nn.parallel.DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(
            model, params_to_ignore
        )

        callbacks = []
        if cfg.train.save_model:
            checkpoint_monitor = (
                cfg.train.early_stopping.monitor
                if cfg.train.early_stopping.enable
                else "val/epoch_NLL"
            )
            checkpoint_mode = (
                cfg.train.early_stopping.mode
                if cfg.train.early_stopping.enable
                else "min"
            )
            checkpoint_callback = ModelCheckpoint(
                dirpath=f"checkpoints/{cfg.general.name}",
                filename="{epoch}",
                monitor=checkpoint_monitor,
                save_top_k=5,
                mode=checkpoint_mode,
                every_n_epochs=1,
            )
            last_ckpt_save = ModelCheckpoint(
                dirpath=f"checkpoints/{cfg.general.name}",
                filename="last",
                every_n_epochs=1,
            )
            callbacks.append(last_ckpt_save)
            callbacks.append(checkpoint_callback)

        # Add early stopping callback if enabled
        if cfg.train.early_stopping.enable and not cfg.general.name == "debug":
            early_stopping_callback = EarlyStopping(
                monitor=cfg.train.early_stopping.monitor,
                patience=cfg.train.early_stopping.patience,
                min_delta=cfg.train.early_stopping.min_delta,
                mode=cfg.train.early_stopping.mode,
                check_finite=cfg.train.early_stopping.check_finite,
                stopping_threshold=cfg.train.early_stopping.stopping_threshold,
                divergence_threshold=cfg.train.early_stopping.divergence_threshold,
                check_on_train_epoch_end=cfg.train.early_stopping.check_on_train_epoch_end,
                verbose=True  # Print early stopping messages
            )
            callbacks.append(early_stopping_callback)
            print(f"[INFO] Early stopping enabled: monitoring '{cfg.train.early_stopping.monitor}' with patience {cfg.train.early_stopping.patience}")

        is_debug_run = cfg.general.name == "debug"
        if is_debug_run:
            print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run")
            print(
                "[WARNING]: Run is called 'debug' -- it will run with only 1 CPU core"
            )
        use_gpu = torch.cuda.is_available() and not is_debug_run
        trainer = Trainer(
            gradient_clip_val=cfg.train.clip_grad,
            strategy="ddp",
            accelerator="gpu" if use_gpu else "cpu",
            devices=-1 if use_gpu else 1,
            max_epochs=cfg.train.n_epochs,
            min_epochs=(
                cfg.train.early_stopping.min_epochs
                if cfg.train.early_stopping.enable
                else None
            ),
            check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
            num_sanity_val_steps=0,
            fast_dev_run=is_debug_run,
            enable_progress_bar=False,
            callbacks=callbacks,
            log_every_n_steps=1 if is_debug_run else 50,
            logger=[],
            limit_test_batches=(
                1 if getattr(cfg.general, "sampling_only", False) else 1.0
            ),
        )

        if getattr(cfg.general, "sampling_only", False):
            trainer.test(
                model, datamodule=datamodule, ckpt_path=cfg.general.test_only
            )
        elif not cfg.general.test_only:
            trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
            # Test the checkpoint selected by the configured validation monitor,
            # not the final in-memory weights. Runs without checkpointing retain
            # old behavior.
            test_ckpt_path = "best" if cfg.train.save_model else None
            trainer.test(model, datamodule=datamodule, ckpt_path=test_ckpt_path)
        else:
            # Start by evaluating test_only_path
            for i in range(5):
                new_seed = i * 1000
                pl.seed_everything(new_seed)
                cfg.train.seed = new_seed
                trainer.test(
                    model, datamodule=datamodule, ckpt_path=cfg.general.test_only
                )
            if cfg.general.evaluate_all_checkpoints:
                directory = pathlib.Path(cfg.general.test_only).parents[0]
                print("Directory:", directory)
                files_list = os.listdir(directory)
                for file in files_list:
                    if ".ckpt" in file:
                        ckpt_path = os.path.join(directory, file)
                        if ckpt_path == cfg.general.test_only:
                            continue
                        print("Loading checkpoint", ckpt_path)
                        trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)
    else:
        model = BaselineModel(
            cfg=cfg,
            dataset_infos=dataset_infos,
            val_sampling_metrics=val_sampling_metrics,
            test_sampling_metrics=test_sampling_metrics,
        )

        model.train(datamodule=datamodule)
        for i in range(5):
            new_seed = i * 1000
            pl.seed_everything(new_seed)
            cfg.train.seed = new_seed
            model.test()


if __name__ == "__main__":
    main()
