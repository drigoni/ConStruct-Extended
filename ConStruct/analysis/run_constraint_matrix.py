"""Generate and cross-evaluate the predefined QM9 projection matrix.

This command treats generated sample pickles as immutable source artifacts. It
can be rerun after interruption: completed generation cells and metric caches
with matching provenance are reused, while a failed cell is retried before a
structured error record is written.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ConStruct.metrics.sampling_molecular_metrics import (
    Molecule,
    compute_valid_molecules,
)
from ConStruct.projector.graph_cycles import enumerate_simple_cycles_unique
from ConStruct.projector.projector_utils import build_simple_graph_from_edge_tensor
from ConStruct.utils import PlaceHolder


SCHEMA_VERSION = 1
COUNT_LEVELS = (None, 1, 2)
LENGTH_LEVELS = (None, 4, 5)
QM9_NO_H_ATOM_DECODER = ["C", "N", "O", "F"]


@dataclass(frozen=True)
class MatrixProfile:
    name: str
    min_rings: int | None
    min_ring_length: int | None

    @property
    def constraints(self) -> list[dict[str, Any]]:
        constraints = []
        if self.min_rings is not None:
            constraints.append(
                {"type": "ring_count_at_least", "min_rings": self.min_rings}
            )
        if self.min_ring_length is not None:
            constraints.append(
                {
                    "type": "ring_length_at_least",
                    "min_ring_length": self.min_ring_length,
                }
            )
        return constraints

    @property
    def projection_enabled(self) -> bool:
        return bool(self.constraints)


PROFILES = tuple(
    MatrixProfile(
        name=(
            f"count_{'disabled' if count is None else count}_"
            f"length_{'disabled' if length is None else length}"
        ),
        min_rings=count,
        min_ring_length=length,
    )
    for count in COUNT_LEVELS
    for length in LENGTH_LEVELS
)
PROFILE_BY_NAME = {profile.name: profile for profile in PROFILES}

METRICS = (
    ("molecular_validity_pct", "Molecular validity"),
    ("ring_count_at_least_1_pct", "Ring count ≥ 1"),
    ("ring_count_at_least_2_pct", "Ring count ≥ 2"),
    ("ring_length_at_least_4_pct", "Maximum cycle length ≥ 4"),
    ("ring_length_at_least_5_pct", "Maximum cycle length ≥ 5"),
)


class ArtifactMismatchError(RuntimeError):
    """A completed artifact belongs to a different matrix request."""


def _utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _rank_number(path: Path) -> int:
    match = re.search(r"rank(\d+)", path.name)
    return int(match.group(1)) if match else sys.maxsize


def _rank_files(sample_dir: Path) -> list[Path]:
    return sorted(sample_dir.glob("generated_samples_rank*.pkl"), key=_rank_number)


def _source_fingerprint(sample_dir: Path) -> list[dict[str, Any]]:
    fingerprints = []
    for path in _rank_files(sample_dir):
        stat = path.stat()
        fingerprints.append(
            {"name": path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        )
    metrics_path = sample_dir / "sampling_metrics.json"
    stat = metrics_path.stat()
    fingerprints.append(
        {
            "name": metrics_path.name,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    )
    return fingerprints


def _normalized_checkpoint(value: str | Path) -> str:
    return str(Path(value).expanduser().resolve())


def validate_generation_artifacts(
    sample_dir: Path,
    profile: MatrixProfile,
    checkpoint: Path,
    seed: int,
    requested_samples: int,
) -> dict[str, Any] | None:
    """Return metadata for a complete matching cell, or None if incomplete."""
    metrics_path = sample_dir / "sampling_metrics.json"
    if not metrics_path.exists():
        return None

    payload = _read_json(metrics_path)
    metadata = payload.get("metadata", {})
    expected = {
        "checkpoint": _normalized_checkpoint(checkpoint),
        "seed": seed,
        "requested_samples": requested_samples,
        "constraints": profile.constraints,
        "projection_enabled": profile.projection_enabled,
    }
    actual = {
        "checkpoint": _normalized_checkpoint(metadata.get("checkpoint", "")),
        "seed": metadata.get("seed"),
        "requested_samples": metadata.get("requested_samples"),
        "constraints": metadata.get("constraints", []),
        "projection_enabled": metadata.get("projection_enabled"),
    }
    if actual != expected:
        raise ArtifactMismatchError(
            f"Completed artifacts in {sample_dir} do not match this request. "
            f"Expected {expected}, found {actual}."
        )
    if metadata.get("dataset") != "qm9":
        raise ArtifactMismatchError(
            f"Matrix evaluation supports full QM9 artifacts, got "
            f"dataset={metadata.get('dataset')!r} in {metrics_path}."
        )
    if int(metadata.get("generated_samples", 0)) < requested_samples:
        return None

    rank_files = _rank_files(sample_dir)
    expected_ranks = int(metadata.get("device_count", 1))
    if len(rank_files) != expected_ranks:
        return None
    return metadata


def _hydra_string_override(key: str, value: str | Path) -> str:
    escaped = str(value).replace("\\", "\\\\").replace("'", "\\'")
    return f"{key}='{escaped}'"


def generation_command(
    repo_root: Path,
    experiment: str,
    profile: MatrixProfile,
    checkpoint: Path,
    seed: int,
    requested_samples: int,
    sample_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        str(repo_root / "main.py"),
        f"+experiment={experiment}",
        f"+constraint={profile.name}",
        f"train.seed={seed}",
        _hydra_string_override("general.test_only", checkpoint),
        "general.sampling_only=true",
        "general.wandb=disabled",
        f"general.final_model_samples_to_generate={requested_samples}",
        "general.final_model_samples_to_save=0",
        "general.final_model_chains_to_save=0",
        _hydra_string_override("general.sampling_output_dir", sample_dir),
    ]


def _write_error(
    error_path: Path,
    stage: str,
    profile: MatrixProfile | None,
    attempts: int,
    error: BaseException,
    logs: Iterable[Path] = (),
    traceback_text: str | None = None,
) -> None:
    _atomic_write_json(
        error_path,
        {
            "schema_version": SCHEMA_VERSION,
            "timestamp": _utc_timestamp(),
            "stage": stage,
            "profile": profile.name if profile else None,
            "attempts": attempts,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback_text,
            "logs": [str(path) for path in logs],
        },
    )


def _clear_error(path: Path) -> None:
    if path.exists():
        path.unlink()


def _retry(
    operation: Callable[[int], Any],
    *,
    attempts: int,
    error_path: Path,
    stage: str,
    profile: MatrixProfile | None,
    logs: list[Path] | None = None,
) -> Any:
    last_error = None
    last_traceback = None
    attempts_completed = 0
    for attempt in range(1, attempts + 1):
        attempts_completed = attempt
        try:
            result = operation(attempt)
            _clear_error(error_path)
            return result
        except ArtifactMismatchError as error:
            last_error = error
            last_traceback = traceback.format_exc()
            break
        except Exception as error:  # retry boundary intentionally records all failures
            last_error = error
            last_traceback = traceback.format_exc()
            print(
                f"[{stage}] attempt {attempt}/{attempts} failed for "
                f"{profile.name if profile else 'aggregate'}: {error}",
                file=sys.stderr,
            )
    assert last_error is not None
    _write_error(
        error_path,
        stage,
        profile,
        attempts_completed,
        last_error,
        logs or (),
        last_traceback,
    )
    raise RuntimeError(
        f"{stage} failed after {attempts_completed} attempts; details: {error_path}"
    ) from last_error


def _archive_forced_cell(sample_dir: Path) -> Path | None:
    if not sample_dir.exists():
        return None
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    archive = sample_dir.with_name(f"{sample_dir.name}.replaced_{timestamp}")
    counter = 1
    while archive.exists():
        archive = sample_dir.with_name(
            f"{sample_dir.name}.replaced_{timestamp}_{counter}"
        )
        counter += 1
    shutil.move(str(sample_dir), str(archive))
    return archive


def generate_profile(
    *,
    repo_root: Path,
    output_root: Path,
    matrix_root: Path,
    experiment: str,
    profile: MatrixProfile,
    checkpoint: Path,
    seed: int,
    requested_samples: int,
    attempts: int,
    force: bool,
) -> dict[str, Any]:
    sample_dir = output_root / profile.name / f"seed_{seed}"
    error_path = matrix_root / "errors" / f"generation_{profile.name}.json"
    if force:
        archive = _archive_forced_cell(sample_dir)
        if archive is not None:
            print(f"Archived forced profile artifacts to {archive}")
    else:
        try:
            metadata = validate_generation_artifacts(
                sample_dir, profile, checkpoint, seed, requested_samples
            )
        except ArtifactMismatchError as error:
            _write_error(
                error_path,
                "generation",
                profile,
                1,
                error,
                traceback_text=traceback.format_exc(),
            )
            raise
        if metadata is not None:
            print(f"[generation] reusing complete profile {profile.name}")
            return metadata

    logs = [
        matrix_root / "logs" / f"generation_{profile.name}_attempt_{attempt}.log"
        for attempt in range(1, attempts + 1)
    ]
    def run(attempt: int):
        command = generation_command(
            repo_root,
            experiment,
            profile,
            checkpoint,
            seed,
            requested_samples,
            sample_dir,
        )
        log_path = logs[attempt - 1]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_handle:
            result = subprocess.run(
                command,
                cwd=repo_root,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"sampling subprocess exited with code {result.returncode}; "
                f"see {log_path}"
            )
        metadata = validate_generation_artifacts(
            sample_dir, profile, checkpoint, seed, requested_samples
        )
        if metadata is None:
            raise RuntimeError(
                f"sampling subprocess completed without a complete artifact set in "
                f"{sample_dir}"
            )
        return metadata

    return _retry(
        run,
        attempts=attempts,
        error_path=error_path,
        stage="generation",
        profile=profile,
        logs=logs,
    )


def _slice_batch(batch: PlaceHolder, count: int) -> PlaceHolder:
    def sliced(value):
        return value[:count].clone() if value is not None else None

    return PlaceHolder(
        X=sliced(batch.X),
        charges=sliced(batch.charges),
        E=sliced(batch.E),
        y=sliced(batch.y),
        t_int=sliced(batch.t_int),
        t=sliced(batch.t),
        node_mask=sliced(batch.node_mask),
    )


def load_trimmed_batches(sample_dir: Path, requested_samples: int) -> list[PlaceHolder]:
    """Load trusted local rank pickles and cap them to an exact denominator."""
    batches = []
    remaining = requested_samples
    for rank_file in _rank_files(sample_dir):
        with rank_file.open("rb") as handle:
            rank_batches = pickle.load(handle)
        if not isinstance(rank_batches, list):
            raise TypeError(f"Expected a list of batches in {rank_file}.")
        for batch in rank_batches:
            if not isinstance(batch, PlaceHolder) or batch.node_mask is None:
                raise TypeError(f"Unexpected graph batch in {rank_file}.")
            batch_size = int(batch.X.shape[0])
            take = min(batch_size, remaining)
            if take:
                batches.append(batch if take == batch_size else _slice_batch(batch, take))
                remaining -= take
            if remaining == 0:
                return batches
    raise ValueError(
        f"Only {requested_samples - remaining} graphs were available in {sample_dir}; "
        f"{requested_samples} were requested."
    )


def evaluate_batches(
    batches: list[PlaceHolder], atom_decoder: list[str], direction: str = "at_least"
) -> dict[str, Any]:
    """Compute the same five targets for a saved sample set in one pass."""
    if direction not in {"at_least", "at_most"}:
        raise ValueError(f"Unknown cycle metric direction: {direction}")
    molecules = []
    cycle_counts = []
    maximum_cycle_lengths = []
    for batch in batches:
        molecules.extend(
            Molecule(graph, atom_decoder=atom_decoder) for graph in batch.split()
        )
        for edge_matrix, node_mask in zip(batch.E, batch.node_mask):
            graph = build_simple_graph_from_edge_tensor(edge_matrix, node_mask)
            cycles = list(enumerate_simple_cycles_unique(graph))
            cycle_counts.append(len(cycles))
            maximum_cycle_lengths.append(
                max((len(cycle) for cycle in cycles), default=0)
            )

    total = len(cycle_counts)
    if total == 0 or len(molecules) != total:
        raise ValueError("Cannot evaluate an empty or inconsistent graph collection.")
    valid, _, errors = compute_valid_molecules(molecules)
    comparison = (lambda value, threshold: value >= threshold) if direction == "at_least" else (lambda value, threshold: value <= threshold)
    counts = {"molecular_validity": len(valid)}
    for threshold in (1, 2):
        counts[f"ring_count_{direction}_{threshold}"] = sum(
            comparison(value, threshold) for value in cycle_counts
        )
    for threshold in (4, 5):
        counts[f"ring_length_{direction}_{threshold}"] = sum(
            comparison(value, threshold) for value in maximum_cycle_lengths
        )
    metrics = {f"{key}_pct": 100.0 * value / total for key, value in counts.items()}
    return {
        "num_graphs": total,
        "counts": counts,
        "metrics": metrics,
        "molecular_errors": {str(key): value for key, value in errors.items()},
    }


def _cell_cache_matches(
    cell: dict[str, Any],
    profile: MatrixProfile,
    checkpoint: Path,
    seed: int,
    requested_samples: int,
    fingerprint: list[dict[str, Any]],
) -> bool:
    return (
        cell.get("schema_version") == SCHEMA_VERSION
        and cell.get("profile") == profile.name
        and cell.get("checkpoint") == _normalized_checkpoint(checkpoint)
        and cell.get("seed") == seed
        and cell.get("requested_samples") == requested_samples
        and cell.get("source_fingerprint") == fingerprint
        and set(cell.get("metrics", {})) == {key for key, _ in METRICS}
    )


def evaluate_profile(
    *,
    output_root: Path,
    matrix_root: Path,
    profile: MatrixProfile,
    checkpoint: Path,
    seed: int,
    requested_samples: int,
    attempts: int,
) -> dict[str, Any]:
    sample_dir = output_root / profile.name / f"seed_{seed}"
    error_path = matrix_root / "errors" / f"metrics_{profile.name}.json"
    try:
        metadata = validate_generation_artifacts(
            sample_dir, profile, checkpoint, seed, requested_samples
        )
    except ArtifactMismatchError as error:
        _write_error(
            error_path,
            "metrics",
            profile,
            1,
            error,
            traceback_text=traceback.format_exc(),
        )
        raise
    if metadata is None:
        error = RuntimeError(
            f"Generation artifacts are incomplete for {profile.name}."
        )
        _write_error(error_path, "metrics", profile, 1, error)
        raise error
    fingerprint = _source_fingerprint(sample_dir)
    cache_path = matrix_root / "cells" / f"{profile.name}.json"
    if cache_path.exists():
        cached = _read_json(cache_path)
        if _cell_cache_matches(
            cached, profile, checkpoint, seed, requested_samples, fingerprint
        ):
            print(f"[metrics] reusing complete profile {profile.name}")
            return cached

    def run(_attempt: int):
        batches = load_trimmed_batches(sample_dir, requested_samples)
        result = evaluate_batches(batches, QM9_NO_H_ATOM_DECODER)
        cell = {
            "schema_version": SCHEMA_VERSION,
            "timestamp": _utc_timestamp(),
            "profile": profile.name,
            "enforced_min_rings": profile.min_rings,
            "enforced_min_ring_length": profile.min_ring_length,
            "checkpoint": _normalized_checkpoint(checkpoint),
            "seed": seed,
            "requested_samples": requested_samples,
            "source_fingerprint": fingerprint,
            **result,
        }
        _atomic_write_json(cache_path, cell)
        return cell

    return _retry(
        run,
        attempts=attempts,
        error_path=error_path,
        stage="metrics",
        profile=profile,
    )


def _matrix_for(cells: dict[str, dict[str, Any]], metric: str) -> np.ndarray:
    return np.array(
        [
            [
                cells[
                    next(
                        profile.name
                        for profile in PROFILES
                        if profile.min_rings == count
                        and profile.min_ring_length == length
                    )
                ]["metrics"][metric]
                for length in LENGTH_LEVELS
            ]
            for count in COUNT_LEVELS
        ],
        dtype=float,
    )


def matrix_for_profiles(
    cells: dict[str, dict[str, Any]],
    metric: str,
    profiles: Iterable[Any],
    count_levels: tuple[int | None, ...],
    length_levels: tuple[int | None, ...],
    count_attribute: str,
    length_attribute: str,
) -> np.ndarray:
    profiles = tuple(profiles)
    return np.array(
        [
            [
                cells[
                    next(
                        profile.name
                        for profile in profiles
                        if getattr(profile, count_attribute) == count
                        and getattr(profile, length_attribute) == length
                    )
                ]["metrics"][metric]
                for length in length_levels
            ]
            for count in count_levels
        ],
        dtype=float,
    )


def _level_label(value: int | None, comparison_symbol: str = "≥") -> str:
    return "none" if value is None else f"{comparison_symbol}{value}"


def _metric_slug(metric: str) -> str:
    return metric.removesuffix("_pct")


def _render_markdown_grid(
    title: str,
    matrix: np.ndarray,
    count_levels: tuple[int | None, ...] = COUNT_LEVELS,
    length_levels: tuple[int | None, ...] = LENGTH_LEVELS,
    bound_label: str = "minimum",
    comparison_symbol: str = "≥",
) -> str:
    headers = [_level_label(value, comparison_symbol) for value in length_levels]
    lines = [
        f"# {title}",
        "",
        "Values are percentages over the exact requested graph count.",
        "",
        f"| Enforced {bound_label} ring count \\ enforced {bound_label} maximum-cycle length | "
        + " | ".join(headers)
        + " |",
        "|---|" + "---:|" * len(headers),
    ]
    for row_index, count in enumerate(count_levels):
        values = " | ".join(f"{value:.4f}%" for value in matrix[row_index])
        lines.append(f"| {_level_label(count, comparison_symbol)} | {values} |")
    return "\n".join(lines) + "\n"


def _save_heatmap(
    path: Path,
    title: str,
    matrix: np.ndarray,
    count_levels: tuple[int | None, ...] = COUNT_LEVELS,
    length_levels: tuple[int | None, ...] = LENGTH_LEVELS,
    bound_label: str = "minimum",
    comparison_symbol: str = "≥",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.2, 5.4))
    image = axis.imshow(matrix, cmap="viridis", vmin=0, vmax=100, aspect="auto")
    axis.set_xticks(range(len(length_levels)))
    axis.set_xticklabels(
        [_level_label(value, comparison_symbol) for value in length_levels]
    )
    axis.set_yticks(range(len(count_levels)))
    axis.set_yticklabels(
        [_level_label(value, comparison_symbol) for value in count_levels]
    )
    axis.set_xlabel(f"Enforced {bound_label} maximum-cycle length")
    axis.set_ylabel(f"Enforced {bound_label} ring count")
    axis.set_title(title)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            color = "white" if value < 50 else "black"
            axis.text(
                column,
                row,
                f"{value:.2f}%",
                ha="center",
                va="center",
                color=color,
                fontweight="bold",
            )
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Satisfied graphs (%)")
    figure.tight_layout()
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=path.suffix, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        figure.savefig(temporary, dpi=180, bbox_inches="tight")
        os.replace(temporary, path)
    finally:
        plt.close(figure)
        if temporary.exists():
            temporary.unlink()


def render_aggregate(
    matrix_root: Path,
    cells: dict[str, dict[str, Any]],
    checkpoint: Path,
    seed: int,
    requested_samples: int,
    *,
    profiles: tuple[Any, ...] = PROFILES,
    metrics: tuple[tuple[str, str], ...] = METRICS,
    count_levels: tuple[int | None, ...] = COUNT_LEVELS,
    length_levels: tuple[int | None, ...] = LENGTH_LEVELS,
    count_attribute: str = "min_rings",
    length_attribute: str = "min_ring_length",
    count_cell_field: str = "enforced_min_rings",
    length_cell_field: str = "enforced_min_ring_length",
    bound_label: str = "minimum",
    comparison_symbol: str = "≥",
) -> None:
    metric_matrices = {
        metric: matrix_for_profiles(
            cells,
            metric,
            profiles,
            count_levels,
            length_levels,
            count_attribute,
            length_attribute,
        )
        for metric, _ in metrics
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": _utc_timestamp(),
        "checkpoint": _normalized_checkpoint(checkpoint),
        "seed": seed,
        "requested_samples_per_cell": requested_samples,
        "row_axis": {
            "name": count_cell_field,
            "values": list(count_levels),
        },
        "column_axis": {
            "name": length_cell_field,
            "values": list(length_levels),
        },
        "cells": [cells[profile.name] for profile in profiles],
        "grids": {
            metric: matrix.tolist() for metric, matrix in metric_matrices.items()
        },
    }
    _atomic_write_json(matrix_root / "metrics.json", payload)

    stream = io.StringIO()
    fieldnames = [
        "profile",
        count_cell_field,
        length_cell_field,
        "num_graphs",
        *(metric for metric, _ in metrics),
    ]
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    for profile in profiles:
        cell = cells[profile.name]
        writer.writerow(
            {
                "profile": profile.name,
                count_cell_field: getattr(profile, count_attribute),
                length_cell_field: getattr(profile, length_attribute),
                "num_graphs": cell["num_graphs"],
                **cell["metrics"],
            }
        )
    _atomic_write_text(matrix_root / "metrics.csv", stream.getvalue())

    for metric, title in metrics:
        slug = _metric_slug(metric)
        matrix = metric_matrices[metric]
        _atomic_write_text(
            matrix_root / "grids" / f"{slug}.md",
            _render_markdown_grid(
                title,
                matrix,
                count_levels,
                length_levels,
                bound_label,
                comparison_symbol,
            ),
        )
        _save_heatmap(
            matrix_root / "heatmaps" / f"{slug}.png",
            title,
            matrix,
            count_levels,
            length_levels,
            bound_label,
            comparison_symbol,
        )
        _save_heatmap(
            matrix_root / "heatmaps" / f"{slug}.pdf",
            title,
            matrix,
            count_levels,
            length_levels,
            bound_label,
            comparison_symbol,
        )


def _manifest_request(
    checkpoint: Path,
    experiment: str,
    seed: int,
    requested_samples: int,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "checkpoint": _normalized_checkpoint(checkpoint),
        "experiment": experiment,
        "seed": seed,
        "requested_samples_per_cell": requested_samples,
        "profiles": [profile.name for profile in PROFILES],
        "metrics": [metric for metric, _ in METRICS],
    }


def initialize_manifest(
    matrix_root: Path,
    checkpoint: Path,
    experiment: str,
    seed: int,
    requested_samples: int,
) -> dict[str, Any]:
    request = _manifest_request(checkpoint, experiment, seed, requested_samples)
    path = matrix_root / "manifest.json"
    if path.exists():
        existing = _read_json(path)
        existing_request = {key: existing.get(key) for key in request}
        if existing_request != request:
            raise ArtifactMismatchError(
                f"Matrix manifest {path} belongs to a different request."
            )
        return existing
    manifest = {
        **request,
        "created_at": _utc_timestamp(),
        "generation_complete": False,
        "metrics_complete": False,
    }
    _atomic_write_json(path, manifest)
    return manifest


def update_manifest(matrix_root: Path, **updates: Any) -> None:
    path = matrix_root / "manifest.json"
    manifest = _read_json(path)
    manifest.update(updates)
    manifest["updated_at"] = _utc_timestamp()
    _atomic_write_json(path, manifest)


def run_matrix(args: argparse.Namespace) -> Path:
    repo_root = Path(args.repo_root).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    output_root = Path(args.output_root).expanduser().resolve()
    matrix_root = output_root / "matrix" / f"seed_{args.seed}"
    matrix_root.mkdir(parents=True, exist_ok=True)
    initialize_manifest(
        matrix_root,
        checkpoint,
        args.experiment,
        args.seed,
        args.samples,
    )

    if args.phase in {"all", "generate"}:
        forced = set(args.force_profile or ())
        unknown = forced - set(PROFILE_BY_NAME)
        if unknown:
            raise ValueError(f"Unknown forced profiles: {sorted(unknown)}")
        for profile in PROFILES:
            generate_profile(
                repo_root=repo_root,
                output_root=output_root,
                matrix_root=matrix_root,
                experiment=args.experiment,
                profile=profile,
                checkpoint=checkpoint,
                seed=args.seed,
                requested_samples=args.samples,
                attempts=args.max_attempts,
                force=profile.name in forced,
            )
        update_manifest(matrix_root, generation_complete=True)

    if args.phase in {"all", "metrics"}:
        cells = {}
        for profile in PROFILES:
            cells[profile.name] = evaluate_profile(
                output_root=output_root,
                matrix_root=matrix_root,
                profile=profile,
                checkpoint=checkpoint,
                seed=args.seed,
                requested_samples=args.samples,
                attempts=args.max_attempts,
            )

        error_path = matrix_root / "errors" / "aggregate.json"
        _retry(
            lambda _attempt: render_aggregate(
                matrix_root, cells, checkpoint, args.seed, args.samples
            ),
            attempts=args.max_attempts,
            error_path=error_path,
            stage="aggregate",
            profile=None,
        )
        update_manifest(matrix_root, generation_complete=True, metrics_complete=True)

    return matrix_root


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Generate the nine predefined QM9 projection profiles and "
            "cross-evaluate five metrics over every saved sample set."
        )
    )
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument(
        "--phase", choices=("all", "generate", "metrics"), default="all"
    )
    parser.add_argument(
        "--experiment",
        default="training/qm9_no_constraint_edge_addition",
    )
    parser.add_argument(
        "--output-root",
        default="samples/qm9_no_constraint_edge_addition",
    )
    parser.add_argument("--repo-root", default=str(repo_root))
    parser.add_argument(
        "--force-profile",
        action="append",
        choices=tuple(PROFILE_BY_NAME),
        help="Archive and regenerate one completed profile; may be repeated.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive.")
    if args.max_attempts <= 0:
        raise ValueError("--max-attempts must be positive.")
    matrix_root = run_matrix(args)
    print(f"Constraint matrix complete: {matrix_root}")


if __name__ == "__main__":
    main()
