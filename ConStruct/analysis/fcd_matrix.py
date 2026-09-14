"""Validation-reference FCD support shared by the QM9 matrix runners."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import pickle
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import fcd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from rdkit import Chem

from ConStruct.metrics.sampling_molecular_metrics import SamplingMolecularMetrics
from ConStruct.projector.graph_cycles import enumerate_simple_cycles_unique


REFERENCE_CACHE_SCHEMA_VERSION = 1


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def fcd_version() -> str:
    try:
        return importlib.metadata.version("fcd")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ValidationReferenceProfile:
    name: str
    ring_count: int | None
    max_cycle_length: int | None
    direction: str

    @property
    def constraints(self) -> list[dict[str, Any]]:
        result = []
        if self.ring_count is not None:
            bound = "min" if self.direction == "at_least" else "max"
            result.append(
                {
                    "type": f"ring_count_{self.direction}",
                    f"{bound}_rings": self.ring_count,
                }
            )
        if self.max_cycle_length is not None:
            bound = "min" if self.direction == "at_least" else "max"
            result.append(
                {
                    "type": f"ring_length_{self.direction}",
                    f"{bound}_ring_length": self.max_cycle_length,
                }
            )
        return result

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "direction": self.direction,
            "ring_count": self.ring_count,
            "max_cycle_length": self.max_cycle_length,
            "constraints": self.constraints,
        }


def validation_reference_profiles(
    profiles: Iterable[Any],
    *,
    direction: str,
    count_attribute: str,
    length_attribute: str,
) -> tuple[ValidationReferenceProfile, ...]:
    if direction not in {"at_least", "at_most"}:
        raise ValueError(f"Unknown validation reference direction: {direction}")
    return tuple(
        ValidationReferenceProfile(
            name=profile.name,
            ring_count=getattr(profile, count_attribute),
            max_cycle_length=getattr(profile, length_attribute),
            direction=direction,
        )
        for profile in profiles
    )


def fcd_metric_key(profile: ValidationReferenceProfile) -> str:
    return f"fcd_vs_validation_{profile.name}"


def _canonical_validation_smiles(path: Path) -> tuple[list[str], int, int]:
    with path.open("rb") as handle:
        source = pickle.load(handle)
    if not isinstance(source, (set, list, tuple)):
        raise TypeError(
            f"Expected a SMILES collection in {path}, got {type(source)!r}."
        )

    canonical = set()
    invalid = 0
    for smile in source:
        try:
            molecule = Chem.MolFromSmiles(smile)
            if molecule is None:
                invalid += 1
                continue
            canonical.add(Chem.MolToSmiles(molecule, canonical=True))
        except Exception:
            invalid += 1
    return sorted(canonical), len(source), invalid


def _cycle_properties(smile: str) -> tuple[int, int]:
    molecule = Chem.MolFromSmiles(smile)
    if molecule is None:
        raise ValueError(f"Canonical validation SMILES could not be parsed: {smile}")
    graph = nx.Graph()
    graph.add_nodes_from(range(molecule.GetNumAtoms()))
    graph.add_edges_from(
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        for bond in molecule.GetBonds()
    )
    cycles = list(enumerate_simple_cycles_unique(graph))
    return len(cycles), max((len(cycle) for cycle in cycles), default=0)


def profile_matches(
    profile: ValidationReferenceProfile, ring_count: int, max_cycle_length: int
) -> bool:
    compare = (
        (lambda value, bound: value >= bound)
        if profile.direction == "at_least"
        else (lambda value, bound: value <= bound)
    )
    return (
        (profile.ring_count is None or compare(ring_count, profile.ring_count))
        and (
            profile.max_cycle_length is None
            or compare(max_cycle_length, profile.max_cycle_length)
        )
    )


def _cache_identity(
    validation_smiles: Path,
    profiles: tuple[ValidationReferenceProfile, ...],
) -> dict[str, Any]:
    return {
        "schema_version": REFERENCE_CACHE_SCHEMA_VERSION,
        "validation_smiles": str(validation_smiles.resolve()),
        "validation_sha256": sha256_file(validation_smiles),
        "fcd_version": fcd_version(),
        "profiles": [profile.as_dict() for profile in profiles],
    }


def _atomic_save_npz(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".npz", delete=False
    ) as handle:
        temporary = Path(handle.name)
        np.savez(handle, **arrays)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_reference_cache(
    path: Path, identity: dict[str, Any]
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as cache:
            metadata = json.loads(str(cache["metadata_json"].item()))
            if {key: metadata.get(key) for key in identity} != identity:
                return None
            if metadata.get("retryable_failures"):
                return None
            statistics = {}
            for profile in identity["profiles"]:
                name = profile["name"]
                entry = metadata["statistics"][name]
                statistics[name] = {
                    **entry,
                    "mu_fcd": (
                        cache[f"{name}__mu"] if entry["status"] == "ok" else None
                    ),
                    "sigma_fcd": (
                        cache[f"{name}__sigma"]
                        if entry["status"] == "ok"
                        else None
                    ),
                }
            return {
                "metadata": metadata,
                "statistics": statistics,
                "cache_path": str(path.resolve()),
            }
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None


def load_or_build_reference_cache(
    validation_smiles: Path,
    cache_path: Path,
    profiles: tuple[ValidationReferenceProfile, ...],
) -> dict[str, Any]:
    validation_smiles = validation_smiles.expanduser().resolve()
    cache_path = cache_path.expanduser().resolve()
    if not validation_smiles.is_file():
        raise FileNotFoundError(f"Validation SMILES not found: {validation_smiles}")

    identity = _cache_identity(validation_smiles, profiles)
    cached = _load_reference_cache(cache_path, identity)
    if cached is not None:
        return cached

    canonical, source_count, invalid_count = _canonical_validation_smiles(
        validation_smiles
    )
    properties = [_cycle_properties(smile) for smile in canonical]
    model = fcd.load_ref_model()
    arrays: dict[str, Any] = {}
    statistics: dict[str, dict[str, Any]] = {}
    metadata_statistics: dict[str, dict[str, Any]] = {}

    for profile in profiles:
        subset = [
            smile
            for smile, (ring_count, max_cycle_length) in zip(canonical, properties)
            if profile_matches(profile, ring_count, max_cycle_length)
        ]
        entry: dict[str, Any] = {
            "profile": profile.as_dict(),
            "molecule_count": len(subset),
        }
        mu_fcd = None
        sigma_fcd = None
        if len(subset) < 2:
            entry.update(
                status="unavailable",
                error="fewer_than_two_validation_molecules",
            )
        else:
            try:
                activations = fcd.get_predictions(model, subset, n_jobs=0)
                mu_fcd = np.mean(activations, axis=0)
                sigma_fcd = np.cov(activations.T)
                if not np.isfinite(mu_fcd).all() or not np.isfinite(
                    sigma_fcd
                ).all():
                    raise ValueError(
                        "reference FCD statistics contain non-finite values"
                    )
                arrays[f"{profile.name}__mu"] = mu_fcd
                arrays[f"{profile.name}__sigma"] = sigma_fcd
                entry.update(status="ok", error=None)
            except Exception as error:
                entry.update(
                    status="unavailable",
                    error=f"{type(error).__name__}: {error}",
                )
        metadata_statistics[profile.name] = entry
        statistics[profile.name] = {
            **entry,
            "mu_fcd": mu_fcd,
            "sigma_fcd": sigma_fcd,
        }

    metadata = {
        **identity,
        "created_at": utc_timestamp(),
        "source_molecule_count": source_count,
        "canonical_unique_molecule_count": len(canonical),
        "invalid_source_molecule_count": invalid_count,
        "retryable_failures": any(
            entry["status"] != "ok"
            and entry.get("error") != "fewer_than_two_validation_molecules"
            for entry in metadata_statistics.values()
        ),
        "statistics": metadata_statistics,
    }
    arrays["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True))
    _atomic_save_npz(cache_path, arrays)
    return {
        "metadata": metadata,
        "statistics": statistics,
        "cache_path": str(cache_path),
    }


def reference_fingerprint(reference_bundle: dict[str, Any]) -> dict[str, Any]:
    metadata = reference_bundle["metadata"]
    cache_path = Path(reference_bundle["cache_path"])
    return {
        "cache_path": reference_bundle["cache_path"],
        "cache_sha256": sha256_file(cache_path) if cache_path.is_file() else None,
        "validation_sha256": metadata["validation_sha256"],
        "fcd_version": metadata["fcd_version"],
        "profiles": metadata["profiles"],
        "statistics": metadata["statistics"],
    }


def unavailable_reference_bundle(
    profiles: tuple[ValidationReferenceProfile, ...],
    validation_smiles: Path,
    cache_path: Path,
    error: BaseException,
) -> dict[str, Any]:
    message = f"{type(error).__name__}: {error}"
    profile_payloads = [profile.as_dict() for profile in profiles]
    statistics = {
        profile.name: {
            "profile": profile.as_dict(),
            "molecule_count": 0,
            "status": "unavailable",
            "error": message,
            "mu_fcd": None,
            "sigma_fcd": None,
        }
        for profile in profiles
    }
    return {
        "cache_path": str(cache_path.expanduser().resolve()),
        "metadata": {
            "schema_version": REFERENCE_CACHE_SCHEMA_VERSION,
            "validation_smiles": str(validation_smiles.expanduser().resolve()),
            "validation_sha256": None,
            "fcd_version": fcd_version(),
            "profiles": profile_payloads,
            "created_at": utc_timestamp(),
            "source_molecule_count": 0,
            "canonical_unique_molecule_count": 0,
            "invalid_source_molecule_count": 0,
            "statistics": {
                name: {
                    key: value
                    for key, value in entry.items()
                    if key not in {"mu_fcd", "sigma_fcd"}
                }
                for name, entry in statistics.items()
            },
            "error": message,
        },
        "statistics": statistics,
    }


def build_fcd_evaluators(
    reference_bundle: dict[str, Any], atom_decoder: list[str]
) -> dict[str, SamplingMolecularMetrics | None]:
    evaluators = {}
    for name, statistics in reference_bundle["statistics"].items():
        if statistics["status"] != "ok":
            evaluators[name] = None
            continue
        stats = (statistics["mu_fcd"], statistics["sigma_fcd"])
        infos = SimpleNamespace(
            is_molecular=True,
            atom_decoder=atom_decoder,
            remove_h=True,
            train_smiles=set(),
            val_smiles=set(),
            test_smiles=set(),
            val_fcd_stats=stats,
            test_fcd_stats=stats,
        )
        try:
            evaluators[name] = SamplingMolecularMetrics(infos, test=False, cfg=None)
        except Exception as error:
            statistics["status"] = "unavailable"
            statistics["error"] = f"{type(error).__name__}: {error}"
            metadata = reference_bundle["metadata"]["statistics"][name]
            metadata["status"] = "unavailable"
            metadata["error"] = statistics["error"]
            evaluators[name] = None
    return evaluators


def evaluate_generated_smiles(
    generated_smiles: list[str],
    reference_bundle: dict[str, Any],
    evaluators: dict[str, SamplingMolecularMetrics | None],
) -> tuple[dict[str, float | None], dict[str, dict[str, Any]]]:
    metrics: dict[str, float | None] = {}
    details: dict[str, dict[str, Any]] = {}
    for profile in reference_bundle["metadata"]["profiles"]:
        name = profile["name"]
        key = f"fcd_vs_validation_{name}"
        reference = reference_bundle["statistics"][name]
        evaluator = evaluators[name]
        score = None
        error = reference.get("error")
        if len(generated_smiles) < 2:
            error = "fewer_than_two_generated_molecules"
        elif evaluator is not None:
            try:
                score = evaluator.compute_fcd(generated_smiles).get(
                    "val_sampling/fcd score"
                )
                if score is None or not np.isfinite(score):
                    score = None
                    error = "compute_fcd_returned_no_finite_score"
            except Exception as exception:
                error = f"{type(exception).__name__}: {exception}"
        metrics[key] = None if score is None else float(score)
        details[name] = {
            "metric": key,
            "score": metrics[key],
            "status": "ok" if score is not None else "unavailable",
            "error": error,
            "generated_molecule_count": len(generated_smiles),
            "reference_molecule_count": reference["molecule_count"],
            "validation_profile": profile,
        }
    return metrics, details


def fcd_details_cacheable(details: dict[str, dict[str, Any]]) -> bool:
    deterministic_unavailable = {
        "fewer_than_two_generated_molecules",
        "fewer_than_two_validation_molecules",
    }
    return bool(details) and all(
        item.get("status") == "ok"
        or item.get("error") in deterministic_unavailable
        for item in details.values()
    )


def render_fcd_markdown(
    title: str,
    matrix: np.ndarray,
    row_labels: list[str],
    column_labels: list[str],
    row_heading: str,
) -> str:
    lines = [
        f"# {title}",
        "",
        "FCD is computed over valid canonical generated molecules; lower is better.",
        "",
        f"| {row_heading} | " + " | ".join(column_labels) + " |",
        "|---|" + "---:|" * len(column_labels),
    ]
    for row, label in enumerate(row_labels):
        values = " | ".join(
            "N/A" if not np.isfinite(value) else f"{value:.6f}"
            for value in matrix[row]
        )
        lines.append(f"| {label} | {values} |")
    return "\n".join(lines) + "\n"


def save_fcd_heatmap(
    path: Path,
    title: str,
    matrix: np.ndarray,
    row_labels: list[str],
    column_labels: list[str],
    x_label: str,
    y_label: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    finite = matrix[np.isfinite(matrix)]
    vmin = float(finite.min()) if finite.size else 0.0
    vmax = float(finite.max()) if finite.size else 1.0
    if vmax <= vmin:
        padding = max(1e-9, abs(vmin) * 1e-6)
        vmin -= padding
        vmax += padding

    figure, axis = plt.subplots(figsize=(7.2, 5.4))
    cmap = plt.get_cmap("viridis_r").copy()
    cmap.set_bad("#bdbdbd")
    image = axis.imshow(
        np.ma.masked_invalid(matrix),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
    )
    axis.set_xticks(range(len(column_labels)), column_labels)
    axis.set_yticks(range(len(row_labels)), row_labels)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_title(title)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                "N/A" if not np.isfinite(value) else f"{value:.3f}",
                ha="center",
                va="center",
                color="black",
                fontweight="bold",
            )
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("FCD (lower is better)")
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


def json_matrix(matrix: np.ndarray) -> list[list[float | None]]:
    return [
        [None if not np.isfinite(value) else float(value) for value in row]
        for row in matrix
    ]
