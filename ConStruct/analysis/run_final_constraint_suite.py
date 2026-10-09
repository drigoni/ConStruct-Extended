#!/usr/bin/env python3
"""Resumable 30-profile QM9 generation and cross-evaluation suite.

Generation is deliberately isolated from metrics: the model process writes an
immutable sample pickle plus compact telemetry, while this command computes
all CPU/ChemNet analysis later. One invocation handles one modality and seed,
which makes insertion/deletion safe to place in separate GPU containers.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import pickle
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import fcd
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from ConStruct.analysis.fcd_matrix import (
    fcd_version,
    load_or_build_reference_cache,
    validation_reference_profiles,
)
from ConStruct.analysis.final_suite_telemetry import atomic_json
from ConStruct.metrics.sampling_molecular_metrics import (
    Molecule,
    _stable_frechet_distance,
    compute_valid_molecules,
)
from ConStruct.projector.graph_cycles import bitmask_cycle_summary, topology_bitmask
from ConStruct.utils import PlaceHolder


SCHEMA_VERSION = 1
DEAD_END_DEFINITION_VERSION = "global_constrained_reachability_v1"
CYCLE_NUM_LEVELS = (None, 0, 1, 2, 3, 9)
MAX_CYCLE_LEN_LEVELS = (None, 0, 3, 4, 9)
REQUIRED_GENERATION_FILES = (
    "generated_samples_rank0.pkl",
    "generation_telemetry.npz",
    "blocked_actions.json",
)
ATOM_DECODER = ["C", "N", "O", "F"]


@dataclass(frozen=True)
class Profile:
    cycle_num: int | None
    max_cycle_len: int | None

    @property
    def name(self) -> str:
        count = "none" if self.cycle_num is None else str(self.cycle_num)
        length = "none" if self.max_cycle_len is None else str(self.max_cycle_len)
        return f"cycle_num_{count}__max_cycle_len_{length}"

    def effective_values(self, modality: str) -> tuple[int | None, int | None]:
        if modality == "insertion":
            return (
                None if self.cycle_num in (None, 0) else self.cycle_num,
                None if self.max_cycle_len in (None, 0) else self.max_cycle_len,
            )
        return self.cycle_num, self.max_cycle_len

    def constraints(self, modality: str) -> list[dict[str, Any]]:
        count, length = self.effective_values(modality)
        suffix = "at_least" if modality == "insertion" else "at_most"
        constraints: list[dict[str, Any]] = []
        if count is not None:
            constraints.append(
                {"type": f"ring_count_{suffix}", ("min_rings" if modality == "insertion" else "max_rings"): count}
            )
        if length is not None:
            constraints.append(
                {"type": f"ring_length_{suffix}", ("min_ring_length" if modality == "insertion" else "max_ring_length"): length}
            )
        return constraints

    def matches(self, modality: str, cycle_num: int, max_cycle_len: int) -> bool:
        compare = (lambda value, bound: value >= bound) if modality == "insertion" else (lambda value, bound: value <= bound)
        return (
            (self.cycle_num is None or compare(cycle_num, self.cycle_num))
            and (self.max_cycle_len is None or compare(max_cycle_len, self.max_cycle_len))
        )


PROFILES = tuple(Profile(count, length) for count in CYCLE_NUM_LEVELS for length in MAX_CYCLE_LEN_LEVELS)
PROFILE_BY_NAME = {profile.name: profile for profile in PROFILES}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def balanced_batch_sizes(total: int, rounds: int) -> list[int]:
    quotient, remainder = divmod(int(total), int(rounds))
    return [quotient + (index < remainder) for index in range(rounds)]


def generation_batch_sizes(total: int, initial_batch_size: int, attempt: int) -> list[int]:
    """Return balanced batches, adding one round after every failed attempt."""
    initial_rounds = max(1, math.ceil(total / initial_batch_size))
    return balanced_batch_sizes(total, initial_rounds + attempt - 1)


@contextmanager
def seed_run_lock(root: Path, seed: int, phase: str):
    """Prevent two runners from mutating the same modality/seed artifacts."""
    lock_dir = root / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"seed_{seed}.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.seek(0)
            owner = handle.read().strip() or "owner information unavailable"
            raise RuntimeError(
                f"Another {root.name} runner already holds {lock_path}: {owner}"
            ) from error
        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "phase": phase,
                    "seed": seed,
                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
                sort_keys=True,
            )
            + "\n"
        )
        handle.flush()
        yield lock_path
    finally:
        handle.close()


@contextmanager
def profile_run_lock(root: Path, seed: int, phase: str, profile_name: str):
    """Serialize mutations of one profile while allowing disjoint shards."""
    lock_dir = root / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"seed_{seed}.{profile_name}.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.seek(0)
            owner = handle.read().strip() or "owner information unavailable"
            raise RuntimeError(
                f"Another runner already holds {lock_path}: {owner}"
            ) from error
        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "phase": phase,
                    "seed": seed,
                    "profile": profile_name,
                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
                sort_keys=True,
            )
            + "\n"
        )
        handle.flush()
        yield lock_path
    finally:
        handle.close()


def select_checkpoint(repo_root: Path, modality: str, seed: int, override: str | None = None) -> Path:
    if override:
        checkpoint = Path(override).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return checkpoint
    family = "edge_addition" if modality == "insertion" else "edge_absorbing"
    directory = repo_root / "checkpoints" / f"qm9_no_constraint_{family}_large_seed_{seed}"
    candidates = []
    for path in directory.glob("epoch=*.ckpt"):
        if path.name.startswith("last"):
            continue
        match = re.fullmatch(r"epoch=(\d+)\.ckpt", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No numeric epoch checkpoint found in {directory}")
    return max(candidates)[1].resolve()


def experiment_name(modality: str, seed: int) -> str:
    family = "edge_addition" if modality == "insertion" else "edge_absorbing"
    return f"training/seeded/qm9_no_constraint_{family}_large_seed_{seed}"


def _hydra_string(key: str, value: str | Path) -> str:
    escaped = str(value).replace(chr(92), chr(92) * 2).replace(chr(39), chr(92) + chr(39))
    return f"{key}='{escaped}'"


def _hydra_constraints(constraints: list[dict[str, Any]]) -> str:
    if not constraints:
        return "model.constraints=[]"
    entries = []
    for constraint in constraints:
        fields = ",".join(f"{key}:{value}" for key, value in constraint.items())
        entries.append("{" + fields + "}")
    return "model.constraints=[" + ",".join(entries) + "]"


def generation_command(
    repo_root: Path,
    modality: str,
    seed: int,
    profile: Profile,
    checkpoint: Path,
    output_dir: Path,
    samples: int,
    batch_sizes: list[int],
    trajectory_samples: int = 256,
) -> list[str]:
    constraints = profile.constraints(modality)
    return [
        sys.executable,
        str(repo_root / "main.py"),
        f"+experiment={experiment_name(modality, seed)}",
        f"train.seed={seed}",
        _hydra_string("general.test_only", checkpoint),
        "general.sampling_only=true",
        "general.wandb=disabled",
        "general.final_model_samples_to_save=0",
        "general.final_model_chains_to_save=0",
        f"general.final_model_samples_to_generate={samples}",
        _hydra_string("general.sampling_output_dir", output_dir),
        "+general.strict_generation=true",
        f"+general.trajectory_samples={trajectory_samples}",
        f"+general.telemetry_seed={seed + 104729}",
        "+general.final_generation_batch_sizes=[" + ",".join(map(str, batch_sizes)) + "]",
        _hydra_constraints(constraints),
        "model.rev_proj=null",
        f"model.use_projection={'true' if constraints else 'false'}",
    ]


def request_identity(
    modality: str,
    seed: int,
    profile: Profile,
    checkpoint: Path,
    samples: int,
    trajectory_samples: int,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "modality": modality,
        "reverse_operation": "deletion" if modality == "insertion" else "insertion",
        "seed": seed,
        "profile": profile.name,
        "labeled_constraints": {"cycle_num": profile.cycle_num, "max_cycle_len": profile.max_cycle_len},
        "effective_constraints": profile.constraints(modality),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "requested_samples": samples,
        "trajectory_samples": min(trajectory_samples, samples),
    }


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def generation_complete(cell: Path, identity: dict[str, Any]) -> bool:
    manifest_path = cell / "generation_manifest.json"
    if not manifest_path.is_file() or not all((cell / name).is_file() for name in REQUIRED_GENERATION_FILES):
        return False
    manifest = _load_json(manifest_path)
    actual = {key: manifest.get(key) for key in identity}
    if actual != identity:
        raise RuntimeError(f"Generation artifact mismatch in {cell}: expected {identity}, found {actual}")
    return all(sha256_file(cell / name) == digest for name, digest in manifest["artifact_sha256"].items())


def archive_cell(cell: Path) -> None:
    if not cell.exists():
        return
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    target = cell.with_name(f"{cell.name}.replaced_{stamp}")
    counter = 1
    while target.exists():
        target = cell.with_name(f"{cell.name}.replaced_{stamp}_{counter}")
        counter += 1
    shutil.move(cell, target)


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def write_manifest(cell: Path, identity: dict[str, Any], batch_sizes: list[int], reused_from: str | None = None) -> None:
    artifacts = {name: sha256_file(cell / name) for name in REQUIRED_GENERATION_FILES}
    completion = cell / "strict_generation_complete.json"
    if completion.is_file():
        artifacts[completion.name] = sha256_file(completion)
    atomic_json(
        cell / "generation_manifest.json",
        {
            **identity,
            "status": "complete",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "batch_sizes": batch_sizes,
            "reused_from": reused_from,
            "artifact_sha256": artifacts,
        },
    )


def generate_cell(args, repo_root: Path, root: Path, profile: Profile, checkpoint: Path, source: Path | None) -> None:
    cell = root / profile.name / f"seed_{args.seed}"
    identity = request_identity(args.modality, args.seed, profile, checkpoint, args.samples, args.trajectory_samples)
    force = profile.name in args.force_profile
    if force:
        archive_cell(cell)
    elif generation_complete(cell, identity):
        print(f"[reuse] {profile.name}")
        return

    if source is not None and not force:
        cell.mkdir(parents=True, exist_ok=True)
        for name in REQUIRED_GENERATION_FILES + ("strict_generation_complete.json",):
            if (source / name).is_file():
                _link_or_copy(source / name, cell / name)
        source_manifest = _load_json(source / "generation_manifest.json")
        write_manifest(cell, identity, source_manifest["batch_sizes"], str(source))
        print(f"[linked] {profile.name} <- {source.parent.name}")
        return

    cell.parent.mkdir(parents=True, exist_ok=True)
    last_error = "generation did not start"
    for attempt in range(1, args.max_attempts + 1):
        batch_sizes = generation_batch_sizes(args.samples, args.initial_generation_batch_size, attempt)
        work = cell.parent / f".{cell.name}.{profile.name}.attempt_{attempt}"
        if work.exists():
            shutil.rmtree(work)
        command = generation_command(repo_root, args.modality, args.seed, profile, checkpoint, work, args.samples, batch_sizes, args.trajectory_samples)
        environment = os.environ.copy()
        if args.device == "cpu":
            environment["CUDA_VISIBLE_DEVICES"] = ""
        elif args.device.startswith("cuda:"):
            environment["CUDA_VISIBLE_DEVICES"] = args.device.split(":", 1)[1]
        environment["PYTHONUNBUFFERED"] = "1"
        log_dir = root / "errors" / f"seed_{args.seed}"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{profile.name}_attempt_{attempt}.log"
        print(f"[generate] {profile.name}: attempt {attempt}, batches={batch_sizes}")
        print(f"[log] {log_path}")
        with log_path.open("w", encoding="utf-8", buffering=1) as log_handle:
            log_handle.write(f"Command: {shlex.join(command)}\n\n")
            log_handle.flush()
            result = subprocess.run(
                command,
                cwd=repo_root,
                env=environment,
                text=True,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        complete = result.returncode == 0 and all((work / name).is_file() for name in REQUIRED_GENERATION_FILES)
        if complete:
            if cell.exists():
                archive_cell(cell)
            os.replace(work, cell)
            write_manifest(cell, identity, batch_sizes)
            return
        last_error = log_text[-4000:]
        if work.exists():
            shutil.rmtree(work)
        if "out of memory" not in log_text.lower() and result.returncode != 0:
            print(f"[retry] non-OOM failure for {profile.name}", file=sys.stderr)
    raise RuntimeError(f"Generation failed for {profile.name} after {args.max_attempts} attempts:\n{last_error}")


def _iter_graphs(batches: Iterable[PlaceHolder]):
    for batch in batches:
        yield from batch.split()


def graph_cycle_properties(batches: list[PlaceHolder]) -> tuple[list[int], list[int]]:
    cache: dict[tuple[int, int], tuple[int, int]] = {}
    counts, lengths = [], []
    for graph in _iter_graphs(batches):
        key = topology_bitmask(graph.E.cpu().numpy(), int(graph.X.shape[0]))
        summary = cache.setdefault(key, bitmask_cycle_summary(*key))
        counts.append(summary[0])
        lengths.append(summary[1])
    return counts, lengths


def exact_histograms(counts: list[int], lengths: list[int]) -> dict[str, dict[str, int]]:
    count_hist = {str(value): 0 for value in range(10)}
    count_hist[">=10"] = 0
    length_hist = {str(value): 0 for value in range(10)}
    for value in counts:
        count_hist[str(value) if value < 10 else ">=10"] += 1
    for value in lengths:
        if not 0 <= value <= 9:
            raise ValueError(f"QM9 maximum cycle length outside 0..9: {value}")
        length_hist[str(value)] += 1
    return {"cycle_count": count_hist, "max_cycle_length": length_hist}


def cross_constraint_validity(modality: str, counts: list[int], lengths: list[int]) -> dict[str, Any]:
    denominator = len(counts)
    result = {}
    for target in PROFILES:
        valid = sum(target.matches(modality, count, length) for count, length in zip(counts, lengths))
        result[target.name] = {
            "count": valid,
            "percentage": 100.0 * valid / denominator if denominator else 0.0,
            "denominator": denominator,
            "target": {"cycle_num": target.cycle_num, "max_cycle_len": target.max_cycle_len},
        }
    return result


def trajectory_payload(path: Path, blocked_actions_path: Path | None = None) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as source:
        indices = source["global_indices"].astype(int)
        timesteps = source["timesteps"].astype(int)
        edges = source["E"]
        masks = source["node_mask"]
    cache: dict[tuple[int, int], tuple[int, int]] = {}
    count_rows: list[list[int]] = []
    length_rows: list[list[int]] = []
    previous_keys: list[tuple[int, int] | None] = [None] * len(indices)
    previous_values: list[tuple[int, int] | None] = [None] * len(indices)
    for state_index in range(len(timesteps)):
        row_counts, row_lengths = [], []
        for graph_index in range(len(indices)):
            n = int(masks[state_index, graph_index].sum())
            key = topology_bitmask(edges[state_index, graph_index], n)
            if key == previous_keys[graph_index]:
                value = previous_values[graph_index]
            else:
                value = cache.setdefault(key, bitmask_cycle_summary(*key))
                previous_keys[graph_index] = key
                previous_values[graph_index] = value
            assert value is not None
            row_counts.append(value[0])
            row_lengths.append(value[1])
        count_rows.append(row_counts)
        length_rows.append(row_lengths)
    per_graph = [
        {
            "global_index": int(index),
            "cycle_count": [row[column] for row in count_rows],
            "max_cycle_length": [row[column] for row in length_rows],
        }
        for column, index in enumerate(indices)
    ]
    aggregate = []
    for row, timestep in enumerate(timesteps):
        aggregate.append(
            {
                "timestep": int(timestep),
                "mean_cycle_count": float(np.mean(count_rows[row])) if len(indices) else 0.0,
                "mean_max_cycle_length": float(np.mean(length_rows[row])) if len(indices) else 0.0,
                "cycle_count_sum": int(np.sum(count_rows[row])),
                "max_cycle_length_max": int(np.max(length_rows[row])) if len(indices) else 0,
            }
        )
    edge_changes = None
    if blocked_actions_path is not None:
        blocked = _load_json(blocked_actions_path)
        edge_changes = {
            "per_step": [
                {
                    "timestep": entry["timestep"],
                    "edge_additions": entry["edge_additions"],
                    "edge_deletions": entry["edge_deletions"],
                }
                for entry in blocked["per_step"]
            ],
            "totals": {
                "edge_additions": blocked["totals"]["edge_additions"],
                "edge_deletions": blocked["totals"]["edge_deletions"],
            },
        }
    return {"schema_version": SCHEMA_VERSION, "timesteps": timesteps.tolist(), "global_indices": indices.tolist(), "per_graph": per_graph, "aggregate": aggregate, "edge_changes_all_generated_graphs": edge_changes}


def _molecule_valid(X: np.ndarray, charges: np.ndarray, E: np.ndarray) -> bool:
    graph = PlaceHolder(
        X=torch.as_tensor(X, dtype=torch.long),
        charges=torch.as_tensor(charges, dtype=torch.long),
        E=torch.as_tensor(E, dtype=torch.long),
        y=torch.empty((0,)),
        node_mask=None,
    )
    valid, _, _ = compute_valid_molecules([Molecule(graph, atom_decoder=ATOM_DECODER)])
    return bool(valid)


def _edge_pairs(node_count: int) -> list[tuple[int, int]]:
    return [(u, v) for u in range(node_count) for v in range(u + 1, node_count)]


def _edge_matrix_from_bits(node_count: int, edge_bits: int) -> np.ndarray:
    """Build the neutral-carbon witness edge matrix (all retained bonds single)."""
    edges = np.zeros((node_count, node_count), dtype=np.int64)
    for shift, (u, v) in enumerate(_edge_pairs(node_count)):
        if edge_bits & (1 << shift):
            edges[u, v] = edges[v, u] = 1
    return edges


def _simple_degrees(node_count: int, edge_bits: int) -> list[int]:
    degrees = [0] * node_count
    for shift, (u, v) in enumerate(_edge_pairs(node_count)):
        if edge_bits & (1 << shift):
            degrees[u] += 1
            degrees[v] += 1
    return degrees


def _topology_satisfies_profile(
    modality: str, profile: Profile, node_count: int, edge_bits: int
) -> bool:
    cycle_count, max_cycle_length = bitmask_cycle_summary(node_count, edge_bits)
    count_bound, length_bound = profile.effective_values(modality)
    compare = (lambda value, bound: value >= bound) if modality == "insertion" else (
        lambda value, bound: value <= bound
    )
    return (
        (count_bound is None or compare(cycle_count, count_bound))
        and (length_bound is None or compare(max_cycle_length, length_bound))
    )


def _result(status: str, reason: str, **values: Any) -> dict[str, Any]:
    return {"status": status, "reason": reason, **values}


def _deletion_reachability(
    X: np.ndarray,
    charges: np.ndarray,
    edges: np.ndarray,
    profile: Profile,
    max_search_states: int,
) -> dict[str, Any]:
    """Exact constrained reachability for reverse edge deletion."""
    del X, charges
    node_count, initial_bits = topology_bitmask(edges, len(edges))
    frontier = [initial_bits]
    seen = {initial_bits}
    explored = 0
    constraint_prunes = 0
    pairs = _edge_pairs(node_count)

    while frontier:
        if explored >= max_search_states:
            return _result(
                "indeterminate",
                "search_budget_exhausted",
                states_explored=explored,
                states_seen=len(seen),
                frontier_size=len(frontier),
                constraint_prunes=constraint_prunes,
            )
        edge_bits = frontier.pop()
        explored += 1
        degrees = _simple_degrees(node_count, edge_bits)
        maximum_degree = max(degrees, default=0)
        if maximum_degree <= 4:
            witness_edges = _edge_matrix_from_bits(node_count, edge_bits)
            witness_valid = _molecule_valid(
                np.zeros(node_count, dtype=np.int64),
                np.zeros(node_count, dtype=np.int64),
                witness_edges,
            )
            if not witness_valid:
                return _result(
                    "indeterminate",
                    "canonical_carbon_witness_rejected",
                    states_explored=explored,
                    states_seen=len(seen),
                    constraint_prunes=constraint_prunes,
                )
            cycles = bitmask_cycle_summary(node_count, edge_bits)
            return _result(
                "repairable",
                "canonical_carbon_witness",
                states_explored=explored,
                states_seen=len(seen),
                constraint_prunes=constraint_prunes,
                witness={
                    "maximum_degree": maximum_degree,
                    "cycle_count": cycles[0],
                    "max_cycle_length": cycles[1],
                    "deleted_edges": initial_bits.bit_count() - edge_bits.bit_count(),
                },
            )

        # Every chemically valid endpoint must remove an edge incident to an
        # over-degree vertex. Branching only there is therefore complete.
        over_degree_vertex = max(range(node_count), key=lambda vertex: degrees[vertex])
        for shift, (u, v) in enumerate(pairs):
            mask = 1 << shift
            if not edge_bits & mask or over_degree_vertex not in (u, v):
                continue
            candidate = edge_bits & ~mask
            if candidate in seen:
                continue
            seen.add(candidate)
            if _topology_satisfies_profile("insertion", profile, node_count, candidate):
                frontier.append(candidate)
            else:
                constraint_prunes += 1

    return _result(
        "stuck",
        "all_constraint_preserving_deletion_paths_exhausted",
        states_explored=explored,
        states_seen=len(seen),
        constraint_prunes=constraint_prunes,
    )


def _insertion_reachability(
    X: np.ndarray,
    charges: np.ndarray,
    edges: np.ndarray,
    profile: Profile,
    max_search_states: int,
) -> dict[str, Any]:
    """Exact constrained reachability for reverse edge insertion on QM9."""
    del X, charges, profile, max_search_states
    node_count = len(edges)
    if np.any(edges == 4):
        return _result(
            "indeterminate",
            "aromatic_edge_outside_qm9_proof",
            states_explored=1,
            states_seen=1,
            constraint_prunes=0,
        )
    bond_orders = np.asarray([0.0, 1.0, 2.0, 3.0, 1.5])
    valence = bond_orders[edges].sum(axis=1)
    if bool(np.any(valence > 4.0)):
        return _result(
            "stuck",
            "immutable_bond_valence_exceeds_qm9_maximum",
            states_explored=1,
            states_seen=1,
            constraint_prunes=0,
            maximum_bond_valence=float(np.max(valence)),
        )
    witness_valid = _molecule_valid(
        np.zeros(node_count, dtype=np.int64),
        np.zeros(node_count, dtype=np.int64),
        edges.copy(),
    )
    if not witness_valid:
        return _result(
            "indeterminate",
            "canonical_carbon_witness_rejected",
            states_explored=1,
            states_seen=1,
            constraint_prunes=0,
        )
    return _result(
        "repairable",
        "canonical_carbon_witness",
        states_explored=1,
        states_seen=1,
        constraint_prunes=0,
    )


def _state_reachability(
    modality: str,
    profile: Profile,
    X: np.ndarray,
    charges: np.ndarray,
    edges: np.ndarray,
    max_search_states: int,
) -> dict[str, Any]:
    currently_valid = _molecule_valid(X, charges, edges)
    if currently_valid:
        return _result(
            "repairable",
            "currently_valid",
            currently_valid=True,
            states_explored=0,
            states_seen=0,
            constraint_prunes=0,
        )
    node_count, edge_bits = topology_bitmask(edges, len(edges))
    if not _topology_satisfies_profile(modality, profile, node_count, edge_bits):
        return _result(
            "indeterminate",
            "recorded_state_violates_active_constraints",
            currently_valid=False,
            states_explored=0,
            states_seen=0,
            constraint_prunes=0,
        )
    solver = _deletion_reachability if modality == "insertion" else _insertion_reachability
    return {
        "currently_valid": False,
        **solver(X, charges, edges, profile, max_search_states),
    }


def _stuck_graph(task) -> dict[str, Any]:
    (
        global_index,
        timesteps,
        X_states,
        charge_states,
        E_states,
        mask_states,
        selected,
        modality,
        profile,
        max_search_states,
    ) = task
    cache: dict[tuple[bytes, bytes, bytes], dict[str, Any]] = {}
    queried: dict[int, dict[str, Any]] = {}

    def evaluate(selected_position: int) -> dict[str, Any]:
        state_index = selected[selected_position]
        n = int(mask_states[state_index].sum())
        X = np.ascontiguousarray(X_states[state_index, :n])
        charges = np.ascontiguousarray(charge_states[state_index, :n])
        edges = np.ascontiguousarray(E_states[state_index, :n, :n])
        key = (X.tobytes(), charges.tobytes(), edges.tobytes())
        if key not in cache:
            cache[key] = _state_reachability(
                modality, profile, X, charges, edges, max_search_states
            )
        queried[selected_position] = cache[key]
        return cache[key]

    final_position = len(selected) - 1
    final = evaluate(final_position)
    final_valid = bool(final.get("currently_valid"))
    first_stuck_position: int | None = None
    overall_status = "resolved"
    if final["status"] == "indeterminate":
        overall_status = "indeterminate"
    elif final["status"] == "stuck":
        low, high = 0, final_position
        while low < high:
            middle = (low + high) // 2
            result = evaluate(middle)
            if result["status"] == "indeterminate":
                overall_status = "indeterminate"
                break
            if result["status"] == "stuck":
                high = middle
            else:
                low = middle + 1
        if overall_status == "resolved":
            first_stuck_position = low

    evaluations = []
    for position, state_index in enumerate(selected):
        result = queried.get(position)
        if final_valid:
            status = "not_applicable_final_valid"
            stuck = False
            source = "inferred_from_valid_endpoint"
        elif overall_status == "resolved" and first_stuck_position is None:
            status = "repairable"
            stuck = False
            source = "queried" if result is not None else "inferred_from_repairable_endpoint"
        elif overall_status == "resolved":
            stuck = position >= first_stuck_position
            status = "stuck" if stuck else "repairable"
            source = "queried" if result is not None else "inferred_by_monotonicity"
        elif result is not None and result["status"] != "indeterminate":
            status = result["status"]
            stuck = status == "stuck"
            source = "queried"
        else:
            status = "indeterminate"
            stuck = None
            source = "unresolved_after_indeterminate_query"
        evaluations.append(
            {
                "timestep": int(timesteps[state_index]),
                "status": status,
                "stuck": stuck,
                "source": source,
                "query": result,
            }
        )

    return {
        "global_index": int(global_index),
        "status": "skipped_final_valid" if final_valid else overall_status,
        "final_valid": final_valid,
        "first_stuck_selected_position": first_stuck_position,
        "first_stuck_timestep": (
            int(timesteps[selected[first_stuck_position]])
            if first_stuck_position is not None
            else None
        ),
        "queries_performed": len(queried),
        "unique_states_searched": len(cache),
        "evaluations": evaluations,
    }


def stuck_payload(
    path: Path,
    modality: str,
    profile: Profile,
    stride: int,
    workers: int,
    max_search_states: int,
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as source:
        indices = source["global_indices"].astype(int)
        timesteps = source["timesteps"].astype(int)
        X, charges, E, masks = source["X"], source["charges"], source["E"], source["node_mask"]
    selected = [index for index, timestep in enumerate(timesteps) if timestep % stride == 0]
    if 0 not in selected and len(timesteps):
        selected.append(len(timesteps) - 1)
    if not profile.constraints(modality):
        return {
            "schema_version": SCHEMA_VERSION,
            "definition_version": DEAD_END_DEFINITION_VERSION,
            "status": "not_applicable",
            "reason": "no_effective_projection_constraints",
            "modality": modality,
            "reverse_operation": "deletion" if modality == "insertion" else "insertion",
            "profile": profile.name,
            "stride": stride,
            "selected_timesteps": [int(timesteps[index]) for index in selected],
            "cohort_size": int(len(indices)),
            "per_graph": [],
            "aggregate": [],
        }
    tasks = [
        (
            indices[index], timesteps, X[:, index], charges[:, index], E[:, index],
            masks[:, index], selected, modality, profile, max_search_states,
        )
        for index in range(len(indices))
    ]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            graphs = list(pool.map(_stuck_graph, tasks))
    else:
        graphs = [_stuck_graph(task) for task in tasks]
    aggregate = []
    for position, selected_index in enumerate(selected):
        entries = [graph["evaluations"][position] for graph in graphs]
        eligible = [entry for graph, entry in zip(graphs, entries) if not graph["final_valid"]]
        resolved = [entry for entry in eligible if entry["stuck"] is not None]
        aggregate.append(
            {
                "timestep": int(timesteps[selected_index]),
                "cohort_size": len(entries),
                "final_valid_skipped": len(entries) - len(eligible),
                "eligible_final_invalid": len(eligible),
                "resolved": len(resolved),
                "indeterminate": len(eligible) - len(resolved),
                "repairable": sum(entry["stuck"] is False for entry in resolved),
                "stuck": sum(entry["stuck"] is True for entry in resolved),
                "stuck_percentage_of_resolved": (
                    100.0 * sum(entry["stuck"] is True for entry in resolved) / len(resolved)
                    if resolved else None
                ),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "definition_version": DEAD_END_DEFINITION_VERSION,
        "definition": (
            "On a trajectory with an invalid endpoint, a monitored state from which no "
            "RDKit-valid QM9 molecule is reachable by any finite sequence of reverse-kernel-"
            "supported mutations while all active "
            "projection constraints remain satisfied."
        ),
        "status": "complete",
        "modality": modality,
        "reverse_operation": "deletion" if modality == "insertion" else "insertion",
        "profile": profile.name,
        "effective_constraints": profile.constraints(modality),
        "stride": stride,
        "max_search_states_per_query": max_search_states,
        "selected_timesteps": [int(timesteps[index]) for index in selected],
        "proof_assumptions": [
            "QM9 contains at most nine C/N/O/F heavy atoms",
            "atom classes and charges -1/0/+1 are mutable",
            "projection constraints depend only on graph topology",
            "reverse deletion may retype retained positive bonds",
            "reverse insertion preserves existing positive bond classes",
        ],
        "per_graph": graphs,
        "aggregate": aggregate,
        "summary": {
            "cohort_size": len(graphs),
            "final_valid_skipped": sum(graph["final_valid"] for graph in graphs),
            "resolved_final_invalid": sum(
                not graph["final_valid"] and graph["status"] == "resolved" for graph in graphs
            ),
            "indeterminate_final_invalid": sum(
                graph["status"] == "indeterminate" for graph in graphs
            ),
            "graphs_with_dead_end_onset": sum(
                graph["first_stuck_selected_position"] is not None for graph in graphs
            ),
            "reachability_queries": sum(graph["queries_performed"] for graph in graphs),
        },
    }

def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, torch.Tensor):
        return _json_ready(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class StandardMetricsContext:
    def __init__(self, repo_root: Path, modality: str, seed: int):
        from ConStruct.datasets.qm9_dataset import QM9DataModule, QM9Infos

        with initialize_config_dir(version_base="1.3", config_dir=str(repo_root / "configs")):
            self.cfg = compose(
                config_name="config",
                overrides=[f"+experiment={experiment_name(modality, seed)}"],
                return_hydra_config=True,
            )
        HydraConfig.instance().set_config(self.cfg)
        with open_dict(self.cfg):
            self.cfg.model.constraints = []
            self.cfg.model.rev_proj = None
            self.cfg.model.use_projection = False
            self.cfg.general.suppress_metric_artifacts = True
            self.cfg.general.skip_standard_fcd = True
        datamodule = QM9DataModule(self.cfg)
        self.infos = QM9Infos(datamodule, self.cfg)

    def evaluate(self, batches: list[PlaceHolder]) -> dict[str, Any]:
        from ConStruct.metrics.sampling_metrics import SamplingMetrics

        evaluator = SamplingMetrics(self.infos, test=True, cfg=self.cfg)
        values = evaluator.compute_all_metrics(
            batches, current_epoch=0, local_rank=0, log_to_wandb=False
        )[0]
        return _json_ready(values)


class FCDContext:
    def __init__(self, repo_root: Path, modality: str, device: str, workers: int):
        self.repo_root = repo_root
        self.modality = modality
        self.device = device
        self.workers = workers
        self.model = fcd.load_ref_model()
        direction = "at_least" if modality == "insertion" else "at_most"
        self.references = validation_reference_profiles(
            PROFILES, direction=direction, count_attribute="cycle_num", length_attribute="max_cycle_len"
        )
        self.bundle = load_or_build_reference_cache(
            repo_root / "data/qm9/processed/val_smiles_noh.pickle",
            repo_root / f"data/qm9/processed/val_fcd_reference_profiles_{direction}_v2.npz",
            self.references,
            model=self.model,
        )

    def scores(self, valid_smiles: list[str], sample_sha: str, cache_root: Path) -> dict[str, Any]:
        cache_root.mkdir(parents=True, exist_ok=True)
        cache = cache_root / f"generated_fcd_{sample_sha}.npz"
        mu = sigma = None
        error = None
        if cache.is_file():
            with np.load(cache, allow_pickle=False) as stored:
                metadata = json.loads(str(stored["metadata_json"].item()))
                if metadata.get("sample_sha256") == sample_sha and metadata.get("fcd_version") == fcd_version():
                    mu, sigma, error = stored.get("mu"), stored.get("sigma"), metadata.get("error")
        if mu is None and error is None:
            if len(valid_smiles) < 2:
                error = "fewer_than_two_generated_valid_molecules"
            else:
                activations = fcd.get_predictions(self.model, valid_smiles, n_jobs=self.workers, device=self.device)
                mu, sigma = np.mean(activations, axis=0), np.cov(activations.T)
            arrays = {"metadata_json": np.asarray(json.dumps({"sample_sha256": sample_sha, "fcd_version": fcd_version(), "valid_smiles": len(valid_smiles), "error": error}, sort_keys=True))}
            if mu is not None:
                arrays.update(mu=mu, sigma=sigma)
            with tempfile.NamedTemporaryFile(dir=cache_root, suffix=".npz", delete=False) as handle:
                temporary = Path(handle.name)
                np.savez(handle, **arrays)
            os.replace(temporary, cache)
        result = {}
        for reference in self.references:
            statistics = self.bundle["statistics"][reference.name]
            score = None
            current_error = error or statistics.get("error")
            if mu is not None and statistics["status"] == "ok":
                try:
                    score = float(_stable_frechet_distance(mu, sigma, statistics["mu_fcd"], statistics["sigma_fcd"]))
                except Exception as exception:
                    current_error = f"{type(exception).__name__}: {exception}"
            result[reference.name] = {"score": score, "error": current_error, "generated_valid_count": len(valid_smiles), "reference_count": statistics["molecule_count"]}
        return result


def metrics_cell(args, repo_root: Path, root: Path, profile: Profile, checkpoint: Path, standard_context: StandardMetricsContext, fcd_context: FCDContext) -> bool:
    cell = root / profile.name / f"seed_{args.seed}"
    identity = request_identity(args.modality, args.seed, profile, checkpoint, args.samples, args.trajectory_samples)
    if not generation_complete(cell, identity):
        print(f"[skip metrics: generation incomplete] {profile.name}", flush=True)
        return False
    metrics_path = cell / "metrics.json"
    trajectories_path = cell / "trajectories.json"
    stuck_path = cell / "stuck_states.json"
    sample_path = cell / "generated_samples_rank0.pkl"
    telemetry = cell / "generation_telemetry.npz"
    sample_sha = sha256_file(sample_path)
    telemetry_sha = sha256_file(telemetry)
    analysis_identity = {
        "schema_version": SCHEMA_VERSION,
        "sample_sha256": sample_sha,
        "telemetry_sha256": telemetry_sha,
        "stuck_definition_version": DEAD_END_DEFINITION_VERSION,
        "stuck_stride": args.stuck_stride,
        "stuck_max_search_states": args.stuck_max_search_states,
        "modality": args.modality,
    }
    if (
        metrics_path.is_file()
        and trajectories_path.is_file()
        and stuck_path.is_file()
        and profile.name not in args.force_profile
    ):
        existing = _load_json(metrics_path)
        if existing.get("analysis_identity") == analysis_identity and existing.get("status") == "complete":
            print(f"[reuse metrics] {profile.name}", flush=True)
            return True
    with sample_path.open("rb") as handle:
        batches = pickle.load(handle)
    generated = sum(int(batch.X.shape[0]) for batch in batches)
    if generated != args.samples:
        raise RuntimeError(f"Expected {args.samples} generated graphs, found {generated}")
    errors = []
    try:
        standard = standard_context.evaluate(batches)
    except Exception as exception:
        standard = {}
        errors.append({"stage": "standard_metrics", "error": f"{type(exception).__name__}: {exception}"})
    molecules = [Molecule(graph, atom_decoder=ATOM_DECODER) for graph in _iter_graphs(batches)]
    valid_smiles, _, molecular_errors = compute_valid_molecules(molecules)
    counts, lengths = graph_cycle_properties(batches)
    fcd_scores = fcd_context.scores(valid_smiles, sample_sha, root / ".cache")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "sample_sha256": sample_sha,
        "analysis_identity": analysis_identity,
        "fcd_reference": {
            "validation_sha256": fcd_context.bundle["metadata"]["validation_sha256"],
            "fcd_version": fcd_context.bundle["metadata"]["fcd_version"],
            "cache_path": fcd_context.bundle["cache_path"],
        },
        "generated_graphs": generated,
        "standard_metrics": standard,
        "molecular_error_counts": {str(key): value for key, value in molecular_errors.items()},
        "molecular_valid_count": len(valid_smiles),
        "molecular_validity_percentage": 100.0 * len(valid_smiles) / generated,
        "distributions": exact_histograms(counts, lengths),
        "constraint_validity": cross_constraint_validity(args.modality, counts, lengths),
        "fcd": fcd_scores,
        "errors": errors,
    }
    atomic_json(trajectories_path, trajectory_payload(telemetry, cell / "blocked_actions.json"))
    dead_end_cache_identity = {
        "definition_version": DEAD_END_DEFINITION_VERSION,
        "telemetry_sha256": telemetry_sha,
        "modality": args.modality,
        "effective_constraints": profile.constraints(args.modality),
        "stride": args.stuck_stride,
        "max_search_states": args.stuck_max_search_states,
    }
    dead_end_cache_key = hashlib.sha256(
        json.dumps(dead_end_cache_identity, sort_keys=True).encode("utf-8")
    ).hexdigest()
    dead_end_cache_path = root / ".cache" / f"stuck_states_{dead_end_cache_key}.json"
    if dead_end_cache_path.is_file():
        dead_end_payload = _load_json(dead_end_cache_path)
    else:
        dead_end_payload = stuck_payload(
            telemetry,
            args.modality,
            profile,
            args.stuck_stride,
            args.metrics_workers,
            args.stuck_max_search_states,
        )
        dead_end_payload["cache_identity"] = dead_end_cache_identity
        atomic_json(dead_end_cache_path, dead_end_payload)
    atomic_json(stuck_path, dead_end_payload)
    # Written last: this is the completion marker for all derived artifacts.
    atomic_json(metrics_path, _json_ready(payload))
    return True


def partition_metric_profiles(args, root: Path, checkpoint: Path, profiles):
    """Return all ready cells, including out-of-order cells from generation shards."""
    ready, incomplete = [], []
    for profile in profiles:
        cell = root / profile.name / f"seed_{args.seed}"
        identity = request_identity(
            args.modality,
            args.seed,
            profile,
            checkpoint,
            args.samples,
            args.trajectory_samples,
        )
        if generation_complete(cell, identity):
            ready.append(profile)
        else:
            incomplete.append(profile)
    return ready, incomplete


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modality", choices=("insertion", "deletion"), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--phase", choices=("generate", "metrics", "all"), default="all")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--checkpoint")
    parser.add_argument("--initial-generation-batch-size", type=int, default=5000)
    parser.add_argument("--trajectory-samples", type=int, default=256)
    parser.add_argument("--stuck-stride", type=int, default=25)
    parser.add_argument("--stuck-max-search-states", type=int, default=100000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--metrics-workers", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--profile-start-index", type=int, default=0)
    parser.add_argument("--profile-end-index", type=int, default=len(PROFILES))
    parser.add_argument("--force-profile", action="append", default=[])
    args = parser.parse_args(argv)
    if (
        args.samples <= 0
        or args.initial_generation_batch_size <= 0
        or args.trajectory_samples < 0
        or args.stuck_stride <= 0
        or args.stuck_max_search_states <= 0
        or args.metrics_workers <= 0
        or args.max_attempts <= 0
    ):
        parser.error("sample, batch, stride, search, worker, and attempt counts must be positive")
    if not 0 <= args.profile_start_index < args.profile_end_index <= len(PROFILES):
        parser.error(
            f"profile range must satisfy 0 <= start < end <= {len(PROFILES)}"
        )
    unknown = sorted(set(args.force_profile) - set(PROFILE_BY_NAME))
    if unknown:
        parser.error(f"unknown --force-profile values: {unknown}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    root = repo_root / "samples" / "final" / args.modality
    checkpoint = select_checkpoint(repo_root, args.modality, args.seed, args.checkpoint)
    profiles = PROFILES[args.profile_start_index : args.profile_end_index]
    print(
        f"[profile range] {args.profile_start_index}:{args.profile_end_index} "
        f"({len(profiles)} profiles)"
    )
    if args.phase in {"generate", "all"}:
        effective_sources: dict[tuple[int | None, int | None], Path] = {}
        for profile in profiles:
            with profile_run_lock(root, args.seed, "generate", profile.name):
                key = profile.effective_values(args.modality)
                source = effective_sources.get(key)
                generate_cell(args, repo_root, root, profile, checkpoint, source)
                effective_sources.setdefault(
                    key, root / profile.name / f"seed_{args.seed}"
                )
    if args.phase in {"metrics", "all"}:
        ready_profiles, incomplete_profiles = partition_metric_profiles(
            args, root, checkpoint, profiles
        )
        for profile in incomplete_profiles:
            print(
                f"[skip metrics: generation incomplete] {profile.name}",
                flush=True,
            )
        processed = 0
        if ready_profiles:
            standard_context = StandardMetricsContext(
                repo_root, args.modality, args.seed
            )
            fcd_context = FCDContext(
                repo_root, args.modality, args.device, args.metrics_workers
            )
            for profile in ready_profiles:
                with profile_run_lock(root, args.seed, "metrics", profile.name):
                    processed += int(
                        metrics_cell(
                            args,
                            repo_root,
                            root,
                            profile,
                            checkpoint,
                            standard_context,
                            fcd_context,
                        )
                    )
        print(
            f"[metrics scan complete] ready={len(ready_profiles)} "
            f"processed_or_reused={processed} "
            f"skipped_incomplete={len(incomplete_profiles)}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
