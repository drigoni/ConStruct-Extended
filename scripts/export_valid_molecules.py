#!/usr/bin/env python3
"""Export reproducibly selected valid molecules from final-suite seed-0 batches."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import pickle
import random
import shutil
import tempfile
from pathlib import Path
from typing import Iterable

from rdkit import Chem
from rdkit.Chem import rdDepictor

from ConStruct.analysis.run_final_constraint_suite import (
    ATOM_DECODER,
    PROFILES,
    Profile,
)
from ConStruct.metrics.sampling_molecular_metrics import Molecule
from ConStruct.projector.graph_cycles import bitmask_cycle_summary, topology_bitmask
from ConStruct.utils import PlaceHolder


SOURCE_SEED = 0
MODALITIES = ("insertion", "deletion")
LEAF_NAME = "molecules.sdf"
VALIDITY_POLICY = "rdkit_largest_fragment"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--count",
        type=positive_int,
        default=1,
        help="molecules to select without replacement per profile (default: 1)",
    )
    parser.add_argument(
        "--selection-seed",
        type=int,
        default=0,
        help="seed used to initialize one RNG stream per modality (default: 0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("samples/mols"),
        help="output directory, relative to the repository root by default",
    )
    return parser.parse_args(argv)


def iter_graphs(batches: Iterable[PlaceHolder]):
    for batch in batches:
        yield from batch.split()


def largest_valid_fragment(graph: PlaceHolder) -> tuple[Chem.Mol, str] | None:
    """Apply the same RDKit/largest-fragment policy used by sampling metrics."""
    molecule = Molecule(graph, atom_decoder=ATOM_DECODER)
    rdmol = molecule.rdkit_mol
    if rdmol is None:
        return None
    try:
        fragments = Chem.rdmolops.GetMolFrags(
            rdmol, asMols=True, sanitizeFrags=True
        )
        largest = max(fragments, key=lambda fragment: fragment.GetNumAtoms())
        Chem.SanitizeMol(largest)
        smiles = Chem.MolToSmiles(largest, canonical=True)
    except (
        Chem.rdchem.AtomValenceException,
        Chem.rdchem.KekulizeException,
        Chem.rdchem.AtomKekulizeException,
        ValueError,
    ):
        return None
    return largest, smiles


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_fingerprint(
    sample_path: Path, cache: dict[tuple[int, int, int, int], str]
) -> str:
    stat = sample_path.stat()
    key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if key not in cache:
        cache[key] = sha256_file(sample_path)
    return cache[key]


def load_source(
    final_root: Path,
    modality: str,
    profile: Profile,
    fingerprint_cache: dict[tuple[int, int, int, int], str],
) -> tuple[list[PlaceHolder], str, Path]:
    cell = final_root / modality / profile.name / f"seed_{SOURCE_SEED}"
    sample_path = cell / "generated_samples_rank0.pkl"
    metrics_path = cell / "metrics.json"
    manifest_path = cell / "generation_manifest.json"
    for required in (sample_path, metrics_path, manifest_path):
        if not required.is_file():
            raise FileNotFoundError(f"Required source artifact is missing: {required}")

    with metrics_path.open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if metrics.get("status") != "complete":
        raise RuntimeError(f"Metrics are not complete for {cell}")
    if manifest.get("status") not in {"complete", "completed"}:
        raise RuntimeError(f"Generation is not complete for {cell}")

    actual_fingerprint = sample_fingerprint(sample_path, fingerprint_cache)
    if metrics.get("sample_sha256") != actual_fingerprint:
        raise RuntimeError(
            f"Sample fingerprint does not match metrics for {cell}: "
            f"{actual_fingerprint} != {metrics.get('sample_sha256')}"
        )

    with sample_path.open("rb") as handle:
        batches = pickle.load(handle)
    generated = sum(int(batch.X.shape[0]) for batch in batches)
    if generated != metrics.get("generated_graphs"):
        raise RuntimeError(
            f"Generated graph count does not match metrics for {cell}: "
            f"{generated} != {metrics.get('generated_graphs')}"
        )
    return batches, actual_fingerprint, sample_path


def graph_properties(graph: PlaceHolder) -> tuple[int, int]:
    key = topology_bitmask(graph.E.detach().cpu().numpy(), int(graph.X.shape[0]))
    return bitmask_cycle_summary(*key)


def select_profile_molecules(
    batches: list[PlaceHolder],
    modality: str,
    profile: Profile,
    count: int,
    rng: random.Random,
) -> tuple[list[tuple[int, Chem.Mol, str, int, int]], int]:
    graphs = list(iter_graphs(batches))
    indices = list(range(len(graphs)))
    rng.shuffle(indices)
    selected: list[tuple[int, Chem.Mol, str, int, int]] = []
    valid_examined = 0
    for global_index in indices:
        graph = graphs[global_index]
        valid = largest_valid_fragment(graph)
        if valid is None:
            continue
        valid_examined += 1
        molecule, smiles = valid
        cycle_count, max_cycle_length = graph_properties(graph)
        if not profile.matches(modality, cycle_count, max_cycle_length):
            raise RuntimeError(
                f"Selected valid graph violates its profile: modality={modality}, "
                f"profile={profile.name}, global_index={global_index}, "
                f"cycles={cycle_count}, max_cycle_length={max_cycle_length}"
            )
        selected.append(
            (global_index, molecule, smiles, cycle_count, max_cycle_length)
        )
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(
            f"Requested {count} valid molecules for {modality}/{profile.name}, "
            f"but found only {valid_examined}"
        )
    return selected, len(graphs)


def write_profile_sdf(
    path: Path,
    selected: list[tuple[int, Chem.Mol, str, int, int]],
    modality: str,
    profile: Profile,
    selection_seed: int,
    sample_sha256: str,
    source_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(path))
    try:
        for ordinal, (
            global_index,
            molecule,
            smiles,
            cycle_count,
            max_cycle_length,
        ) in enumerate(selected, start=1):
            exported = Chem.Mol(molecule)
            if any(atom.GetAtomicNum() == 1 for atom in exported.GetAtoms()):
                raise RuntimeError(
                    f"Explicit hydrogen found in {modality}/{profile.name}"
                )
            rdDepictor.Compute2DCoords(exported)
            exported.SetProp(
                "_Name",
                f"{modality}__{profile.name}__seed_{SOURCE_SEED}__graph_{global_index}",
            )
            properties = {
                "modality": modality,
                "projection_profile": profile.name,
                "source_seed": str(SOURCE_SEED),
                "global_graph_index": str(global_index),
                "selection_ordinal": str(ordinal),
                "selection_seed": str(selection_seed),
                "sample_sha256": sample_sha256,
                "source_batch": str(source_path),
                "validity_policy": VALIDITY_POLICY,
                "canonical_smiles": smiles,
                "cycle_count_full_generated_graph": str(cycle_count),
                "max_cycle_length_full_generated_graph": str(max_cycle_length),
            }
            for key, value in properties.items():
                exported.SetProp(key, value)
            writer.write(exported)
    finally:
        writer.close()


def validate_profile_sdf(
    path: Path,
    count: int,
    modality: str,
    profile: Profile,
) -> None:
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
    molecules = list(supplier)
    if len(molecules) != count or any(molecule is None for molecule in molecules):
        raise RuntimeError(
            f"Invalid SDF record count for {path}: expected {count}, "
            f"read {len(molecules)}, invalid={sum(m is None for m in molecules)}"
        )
    required = {
        "modality",
        "projection_profile",
        "source_seed",
        "global_graph_index",
        "selection_ordinal",
        "selection_seed",
        "sample_sha256",
        "validity_policy",
        "canonical_smiles",
    }
    for molecule in molecules:
        if any(atom.GetAtomicNum() == 1 for atom in molecule.GetAtoms()):
            raise RuntimeError(f"Explicit hydrogen found while validating {path}")
        missing = required - set(molecule.GetPropNames())
        if missing:
            raise RuntimeError(f"Missing SDF properties in {path}: {sorted(missing)}")
        if molecule.GetProp("modality") != modality:
            raise RuntimeError(f"Wrong modality property in {path}")
        if molecule.GetProp("projection_profile") != profile.name:
            raise RuntimeError(f"Wrong profile property in {path}")
        Chem.SanitizeMol(molecule)


def atomic_publish(staging: Path, destination: Path) -> None:
    """Atomically publish a new tree, exchanging it with an existing tree."""
    if destination.exists():
        if destination.is_symlink() or not destination.is_dir():
            raise RuntimeError(
                f"Refusing to replace a non-directory or symlink: {destination}"
            )
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise RuntimeError(
                "Atomic directory replacement requires Linux renameat2 support"
            )
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(staging),
            -100,
            os.fsencode(destination),
            2,
        )
        if result != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), destination)
        shutil.rmtree(staging)
    else:
        os.replace(staging, destination)


def resolve_output(repo_root: Path, requested: Path) -> Path:
    destination = requested if requested.is_absolute() else repo_root / requested
    destination = destination.resolve()
    forbidden = {
        Path("/").resolve(),
        repo_root.resolve(),
        (repo_root / "samples").resolve(),
        (repo_root / "samples" / "final").resolve(),
    }
    if destination in forbidden:
        raise RuntimeError(f"Refusing unsafe output directory: {destination}")
    return destination


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    final_root = repo_root / "samples" / "final"
    if not final_root.is_dir():
        raise SystemExit(f"Final-suite directory does not exist: {final_root}")
    destination = resolve_output(repo_root, args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    fingerprint_cache: dict[tuple[int, int, int, int], str] = {}
    written = 0
    try:
        for modality in MODALITIES:
            rng = random.Random(args.selection_seed)
            for profile in PROFILES:
                batches, sample_sha256, source_path = load_source(
                    final_root, modality, profile, fingerprint_cache
                )
                selected, _ = select_profile_molecules(
                    batches, modality, profile, args.count, rng
                )
                target = (
                    staging
                    / modality
                    / profile.name
                    / f"seed_{SOURCE_SEED}"
                    / LEAF_NAME
                )
                write_profile_sdf(
                    target,
                    selected,
                    modality,
                    profile,
                    args.selection_seed,
                    sample_sha256,
                    source_path.relative_to(repo_root),
                )
                validate_profile_sdf(target, args.count, modality, profile)
                written += len(selected)

        leaves = sorted(staging.rglob(LEAF_NAME))
        if len(leaves) != len(MODALITIES) * len(PROFILES):
            raise RuntimeError(
                f"Expected {len(MODALITIES) * len(PROFILES)} SDF leaves, "
                f"found {len(leaves)}"
            )
        atomic_publish(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    print(
        f"Exported {written} valid molecules across "
        f"{len(MODALITIES) * len(PROFILES)} profiles to {destination}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
