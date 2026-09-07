"""Generate and cross-evaluate the predefined QM9 edge-deletion matrix."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ConStruct.analysis.run_constraint_matrix import (
    QM9_NO_H_ATOM_DECODER,
    SCHEMA_VERSION,
    ArtifactMismatchError,
    _atomic_write_json,
    _normalized_checkpoint,
    _read_json,
    _retry,
    _source_fingerprint,
    _utc_timestamp,
    evaluate_batches,
    generate_profile,
    load_trimmed_batches,
    render_aggregate,
    update_manifest,
    validate_generation_artifacts,
)


COUNT_LEVELS = (None, 1, 2)
LENGTH_LEVELS = (None, 4, 5)


@dataclass(frozen=True)
class EdgeDeletionProfile:
    name: str
    max_rings: int | None
    max_ring_length: int | None

    @property
    def constraints(self) -> list[dict[str, Any]]:
        constraints = []
        if self.max_rings is not None:
            constraints.append(
                {"type": "ring_count_at_most", "max_rings": self.max_rings}
            )
        if self.max_ring_length is not None:
            constraints.append(
                {
                    "type": "ring_length_at_most",
                    "max_ring_length": self.max_ring_length,
                }
            )
        return constraints

    @property
    def projection_enabled(self) -> bool:
        return bool(self.constraints)


PROFILES = tuple(
    EdgeDeletionProfile(
        name=(
            f"at_most_count_{'disabled' if count is None else count}_"
            f"length_{'disabled' if length is None else length}"
        ),
        max_rings=count,
        max_ring_length=length,
    )
    for count in COUNT_LEVELS
    for length in LENGTH_LEVELS
)
PROFILE_BY_NAME = {profile.name: profile for profile in PROFILES}

METRICS = (
    ("molecular_validity_pct", "Molecular validity"),
    ("ring_count_at_most_1_pct", "Ring count ≤ 1"),
    ("ring_count_at_most_2_pct", "Ring count ≤ 2"),
    ("ring_length_at_most_4_pct", "Maximum cycle length ≤ 4"),
    ("ring_length_at_most_5_pct", "Maximum cycle length ≤ 5"),
)


def _cell_cache_matches(
    cell: dict[str, Any],
    profile: EdgeDeletionProfile,
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
    profile: EdgeDeletionProfile,
    checkpoint: Path,
    seed: int,
    requested_samples: int,
    attempts: int,
) -> dict[str, Any]:
    sample_dir = output_root / profile.name / f"seed_{seed}"
    error_path = matrix_root / "errors" / f"metrics_{profile.name}.json"
    metadata = validate_generation_artifacts(
        sample_dir, profile, checkpoint, seed, requested_samples
    )
    if metadata is None:
        error = RuntimeError(
            f"Generation artifacts are incomplete for {profile.name}."
        )
        _atomic_write_json(
            error_path,
            {
                "schema_version": SCHEMA_VERSION,
                "timestamp": _utc_timestamp(),
                "stage": "metrics",
                "profile": profile.name,
                "attempts": 1,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
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
        result = evaluate_batches(
            batches, QM9_NO_H_ATOM_DECODER, direction="at_most"
        )
        cell = {
            "schema_version": SCHEMA_VERSION,
            "timestamp": _utc_timestamp(),
            "profile": profile.name,
            "enforced_max_rings": profile.max_rings,
            "enforced_max_ring_length": profile.max_ring_length,
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


def _manifest_request(
    checkpoint: Path, experiment: str, seed: int, requested_samples: int
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "matrix_kind": "edge_deletion_at_most",
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
) -> None:
    request = _manifest_request(checkpoint, experiment, seed, requested_samples)
    path = matrix_root / "manifest.json"
    if path.exists():
        existing = _read_json(path)
        if {key: existing.get(key) for key in request} != request:
            raise ArtifactMismatchError(
                f"Matrix manifest {path} belongs to a different request."
            )
        return
    _atomic_write_json(
        path,
        {
            **request,
            "created_at": _utc_timestamp(),
            "generation_complete": False,
            "metrics_complete": False,
        },
    )


def run_matrix(args: argparse.Namespace) -> Path:
    repo_root = Path(args.repo_root).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    output_root = Path(args.output_root).expanduser().resolve()
    matrix_root = output_root / "matrix" / f"seed_{args.seed}"
    matrix_root.mkdir(parents=True, exist_ok=True)
    initialize_manifest(
        matrix_root, checkpoint, args.experiment, args.seed, args.samples
    )

    if args.phase in {"all", "generate"}:
        forced = set(args.force_profile or ())
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
        cells = {
            profile.name: evaluate_profile(
                output_root=output_root,
                matrix_root=matrix_root,
                profile=profile,
                checkpoint=checkpoint,
                seed=args.seed,
                requested_samples=args.samples,
                attempts=args.max_attempts,
            )
            for profile in PROFILES
        }
        error_path = matrix_root / "errors" / "aggregate.json"
        _retry(
            lambda _attempt: render_aggregate(
                matrix_root,
                cells,
                checkpoint,
                args.seed,
                args.samples,
                profiles=PROFILES,
                metrics=METRICS,
                count_levels=COUNT_LEVELS,
                length_levels=LENGTH_LEVELS,
                count_attribute="max_rings",
                length_attribute="max_ring_length",
                count_cell_field="enforced_max_rings",
                length_cell_field="enforced_max_ring_length",
                bound_label="maximum",
                comparison_symbol="≤",
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
            "Generate the nine predefined QM9 edge-deletion profiles and "
            "cross-evaluate five upper-bound metrics."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument(
        "--phase", choices=("all", "generate", "metrics"), default="all"
    )
    parser.add_argument(
        "--experiment",
        default="training/qm9_no_constraint_edge_absorbing",
    )
    parser.add_argument(
        "--output-root",
        default="samples/qm9_no_constraint_edge_absorbing",
    )
    parser.add_argument("--repo-root", default=str(repo_root))
    parser.add_argument(
        "--force-profile", action="append", choices=tuple(PROFILE_BY_NAME)
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.samples <= 0 or args.max_attempts <= 0:
        raise ValueError("--samples and --max-attempts must be positive.")
    print(f"Edge-deletion constraint matrix complete: {run_matrix(args)}")


if __name__ == "__main__":
    main()
