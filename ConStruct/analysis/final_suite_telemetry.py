"""Compact, batch-aggregated telemetry for the final constraint suite."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


TELEMETRY_SCHEMA_VERSION = 1


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(data)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _present(z) -> torch.Tensor:
    edges = z.E.argmax(dim=-1) if z.E.dim() == 4 else z.E
    return edges != 0


def _upper_counts(mask: torch.Tensor) -> int:
    n = mask.shape[-1]
    upper = torch.triu(torch.ones((n, n), dtype=torch.bool, device=mask.device), diagonal=1)
    return int((mask & upper.unsqueeze(0)).sum().item())


class FinalSuiteTelemetry:
    """Collect exact topology transitions and a deterministic state cohort.

    The collector intentionally stores only a small cohort of full states. Edge
    transition and projector counters are aggregated over every generated graph.
    """

    def __init__(
        self,
        output_dir: str | Path,
        requested_samples: int,
        cohort_size: int,
        seed: int,
        reverse_operation: str,
        diffusion_steps: int,
        max_nodes: int = 9,
        collapse_charges=None,
    ) -> None:
        if reverse_operation not in {"insertion", "deletion"}:
            raise ValueError(f"Unknown reverse operation: {reverse_operation}")
        self.output_dir = Path(output_dir)
        self.requested_samples = int(requested_samples)
        self.cohort_size = min(int(cohort_size), self.requested_samples)
        self.reverse_operation = reverse_operation
        self.diffusion_steps = int(diffusion_steps)
        self.max_nodes = int(max_nodes)
        self.collapse_charges = collapse_charges
        rng = np.random.default_rng(int(seed))
        self.cohort_indices = np.sort(
            rng.choice(self.requested_samples, size=self.cohort_size, replace=False)
        ).astype(np.int64)
        self._batches: list[dict[str, Any]] = []
        self._active: dict[str, Any] | None = None
        self._steps: dict[int, dict[str, int]] = {}

    def begin_batch(self, z_t, global_offset: int) -> None:
        batch_size = int(z_t.X.shape[0])
        stop = global_offset + batch_size
        selected = self.cohort_indices[
            (self.cohort_indices >= global_offset) & (self.cohort_indices < stop)
        ]
        local = selected - int(global_offset)
        self._active = {
            "global_indices": selected,
            "local_indices": local,
            "timesteps": [],
            "X": [],
            "charges": [],
            "E": [],
            "node_mask": [],
        }
        self._capture_state(z_t, self.diffusion_steps)

    def _capture_state(self, z_t, timestep: int) -> None:
        if self._active is None or len(self._active["local_indices"]) == 0:
            return
        indices = torch.as_tensor(
            self._active["local_indices"], dtype=torch.long, device=z_t.X.device
        )
        collapsed = z_t.collapse(self.collapse_charges)
        self._active["timesteps"].append(int(timestep))
        self._active["X"].append(collapsed.X.index_select(0, indices).cpu().numpy())
        charges = collapsed.charges
        if charges is None or charges.numel() == 0:
            charges = torch.zeros_like(collapsed.X)
        self._active["charges"].append(charges.index_select(0, indices).cpu().numpy())
        self._active["E"].append(collapsed.E.index_select(0, indices).cpu().numpy())
        self._active["node_mask"].append(
            z_t.node_mask.index_select(0, indices).cpu().numpy().astype(np.bool_)
        )

    def record_transition(
        self,
        z_before,
        z_raw,
        z_after,
        timestep: int,
        blocked_attempts: int = 0,
        newly_blocked_pairs: int = 0,
    ) -> None:
        before = _present(z_before)
        raw = _present(z_raw)
        after = _present(z_after)
        proposed_additions = _upper_counts(~before & raw)
        proposed_deletions = _upper_counts(before & ~raw)
        realized_additions = _upper_counts(~before & after)
        realized_deletions = _upper_counts(before & ~after)
        expected_proposals = (
            proposed_additions if self.reverse_operation == "insertion" else proposed_deletions
        )
        expected_realized = (
            realized_additions if self.reverse_operation == "insertion" else realized_deletions
        )
        values = self._steps.setdefault(
            int(timestep),
            {
                "graphs": 0,
                "proposals": 0,
                "blocked_attempts": 0,
                "newly_blocked_graph_edge_pairs": 0,
                "repeated_blocked_attempts": 0,
                "realized_actions": 0,
                "edge_additions": 0,
                "edge_deletions": 0,
            },
        )
        values["graphs"] += int(before.shape[0])
        values["proposals"] += expected_proposals
        values["blocked_attempts"] += int(blocked_attempts)
        values["newly_blocked_graph_edge_pairs"] += int(newly_blocked_pairs)
        values["repeated_blocked_attempts"] += max(
            int(blocked_attempts) - int(newly_blocked_pairs), 0
        )
        values["realized_actions"] += expected_realized
        values["edge_additions"] += realized_additions
        values["edge_deletions"] += realized_deletions
        self._capture_state(z_after, int(timestep))

    def end_batch(self) -> None:
        if self._active is not None and len(self._active["global_indices"]):
            self._batches.append(self._active)
        self._active = None

    @staticmethod
    def projector_snapshot(projector) -> tuple[int, int]:
        if projector is None:
            return 0, 0
        total = int(getattr(projector, "total_blocked", 0))
        unique = sum(
            len(values) for values in getattr(projector, "blocked_edges", {}).values()
        )
        return total, unique

    def _pad(self, value: np.ndarray, dimensions: int, fill: int = -1) -> np.ndarray:
        shape = list(value.shape)
        target = [shape[0], shape[1]] + [self.max_nodes] * dimensions
        result = np.full(target, fill, dtype=value.dtype)
        if dimensions == 1:
            result[:, :, : shape[2]] = value
        else:
            result[:, :, : shape[2], : shape[3]] = value
        return result

    def finish(self) -> dict[str, Any]:
        if self._active is not None:
            self.end_batch()
        ordered = sorted(self._batches, key=lambda item: int(item["global_indices"][0]))
        if ordered:
            timesteps = np.asarray(ordered[0]["timesteps"], dtype=np.int16)
            for item in ordered[1:]:
                if item["timesteps"] != ordered[0]["timesteps"]:
                    raise RuntimeError("Telemetry batches captured different denoising states")
            indices = np.concatenate([item["global_indices"] for item in ordered])
            X = np.concatenate(
                [self._pad(np.stack(item["X"]), 1) for item in ordered], axis=1
            )
            charges = np.concatenate(
                [self._pad(np.stack(item["charges"]), 1, fill=0) for item in ordered], axis=1
            )
            E = np.concatenate(
                [self._pad(np.stack(item["E"]), 2) for item in ordered], axis=1
            )
            node_mask = np.concatenate(
                [self._pad(np.stack(item["node_mask"]), 1, fill=0) for item in ordered], axis=1
            ).astype(np.bool_)
        else:
            timesteps = np.empty((0,), dtype=np.int16)
            indices = np.empty((0,), dtype=np.int64)
            X = charges = E = node_mask = np.empty((0,), dtype=np.int8)

        telemetry_path = self.output_dir / "generation_telemetry.npz"
        telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=telemetry_path.parent, suffix=".npz", delete=False
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(
                handle,
                schema_version=np.asarray(TELEMETRY_SCHEMA_VERSION),
                global_indices=indices,
                timesteps=timesteps,
                X=X,
                charges=charges,
                E=E,
                node_mask=node_mask,
            )
        os.replace(temporary, telemetry_path)

        per_step = []
        for timestep in sorted(self._steps, reverse=True):
            entry = {"timestep": timestep, **self._steps[timestep]}
            proposals = entry["proposals"]
            entry["block_rate"] = entry["blocked_attempts"] / proposals if proposals else 0.0
            per_step.append(entry)
        totals = {
            key: sum(entry[key] for entry in per_step)
            for key in (
                "proposals",
                "blocked_attempts",
                "newly_blocked_graph_edge_pairs",
                "repeated_blocked_attempts",
                "realized_actions",
                "edge_additions",
                "edge_deletions",
            )
        }
        totals["block_rate"] = (
            totals["blocked_attempts"] / totals["proposals"] if totals["proposals"] else 0.0
        )
        blocked_payload = {
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "reverse_operation": self.reverse_operation,
            "counts_include_repeated_attempts": True,
            "generated_graphs": self.requested_samples,
            "per_step": per_step,
            "totals": totals,
        }
        atomic_json(self.output_dir / "blocked_actions.json", blocked_payload)
        return {
            "telemetry_file": telemetry_path.name,
            "telemetry_cohort_size": int(len(indices)),
            "telemetry_states": int(len(timesteps)),
            "blocked_actions_file": "blocked_actions.json",
        }
