"""Normalize legacy and composable structural-constraint configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


THRESHOLD_FIELDS = {
    "ring_count_at_most": "max_rings",
    "ring_length_at_most": "max_ring_length",
    "ring_count_at_least": "min_rings",
    "ring_length_at_least": "min_ring_length",
}
LEGACY_DEFAULTS = {
    "ring_count_at_most": 0,
    "ring_length_at_most": 6,
    "ring_count_at_least": 1,
    "ring_length_at_least": 3,
}
COMPOSABLE_AT_LEAST = {"ring_count_at_least", "ring_length_at_least"}


@dataclass(frozen=True)
class ConstraintSpec:
    type: str
    value: int | None = None

    @property
    def threshold_field(self) -> str | None:
        return THRESHOLD_FIELDS.get(self.type)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"type": self.type}
        if self.threshold_field is not None:
            result[self.threshold_field] = self.value
        return result

    def to_legacy_dict(self) -> dict[str, Any]:
        return {"type": self.type, "value": self.value}


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _validate_threshold(kind: str, raw_value: Any, *, legacy: bool) -> int:
    if isinstance(raw_value, bool) or not isinstance(raw_value, int):
        raise ValueError(f"{THRESHOLD_FIELDS[kind]} must be an integer.")
    if kind == "ring_length_at_least" and raw_value < 3:
        raise ValueError("min_ring_length must be at least 3.")
    if kind == "ring_count_at_least" and raw_value < (0 if legacy else 1):
        qualifier = "non-negative" if legacy else "positive"
        raise ValueError(f"min_rings must be {qualifier}.")
    if kind in {"ring_count_at_most", "ring_length_at_most"} and raw_value < 0:
        raise ValueError(f"{THRESHOLD_FIELDS[kind]} must be non-negative.")
    return int(raw_value)


def _new_constraint_specs(entries: Iterable[Any]) -> list[ConstraintSpec]:
    specs = []
    for index, entry in enumerate(entries):
        kind = _get(entry, "type")
        if kind not in COMPOSABLE_AT_LEAST:
            raise ValueError(
                f"model.constraints[{index}].type must be one of "
                f"{sorted(COMPOSABLE_AT_LEAST)}, got {kind!r}."
            )
        field = THRESHOLD_FIELDS[kind]
        raw_value = _get(entry, field)
        if raw_value is None:
            raise ValueError(f"model.constraints[{index}] requires {field}.")
        specs.append(
            ConstraintSpec(kind, _validate_threshold(kind, raw_value, legacy=False))
        )
    return specs


def resolve_constraints(model_cfg: Any) -> tuple[ConstraintSpec, ...]:
    """Return one validated, deduplicated constraint set for all consumers."""
    configured = _get(model_cfg, "constraints", []) or []
    specs = _new_constraint_specs(configured)

    legacy_kind = _get(model_cfg, "rev_proj")
    if legacy_kind not in (None, ""):
        field = THRESHOLD_FIELDS.get(legacy_kind)
        if field is None:
            legacy_spec = ConstraintSpec(str(legacy_kind), None)
        else:
            raw_value = _get(model_cfg, field, LEGACY_DEFAULTS[legacy_kind])
            legacy_spec = ConstraintSpec(
                str(legacy_kind),
                _validate_threshold(str(legacy_kind), raw_value, legacy=True),
            )
        specs.insert(0, legacy_spec)

    by_type: dict[str, ConstraintSpec] = {}
    for spec in specs:
        previous = by_type.get(spec.type)
        if previous is not None and previous != spec:
            raise ValueError(
                f"Conflicting thresholds for {spec.type}: "
                f"{previous.value} and {spec.value}."
            )
        by_type[spec.type] = spec

    normalized = tuple(by_type.values())
    if len(normalized) > 1 and not {spec.type for spec in normalized}.issubset(
        COMPOSABLE_AT_LEAST
    ):
        raise ValueError(
            "Multiple constraints are supported only for ring_count_at_least and "
            "ring_length_at_least."
        )
    return normalized


def constraints_metadata(specs: Iterable[ConstraintSpec]) -> list[dict[str, Any]]:
    return [spec.to_dict() for spec in specs]
