#!/usr/bin/env python3
"""Copy final-suite JSON artifacts into samples/jsons and zip that tree."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="replace an existing samples/jsons directory and samples/jsons.zip",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    samples = repo_root / "samples"
    source = samples / "final"
    destination = samples / "jsons"
    archive = samples / "jsons.zip"

    if not source.is_dir():
        raise SystemExit(f"Source directory does not exist: {source}")
    if not args.replace and (destination.exists() or archive.exists()):
        raise SystemExit(
            "samples/jsons or samples/jsons.zip already exists; "
            "rerun with --replace to rebuild them"
        )

    json_paths = sorted(path for path in source.rglob("*.json") if path.is_file())
    staging = Path(tempfile.mkdtemp(prefix=".jsons-", dir=samples))
    temporary_archive: Path | None = None
    try:
        for path in json_paths:
            target = staging / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)

        with tempfile.NamedTemporaryFile(
            prefix=".jsons-", suffix=".zip", dir=samples, delete=False
        ) as handle:
            temporary_archive = Path(handle.name)
        with ZipFile(temporary_archive, "w", compression=ZIP_DEFLATED) as bundle:
            for path in sorted(staging.rglob("*.json")):
                bundle.write(path, Path("jsons") / path.relative_to(staging))

        if destination.exists():
            if destination.is_dir() and not destination.is_symlink():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        if archive.exists():
            archive.unlink()
        os.replace(staging, destination)
        os.replace(temporary_archive, archive)
        temporary_archive = None
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if temporary_archive is not None and temporary_archive.exists():
            temporary_archive.unlink()

    print(f"Copied {len(json_paths)} JSON files to {destination}")
    print(f"Created {archive} ({archive.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
