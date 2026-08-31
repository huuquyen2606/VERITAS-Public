import hashlib
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
_SHA256_NAME = re.compile(r"[0-9a-fA-F]{64}")


@dataclass
class PreparationSummary:
    source_dir: Path
    dest_dir: Path
    scanned_files: int = 0
    executable_candidates: int = 0
    copied_files: int = 0
    skipped_existing: int = 0
    skipped_non_executable: int = 0
    failed: int = 0
    cleared_existing: int = 0
    recreated_destination: bool = False


def _compute_sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as infile:
        for chunk in iter(lambda: infile.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _looks_like_pe(file_path: Path) -> bool:
    try:
        with file_path.open("rb") as infile:
            return infile.read(2) == b"MZ"
    except OSError:
        return False


def _is_executable_candidate(file_path: Path) -> bool:
    return file_path.suffix.lower() == ".exe" or _looks_like_pe(file_path)


def _clear_existing_hash_files(dest_dir: Path) -> int:
    removed = 0
    for item in dest_dir.iterdir():
        if item.is_file() and _SHA256_NAME.fullmatch(item.name):
            item.unlink()
            removed += 1
    return removed


def prepare_samples_from_dataset(
    source_dir: str,
    dest_dir: Optional[str] = None,
    excluded_dir_names: Optional[Sequence[str]] = None,
    clear_destination: bool = False,
    recreate_destination: bool = False,
) -> PreparationSummary:
    source = Path(source_dir).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Dataset directory not found: {source}")
    if not source.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {source}")

    destination = Path(dest_dir).expanduser().resolve() if dest_dir else DEFAULT_SAMPLES_DIR
    destination.mkdir(parents=True, exist_ok=True)

    excluded = {name.lower() for name in (excluded_dir_names or ("Benign",))}
    summary = PreparationSummary(source_dir=source, dest_dir=destination)

    if recreate_destination:
        if destination.exists():
            destination_str = str(destination)
            if destination_str in {"/", ""}:
                raise ValueError("Refusing to delete unsafe destination path.")
            shutil.rmtree(destination)
            summary.recreated_destination = True
        destination.mkdir(parents=True, exist_ok=True)
    elif clear_destination:
        summary.cleared_existing = _clear_existing_hash_files(destination)

    for root, dirs, files in os.walk(source):
        dirs[:] = [d for d in dirs if d.lower() not in excluded and not d.startswith(".")]

        for filename in files:
            if filename.startswith("."):
                continue

            sample_path = Path(root) / filename
            if not sample_path.is_file():
                continue

            summary.scanned_files += 1
            if not _is_executable_candidate(sample_path):
                summary.skipped_non_executable += 1
                continue

            summary.executable_candidates += 1
            try:
                sha256 = _compute_sha256(sample_path)
                output_path = destination / sha256

                if output_path.exists():
                    summary.skipped_existing += 1
                    continue

                shutil.copy2(sample_path, output_path)
                summary.copied_files += 1
            except Exception:
                summary.failed += 1

    return summary
