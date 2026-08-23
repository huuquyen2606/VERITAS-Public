from __future__ import annotations

import logging
import random
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Set

try:
    from .base_action import Action as FileAction
except ImportError:
    from base_action import Action as FileAction


logger = logging.getLogger(__name__)


class PackerTransform(FileAction):
    """Disk-backed UPX adapter for offline/path-based experiments."""

    def __init__(
        self,
        level: Optional[int] = None,
        upx_path: Optional[str] = None,
        extra_args: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        timeout: int = 60,
    ) -> None:
        self.level = self._validate_level(level) if level is not None else None
        self.upx_path = str(Path(upx_path).expanduser()) if upx_path else self._resolve_upx_path()
        self.extra_args = [str(arg) for arg in (extra_args or [])]
        self.output_dir = Path(output_dir).expanduser() if output_dir else None
        self.timeout = timeout

    @property
    def name(self) -> str:
        return "PACKER_TRANSFORM"

    @staticmethod
    def _validate_level(level: int) -> int:
        value = int(level)
        if not 1 <= value <= 9:
            raise ValueError(f"UPX compression level must be in 1..9, got {level}")
        return value

    @staticmethod
    def _default_upx_path() -> Optional[Path]:
        repo_root = Path(__file__).resolve().parents[1]
        upx_root = repo_root / "utils" / "upx"
        candidates = sorted(upx_root.glob("*/upx"), reverse=True)
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        return None

    @classmethod
    def _resolve_upx_path(cls) -> str:
        local_upx = cls._default_upx_path()
        if local_upx is not None:
            return str(local_upx)
        return "upx"

    def _choose_level(self) -> int:
        if self.level is not None:
            return self.level
        return random.randint(1, 9)

    def check_precondition(self, file_path: str) -> bool:
        """Return whether this action can be applied to ``file_path``.
        Only applies if the file is not already UPX packed."""
        path = Path(file_path).expanduser()
        if not path.is_file():
            return False

        return not self._has_upx_signature(path)

    def apply(self, file_path: str) -> str:
        """Apply pack/unpack to ``file_path`` and return the transformed path."""
        source_path = Path(file_path).expanduser().resolve()
        if not self.check_precondition(str(source_path)):
            raise RuntimeError(
                f"precondition failed for {self.name}: "
                f"file={source_path} (might already be packed)"
            )

        level = self._choose_level()
        destination_path = self._make_output_path(source_path, level)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if destination_path == source_path:
            raise ValueError("packer output_path must differ from input_path")

        cmd = self._build_command(source_path, destination_path, level)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.timeout,
            check=False,
        )
        if result.returncode != 0:
            stderr = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(f"packer command failed with exit code {result.returncode}: {stderr}")
        if not destination_path.is_file() or destination_path.stat().st_size == 0:
            raise RuntimeError(f"packer did not create a valid output file: {destination_path}")

        return str(destination_path)

    def apply_bytes(self, bytez: bytes) -> bytes:
        """Disk-backed bytes adapter; unsuitable for the in-memory RL hot path."""
        if not bytez:
            return bytez

        with tempfile.TemporaryDirectory(prefix="packer_transform_input_") as tmpdir:
            input_path = Path(tmpdir) / "sample.exe"
            input_path.write_bytes(bytez)
            output_path = self.apply(str(input_path))
            return Path(output_path).read_bytes()

    def _make_output_path(self, input_path: Path, level: int) -> Path:
        target_dir = self.output_dir or input_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)

        suffix = input_path.suffix or ".exe"
        stem = input_path.stem or "sample"
        base_name = f"{stem}.upx{level}{suffix}"

        candidate = (target_dir / base_name).resolve()
        if not candidate.exists():
            return candidate

        idx = 1
        while True:
            candidate = (target_dir / f"{Path(base_name).stem}.{idx}{suffix}").resolve()
            if not candidate.exists():
                return candidate
            idx += 1

    def _build_command(self, input_path: Path, output_path: Path, level: int) -> List[str]:
        if not shutil.which(self.upx_path) and not Path(self.upx_path).exists():
            raise FileNotFoundError(f"UPX executable not found: {self.upx_path}")

        cmd = [
            self.upx_path,
            "--force",
            "--overlay=copy",
            f"-{level}",
        ]
        cmd.extend(self.extra_args)
        cmd.extend([str(input_path), "-o", str(output_path)])
        return cmd

    def _has_upx_signature(self, path: Path) -> bool:
        section_names = self._section_names(path)
        if {"UPX0", "UPX1"}.issubset(section_names):
            return True

        try:
            data = path.read_bytes()
        except OSError:
            return False
        return b"UPX!" in data or b"UPX0" in data or b"UPX1" in data

    @staticmethod
    def _section_names(path: Path) -> Set[str]:
        try:
            data = path.read_bytes()
            if len(data) < 0x40 or data[:2] != b"MZ":
                return set()

            pe_offset = int.from_bytes(data[0x3C:0x40], "little")
            coff_offset = pe_offset + 4
            if pe_offset <= 0 or len(data) < coff_offset + 20 or data[pe_offset:pe_offset + 4] != b"PE\x00\x00":
                return set()

            section_count = int.from_bytes(data[coff_offset + 2:coff_offset + 4], "little")
            optional_header_size = int.from_bytes(data[coff_offset + 16:coff_offset + 18], "little")
            section_offset = coff_offset + 20 + optional_header_size

            names = set()
            for idx in range(section_count):
                name_offset = section_offset + idx * 40
                if len(data) < name_offset + 8:
                    break
                raw_name = data[name_offset:name_offset + 8].split(b"\x00", 1)[0]
                if raw_name:
                    names.add(raw_name.decode("latin-1", errors="ignore").upper())
            return names
        except Exception:
            logger.debug("Could not parse PE sections for %s", path, exc_info=True)
            return set()
