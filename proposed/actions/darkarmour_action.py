from __future__ import annotations

import logging
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple


logger = logging.getLogger(__name__)


class DarkarmourXORTransformAction:
    """Disk-backed whole-file XOR transform via bundled darkarmour CLI."""

    name = "XOR_ENCRYPTION"
    min_iterations = 1
    max_iterations = 3

    def __init__(
        self,
        iterations: Optional[int] = None,
        darkarmour_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        timeout: int = 60,
        isolate_tool_workdir: bool = True,
    ) -> None:
        self.iterations = (
            self._validate_iterations(iterations) if iterations is not None else None
        )
        self.darkarmour_path = (
            Path(darkarmour_path).expanduser()
            if darkarmour_path
            else self._default_darkarmour_path()
        )
        self.output_dir = Path(output_dir).expanduser() if output_dir else None
        self.timeout = timeout
        self.isolate_tool_workdir = isolate_tool_workdir

    @staticmethod
    def _default_darkarmour_path() -> Path:
        repo_root = Path(__file__).resolve().parents[1]
        return repo_root / "utils" / "darkarmour-master" / "darkarmour.py"

    @classmethod
    def _validate_iterations(cls, iterations: int) -> int:
        value = int(iterations)
        if not cls.min_iterations <= value <= cls.max_iterations:
            raise ValueError(
                f"darkarmour XOR iterations must be in "
                f"{cls.min_iterations}..{cls.max_iterations}, got {iterations}"
            )
        return value

    def _choose_iterations(self, iterations: Optional[int]) -> int:
        if iterations is not None:
            return self._validate_iterations(iterations)
        if self.iterations is not None:
            return self.iterations
        return random.randint(self.min_iterations, self.max_iterations)

    def _make_output_path(
        self,
        input_path: Path,
        iterations: int,
        output_dir: Optional[str],
        output_path: Optional[str],
    ) -> Path:
        if output_path:
            return Path(output_path).expanduser().resolve()

        target_dir = Path(output_dir).expanduser() if output_dir else self.output_dir
        if target_dir is None:
            target_dir = input_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)

        suffix = input_path.suffix or ".exe"
        stem = input_path.stem or "sample"
        candidate = target_dir / f"{stem}.xor{iterations}{suffix}"
        if not candidate.exists():
            return candidate.resolve()

        idx = 1
        while True:
            candidate = target_dir / f"{stem}.xor{iterations}.{idx}{suffix}"
            if not candidate.exists():
                return candidate.resolve()
            idx += 1

    def _tool_script_and_cwd(self) -> Tuple[Path, Path, Optional[tempfile.TemporaryDirectory]]:
        script_path = self.darkarmour_path.resolve()
        if not script_path.exists():
            raise FileNotFoundError(f"darkarmour script not found: {script_path}")

        tool_root = script_path.parent
        if not self.isolate_tool_workdir:
            return script_path, tool_root, None

        tmp = tempfile.TemporaryDirectory(prefix="darkarmour_xor_")
        isolated_root = Path(tmp.name) / "darkarmour-master"
        shutil.copytree(tool_root, isolated_root)
        return isolated_root / script_path.name, isolated_root, tmp

    def transform_path(
        self,
        input_path: str,
        iterations: Optional[int] = None,
        output_dir: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """Run darkarmour over ``input_path`` and return the new PE path."""
        source_path = Path(input_path).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"input PE not found: {source_path}")

        level = self._choose_iterations(iterations)
        destination_path = self._make_output_path(source_path, level, output_dir, output_path)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if destination_path == source_path:
            raise ValueError("darkarmour XOR output_path must differ from input_path")

        if shutil.which("x86_64-w64-mingw32-gcc") is None:
            raise RuntimeError("darkarmour requires x86_64-w64-mingw32-gcc to build the wrapper PE")

        script_path, cwd, tmp = self._tool_script_and_cwd()
        try:
            cmd = [
                sys.executable,
                str(script_path),
                "-f",
                str(source_path),
                "--encrypt",
                "xor",
                "--jmp",
                "--loop",
                str(level),
                "--outfile",
                str(destination_path),
            ]
            result = subprocess.run(
                cmd,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
            if result.returncode != 0:
                stderr = (result.stderr or result.stdout or "").strip()
                raise RuntimeError(f"darkarmour failed with exit code {result.returncode}: {stderr}")
            if not destination_path.is_file() or destination_path.stat().st_size == 0:
                raise RuntimeError(f"darkarmour did not create a valid output file: {destination_path}")
            return str(destination_path)
        finally:
            if tmp is not None:
                tmp.cleanup()

    def apply(self, bytez: bytes, iterations: Optional[int] = None) -> bytes:
        """Disk-backed bytes adapter; unsuitable for the in-memory RL hot path."""
        if not bytez:
            return bytez

        with tempfile.TemporaryDirectory(prefix="xor_transform_input_") as tmpdir:
            input_path = Path(tmpdir) / "sample.exe"
            input_path.write_bytes(bytez)
            output_path = self.transform_path(
                str(input_path),
                iterations=iterations,
                output_dir=tmpdir,
            )
            return Path(output_path).read_bytes()
