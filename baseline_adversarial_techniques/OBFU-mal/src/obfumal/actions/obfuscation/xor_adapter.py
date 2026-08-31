import os
import shutil
import tempfile
import sys
from pathlib import Path

from obfumal.actions.base import Action
from obfumal.utils.subprocess_safe import run_command_safe


class DarkarmourXOR(Action):
    def __init__(self, loop_level: int, darkarmour_path: str = "darkarmour"):
        self.loop_level = loop_level
        if darkarmour_path == "darkarmour":
            candidate = Path(__file__).resolve().parents[4] / "darkarmour-master" / "darkarmour-master" / "darkarmour.py"
            self.darkarmour_path = str(candidate) if candidate.exists() else darkarmour_path
        else:
            self.darkarmour_path = darkarmour_path

    @property
    def name(self) -> str:
        return f"DarkarmourXOR_EL{self.loop_level}"

    def apply(self, bytez: bytes) -> bytes:
        if not shutil.which(self.darkarmour_path) and not os.path.exists(self.darkarmour_path):
            return bytez

        with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as tmp_in:
            tmp_in.write(bytez)
            input_path = tmp_in.name

        output_path = input_path + ".xor.exe"

        try:
            cmd_cwd = None
            use_jmp_loader = False
            if self.darkarmour_path.lower().endswith(".py"):
                script_path = os.path.abspath(self.darkarmour_path)
                cmd = [sys.executable, script_path]
                cmd_cwd = os.path.dirname(script_path) or None
                use_jmp_loader = True
            else:
                cmd = [self.darkarmour_path]
            cmd += [
                "-f",
                input_path,
                "--encrypt",
                "xor",
                "--loop",
                str(self.loop_level),
                "-o",
                output_path,
            ]
            if use_jmp_loader:
                cmd.append("-j")

            ok, _ = run_command_safe(cmd, timeout=30, cwd=cmd_cwd)
            if ok and os.path.exists(output_path):
                with open(output_path, "rb") as f:
                    obfuscated_bytes = f.read()
                return obfuscated_bytes
            return bytez
        except Exception:
            return bytez
        finally:
            if os.path.exists(input_path):
                os.remove(input_path)
            if os.path.exists(output_path):
                os.remove(output_path)
