import os
import random
import shutil
import tempfile

from obfumal.actions.base import Action
from obfumal.utils.subprocess_safe import run_command_safe


class UPXPack(Action):
    def __init__(self, upx_path: str = "upx"):
        self.upx_path = upx_path

    @property
    def name(self) -> str:
        return "UPXPack"

    def apply(self, bytez: bytes) -> bytes:
        if not shutil.which(self.upx_path) and not os.path.exists(self.upx_path):
            return bytez

        with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as tmp_in:
            tmp_in.write(bytez)
            input_path = tmp_in.name

        output_path = input_path + ".upx"

        try:
            options = ["--force", "--overlay=copy"]
            compression_level = random.randint(1, 9)
            options += [f"-{compression_level}"]
            options += [f"--compress-exports={random.randint(0, 1)}"]
            options += [f"--compress-icons={random.randint(0, 3)}"]
            options += [f"--compress-resources={random.randint(0, 1)}"]
            options += [f"--strip-relocs={random.randint(0, 1)}"]

            cmd = [self.upx_path] + options + [input_path, "-o", output_path]

            ok, _ = run_command_safe(cmd, timeout=15)
            if ok and os.path.exists(output_path):
                with open(output_path, "rb") as f:
                    packed_bytes = f.read()
                return packed_bytes
            return bytez
        except Exception:
            return bytez
        finally:
            if os.path.exists(input_path):
                os.remove(input_path)
            if os.path.exists(output_path):
                os.remove(output_path)
