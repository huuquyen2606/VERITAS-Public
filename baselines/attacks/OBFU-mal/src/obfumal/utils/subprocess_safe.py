import subprocess
from typing import List, Optional, Tuple


def run_command_safe(
    cmd: List[str], timeout: int = 10, cwd: Optional[str] = None
) -> Tuple[bool, Optional[subprocess.CompletedProcess]]:
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=timeout, cwd=cwd)
        return result.returncode == 0, result
    except Exception:
        return False, None
