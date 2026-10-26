"""Engine-owned source gate with isolated nested pytest artifacts."""
from __future__ import annotations
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from ..models import SourceTestResult

def run_source_tests(worktree: str | Path, *, timeout_s: float = 1800.0) -> SourceTestResult:
    root = Path(worktree).resolve()
    environment = os.environ.copy()
    environment.pop("PYTEST_ADDOPTS", None)
    environment["PYTHONPATH"] = str(root / "src")
    output = []
    with tempfile.TemporaryDirectory(prefix="foam-refiner-source-") as temporary:
        environment["PYTHONPYCACHEPREFIX"] = str(Path(temporary) / "pycache")
        commands = [[sys.executable, "-m", "compileall", "-q", "src", "foambench_main.py"],
                    [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests", "--basetemp=" + str(Path(temporary) / "pytest")]]
        for command in commands:
            try:
                result = subprocess.run(command, cwd=root, env=environment, text=True,
                                        capture_output=True, timeout=timeout_s, check=False)
            except subprocess.TimeoutExpired as exc:
                return SourceTestResult(passed=False, command=command, output=str(exc), timed_out=True)
            output.append(f"$ {' '.join(command)}\n{result.stdout}\n{result.stderr}")
            if result.returncode != 0:
                return SourceTestResult(passed=False, return_code=result.returncode, command=command, output="\n".join(output))
    return SourceTestResult(passed=True, return_code=0, command=commands[-1], output="\n".join(output))
