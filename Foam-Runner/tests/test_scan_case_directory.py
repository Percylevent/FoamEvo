"""Unit tests for scan_case_directory numeric-directory filtering."""

from __future__ import annotations

from pathlib import Path

import pytest

from utils import scan_case_directory


@pytest.fixture
def foam_case(tmp_path: Path) -> Path:
    """Create a minimal OpenFOAM case with time-step output directories."""
    # Standard configuration directories
    (tmp_path / "system").mkdir()
    (tmp_path / "system" / "controlDict").write_text("// controlDict")
    (tmp_path / "system" / "fvSchemes").write_text("// fvSchemes")
    (tmp_path / "system" / "fvSolution").write_text("// fvSolution")

    (tmp_path / "constant").mkdir()
    (tmp_path / "constant" / "transportProperties").write_text("// transport")

    (tmp_path / "0").mkdir()
    (tmp_path / "0" / "U").write_text("// velocity field")
    (tmp_path / "0" / "p").write_text("// pressure field")

    # Solver-generated time-step output directories
    for t in ["0.05", "0.1", "0.15", "0.2", "0.25", "0.3", "0.35"]:
        d = tmp_path / t
        d.mkdir()
        (d / "U").write_text("uniform (0 0 0);\n" * 1600)
        (d / "p").write_text("uniform 0;\n" * 1600)

    # An integer time-step directory (also solver output)
    (tmp_path / "1").mkdir()
    (tmp_path / "1" / "U").write_text("uniform (0 0 0);\n" * 1600)

    return tmp_path


def test_excludes_numeric_timestep_directories(foam_case: Path) -> None:
    """Numeric time-step directories must not appear in the result."""
    result = scan_case_directory(str(foam_case))

    assert "system" in result
    assert "constant" in result
    assert "0" in result

    # None of the numeric time-step directories should be present
    for numeric_dir in ["0.05", "0.1", "0.15", "0.2", "0.25", "0.3", "0.35", "1"]:
        assert numeric_dir not in result, f"Time-step directory '{numeric_dir}' should be excluded"


def test_preserves_zero_directory(foam_case: Path) -> None:
    """The '0' directory (initial conditions) must be preserved."""
    result = scan_case_directory(str(foam_case))
    assert "0" in result
    assert "U" in result["0"]
    assert "p" in result["0"]


def test_includes_non_numeric_directories(foam_case: Path) -> None:
    """Non-numeric directories like postProcessing should still be included."""
    pp = foam_case / "postProcessing"
    pp.mkdir()
    (pp / "probes.dat").write_text("# probe data")

    result = scan_case_directory(str(foam_case))
    assert "postProcessing" in result
    assert "probes.dat" in result["postProcessing"]


def test_excludes_scientific_notation_directories(foam_case: Path) -> None:
    """Edge case: directories named in scientific notation (e.g., '1e-05')."""
    sci = foam_case / "1e-05"
    sci.mkdir()
    (sci / "U").write_text("// data")

    result = scan_case_directory(str(foam_case))
    assert "1e-05" not in result
