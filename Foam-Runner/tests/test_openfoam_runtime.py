from evaluation import evaluate_case
from openfoam_runtime import detect_execution_errors, update_error_progress


def test_fatal_error_keeps_diagnostic_context(tmp_path) -> None:
    (tmp_path / "log.pimpleFoam").write_text(
        "OpenFOAM startup\n\n"
        "--> FOAM FATAL IO ERROR:\n"
        "keyword viscosityModel is undefined in dictionary physicalProperties\n\n"
        "file: constant/physicalProperties\n"
        "FOAM exiting\n",
        encoding="utf-8",
    )

    errors = detect_execution_errors(case_dir=str(tmp_path))
    solver_error = next(item for item in errors if item["source"] == "log.pimpleFoam")

    assert "FOAM FATAL IO ERROR" in solver_error["excerpt"]
    assert "keyword viscosityModel is undefined" in solver_error["excerpt"]
    assert "constant/physicalProperties" in solver_error["excerpt"]


def test_infrastructure_check_uses_final_runner_attempt(tmp_path) -> None:
    (tmp_path / "Allrun").write_text("#!/bin/sh\n", encoding="utf-8")
    (tmp_path / "log.pimpleFoam").write_text("Time = 1\nEnd\n", encoding="utf-8")
    stdout = (
        "<runner>\nOpenFOAM command not found (exit 127)\n</runner>\n"
        "<runner>\nAllrun executed successfully without errors.\n</runner>\n"
        "<workflow_end>Workflow completed successfully!</workflow_end>\n"
    )
    result = evaluate_case(
        case_dir=tmp_path,
        return_code=0,
        stdout=stdout,
        stderr="",
        task={"expected": {"must_run_openfoam": True, "solver": "pimpleFoam"}},
    )
    checks = {item["check_id"]: item for item in result["checks"]}
    assert checks["infrastructure_healthy"]["passed"] is True
    assert result["passed"] is True


def test_infrastructure_progress_is_projected_into_state() -> None:
    state = {"consecutive_infrastructure_failures": 1, "error_signature": None, "repeated_error_count": 0}
    progress = update_error_progress(state, [{"source": "infrastructure", "excerpt": "exit 127"}])
    assert progress["consecutive_infrastructure_failures"] == 2
