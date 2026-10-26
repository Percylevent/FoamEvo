import json

from foam_evo.locator.context import build_locator_context


def test_locator_context_copies_complete_inputs_and_diagnostics(tmp_path):
    source = tmp_path / "record" / "case"
    (source / "inputs" / "nested").mkdir(parents=True)
    (source / "inputs" / "nested" / "case.dict").write_text("input\n")
    (source / "stdout.log").write_text("stdout\n")
    (source / "workflow.log").write_text("workflow\n")
    (source / "log.solver").write_text("solver\n")
    (source / "evaluation.json").write_text(json.dumps({
        "schema_version": 2,
        "passed": False,
        "failure_summary": "mesh failed",
        "checks": [{"check_id": "mesh", "passed": False}],
    }))
    root = tmp_path / "evo"
    build_locator_context(source_cases={"case": source}, case_ids=["case"], root=root)
    index = json.loads((root / "context" / "case-index.json").read_text())
    assert index[0]["summary"]["failed_checks"] == ["mesh"]
    assert (root / "context" / "cases" / "case-001" / "inputs" / "nested" / "case.dict").is_file()
    assert (root / "context" / "cases" / "case-001" / "artifacts" / "log.solver").is_file()
    assert not any(str(value).startswith(str(source)) for value in index[0]["files"].values())
