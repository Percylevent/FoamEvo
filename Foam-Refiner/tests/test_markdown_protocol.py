from foam_evo.roles.markdown_protocol import (
    AMBIGUOUS,
    parse_code_reviewer_verdict,
    parse_cluster_cases,
    parse_iteration_verdict,
    parse_plan_exit,
    parse_plan_reviewer_verdict,
)


def test_reviewer_scanner_uses_small_control_section():
    assert parse_code_reviewer_verdict("prose\n## Verdict\nREJECT_IMPL\n") == "REJECT_IMPL"
    assert parse_plan_reviewer_verdict("## Verdict\nREJECT_COMPLETENESS\n") == "REJECT_COMPLETENESS"
    assert parse_iteration_verdict("## Verdict\nCONVERGED\n") == "CONVERGED"
    assert parse_code_reviewer_verdict("## Verdict\nAPPROVE and REJECT_IMPL\n") == AMBIGUOUS


def test_plan_exit_and_cluster_protocol_are_deterministic():
    assert parse_plan_exit("# plan\n## Exit\nCODE_AND_KNOWLEDGE\n") == "CODE_AND_KNOWLEDGE"
    assert parse_plan_exit("# plan without a control section") is None
    assert parse_cluster_cases(
        "## Cluster mesh\n**Cases:** case-a, case-b\nreason\n## Cluster solver\nCases: case-c\n"
    ) == [("mesh", ["case-a", "case-b"]), ("solver", ["case-c"])]


def test_verdict_heading_prefix_is_not_control_section():
    markdown = "## Verdict Justification\nCONVERGED\n\n## Verdict\nCONVERGED\n"
    assert parse_iteration_verdict(markdown) == "CONVERGED"
