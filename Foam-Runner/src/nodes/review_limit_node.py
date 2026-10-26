def review_limit_node(state):
    """Persist correction-budget exhaustion in graph state."""

    max_rewrites = state["config"].max_loop
    print(
        f"<review_limit>Maximum rewrite count ({max_rewrites}) reached."
        "</review_limit>"
    )
    return {"termination_reason": "max_review_loop_reached"}
