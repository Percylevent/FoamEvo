# reviewer_node.py
from pydantic import BaseModel, Field
from typing import List
from services.review import review_error_logs, generate_rewrite_plan
from logger import log_review
from utils import retrieve_experience, format_experience_guidance


def reviewer_node(state):
    """
    Reviewer node: Reviews the error logs and provides analysis and suggestions
    for fixing the errors. This node only focuses on analysis, not file modification.
    """
    print("<reviewer>")
    if len(state["error_logs"]) == 0:
        print("No error to review.")
        print("</reviewer>")
        return state

    # Log error logs to review.log
    log_review(str(state["error_logs"]), "error_logs")

    # Retrieve against the error text, not the case description: by this point
    # the error is the better signal for which lesson applies.
    config = state.get("config")
    case_stats = state.get("case_stats") or {}
    error_text = "\n".join(str(e) for e in (state.get("error_logs") or []))
    experience_guidance = format_experience_guidance(
        retrieve_experience(
            f"case solver: {state.get('case_solver', '')}\n"
            f"error: {error_text[-2000:]}",
            topk=2,
            config=config,
            solver=state.get("case_solver", ""),
            fork=getattr(config, "openfoam_fork", "") if config else "",
        )
    )

    # Stateless review via service
    history_text = state.get("history_text") or []
    review_content, updated_history = review_error_logs(
        tutorial_reference=state.get('tutorial_reference', ''),
        foamfiles=state.get('foamfiles'),
        error_logs=state.get('error_logs'),
        user_requirement=state.get('user_requirement', ''),
        similar_case_advice=state.get('similar_case_advice'),
        experience_guidance=experience_guidance,
        history_text=history_text,
    )

    log_review(review_content, "review_analysis")

    rewrite_plan = generate_rewrite_plan(
        foamfiles=state.get('foamfiles'),
        error_logs=state.get('error_logs', []),
        review_analysis=review_content,
        user_requirement=state.get('user_requirement', ''),
    )
    log_review(str(rewrite_plan), "rewrite_plan")

    print("</reviewer>")

    return {
        "loop_count": state.get("loop_count", 0) + 1,
        "history_text": updated_history,
        "review_analysis": review_content,
        "rewrite_plan": rewrite_plan,
        "input_writer_mode": "rewrite",
    }
