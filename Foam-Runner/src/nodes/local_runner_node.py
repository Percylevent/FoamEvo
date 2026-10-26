# runner_node.py
from typing import List
import os
from pydantic import BaseModel, Field
import re

from services.run_local import run_allrun_and_collect_errors
from logger import log_review, log_correction_round
from openfoam_runtime import update_error_progress


def local_runner_node(state):
    """
    Runner node: Execute an Allrun script, and check for errors.
    On error, update state.error_command and state.error_content.
    """
    config = state["config"]
    case_dir = state["case_dir"]
    max_time_limit = state["config"].max_time_limit

    print("<runner>")

    # Execute using service and collect errors
    error_logs = run_allrun_and_collect_errors(case_dir, max_time_limit)

    if len(error_logs) > 0:
        print("Errors detected in the Allrun execution.")
        log_review(str(error_logs), "error_logs")
    else:
        print("Allrun executed successfully without errors.")

    print("</runner>")
    progress = update_error_progress(
        state,
        error_logs,
        max_no_progress_loops=getattr(
            config,
            "max_no_progress_loops",
            2,
        ),
    )
    log_correction_round(
        int(state.get("loop_count", 0)),
        runner="local",
        case_dir=case_dir,
        error_logs=error_logs,
        state={**state, **progress},
    )

    # Return updated state
    return {
        **state,
        "error_logs": error_logs,
        **progress,
    }
