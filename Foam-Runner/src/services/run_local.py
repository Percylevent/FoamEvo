import json
import os
import re
from typing import List, Any
from utils import remove_files, remove_file, remove_numeric_folders, run_command
from openfoam_runtime import detect_execution_errors, format_error_logs


def _task_expectations():
    run_dir = os.getenv("FOAM_AGENT_RUN_ROOT")
    if not run_dir:
        return {}
    try:
        with open(os.path.join(run_dir, "task.json"), encoding="utf-8") as stream:
            task = json.load(stream)
    except (OSError, ValueError):
        return {}
    expected = task.get("expected")
    return expected if isinstance(expected, dict) else {}

def run_allrun_and_collect_errors(
    case_dir: str,
    timeout: int = 3600,
    max_retries: int = 1
) -> List[str]:
    """
    Execute the Allrun script and collect any error logs from the simulation.
    
    This function runs the Allrun script in the specified case directory,
    captures the output and error streams, and parses the results to identify
    any OpenFOAM errors that occurred during execution.
    
    Args:
        case_dir (str): Directory path containing the OpenFOAM case and Allrun script
        timeout (int, optional): Maximum execution time in seconds. Defaults to 3600.
        max_retries (int, optional): Maximum number of retry attempts. Defaults to 3.
    
    Returns:
        List[str]: List of error messages found in the simulation logs.
                 Empty list indicates successful execution with no errors.
    
    Raises:
        FileNotFoundError: If Allrun script does not exist in case_dir
        RuntimeError: If Allrun script execution fails repeatedly
        TimeoutError: If execution exceeds timeout limit
    
    Example:
        >>> errors = run_allrun_and_collect_errors(
        ...     case_dir="/path/to/case",
        ...     timeout=1800,
        ...     max_retries=2
        ... )
        >>> if not errors:
        ...     print("Simulation completed successfully")
        >>> else:
        ...     print(f"Found {len(errors)} errors")
    """
    allrun_file_path = os.path.join(case_dir, "Allrun")
    if not os.path.exists(allrun_file_path):
        return format_error_logs(
            detect_execution_errors(
                case_dir=case_dir,
                expected=_task_expectations(),
            )
        )
    
    out_file = os.path.join(case_dir, "Allrun.out")
    err_file = os.path.join(case_dir, "Allrun.err")

    # Cleanup
    remove_files(case_dir, prefix="log")
    remove_file(err_file)
    remove_file(out_file)
    remove_numeric_folders(case_dir)

    last_error_logs = []

    # Run with retries
    for attempt in range(1, max_retries + 1):
        print(f"Running Allrun (attempt {attempt}/{max_retries})")
        command_result = run_command(
            allrun_file_path,
            out_file,
            err_file,
            case_dir,
            timeout,
        )

        evidence = detect_execution_errors(
            case_dir=case_dir,
            command=command_result,
            expected=_task_expectations(),
        )
        error_logs = format_error_logs(evidence)
        if len(error_logs) == 0:
            return []

        last_error_logs = error_logs
        if attempt < max_retries:
            print("Allrun reported errors; retrying after cleanup...")
            remove_files(case_dir, prefix="log")
            remove_file(err_file)
            remove_file(out_file)
            remove_numeric_folders(case_dir)

    return last_error_logs
