"""Build local tutorial indexes using the Runner embedding configuration."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

from src.config import Config

INDEX_SCRIPTS = {
    "openfoam_command_help": "faiss_command_help.py",
    "openfoam_allrun_scripts": "faiss_allrun_scripts.py",
    "openfoam_tutorials_structure": "faiss_tutorials_structure.py",
    "openfoam_tutorials_details": "faiss_tutorials_details.py",
}

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--openfoam_path',
        type=str,
        default=os.getenv("WM_PROJECT_DIR"),
        help="OpenFOAM installation (defaults to WM_PROJECT_DIR)"
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help="Rebuild the raw corpus and all selected-model indexes"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    config = Config()
    root = Path(__file__).resolve().parent
    database = root / "database"
    model_dir = config.embedding_model.replace("/", "_").replace(":", "_")
    commands = []
    raw_files = ["openfoam_case_stats.json", "openfoam_commands.txt"]
    raw_files.extend(name + ".txt" for name in INDEX_SCRIPTS)
    rebuild_raw = args.force or any(not (database / "raw" / name).is_file() for name in raw_files)
    if rebuild_raw:
        if not args.openfoam_path or not (Path(args.openfoam_path) / "tutorials").is_dir():
            raise SystemExit("Source OpenFOAM or pass --openfoam_path with a tutorials directory.")
        commands.append([sys.executable, str(database / "script" / "tutorial_parser.py"),
                         "--output_dir", str(database / "raw"),
                         "--wm_project_dir", str(Path(args.openfoam_path).resolve())])
    for name, script in INDEX_SCRIPTS.items():
        target = database / "faiss" / model_dir / name
        if rebuild_raw or not all((target / file).is_file() for file in ("index.faiss", "index.pkl")):
            commands.append([sys.executable, str(database / "script" / script),
                             "--database_path", str(database),
                             "--embedding_provider", config.embedding_provider,
                             "--embedding_model", config.embedding_model])
    if not commands:
        print("All selected-model indexes already exist. Use --force to rebuild.")
        return
    for command in commands:
        subprocess.run(command, cwd=root, check=True)
    print("Database initialization completed successfully.")

if __name__ == "__main__":
    main()

