"""Materialize the ignored retrieval assets required by every replay checkout."""
from pathlib import Path
import shutil
STATIC_ASSETS = ('database/faiss', 'database/foamgpt/data')
EXPERIENCE_ASSETS = ('database/experience/faiss', 'database/experience/index.jsonl')
def materialize_runtime_assets(source, target):
    source, target = Path(source), Path(target)
    for relative in (*STATIC_ASSETS, *EXPERIENCE_ASSETS):
        src, dst = source / relative, target / relative
        if dst.exists():
            continue
        if src.is_dir():
            shutil.copytree(src, dst)
        elif src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
