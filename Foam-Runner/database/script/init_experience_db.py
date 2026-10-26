"""Build the experience-guidance FAISS index from Markdown entries.

Mirrors ``faiss_tutorials_structure.py``: the ``<index>`` block becomes the
embedded ``page_content`` and the whole entry is carried in metadata so
retrieval can hand the full guidance text to a prompt.

Unlike the tutorial indices this one is expected to be empty at first. An
empty ``entries/`` directory is not an error; the script writes no index and
``load_experience_db()`` degrades to returning no results.
"""

import argparse
import json
import shutil
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

INDEX_NAME = "experience_guidance"


def extract_field(field_name: str, text: str) -> str:
    """Extracts the specified field from the given text."""
    match = re.search(fr"^{re.escape(field_name)}:[ \t]*(.*)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else "Unknown"


def tokenize(text: str) -> str:
    # Replace underscores with spaces
    text = text.replace('_', ' ')
    # Insert a space between a lowercase letter and an uppercase letter (global match)
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    return text.lower()


def load_entries(entries_dir: Path) -> List[Document]:
    """Parse every ``<experience_begin>...<experience_end>`` entry into a Document."""
    documents: List[Document] = []
    for path in sorted(entries_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        match = re.search(
            r"<experience_begin>(.*?)<experience_end>", text, re.DOTALL
        )
        if not match:
            print(f"Warning: no experience block in {path.name}; skipped")
            continue
        full_content = match.group(1).strip()

        index_match = re.search(r"<index>(.*?)</index>", full_content, re.DOTALL)
        if not index_match:
            print(f"Warning: no <index> block in {path.name}; skipped")
            continue
        index_content = index_match.group(1).strip()

        # status is declared in <source>, outside the embedded <index> block:
        # retiring an entry must not change its vector, only its visibility.
        status = extract_field("status", full_content)
        if status == "Unknown":
            status = "active"

        documents.append(Document(
            page_content=tokenize(index_content),  # Only <index> is embedded
            metadata={
                "full_content": full_content,
                "entry_id": path.stem,
                "trigger_type": extract_field("trigger_type", index_content),
                "solver": "" if extract_field("solver", index_content) == "Unknown" else extract_field("solver", index_content),
                "fork": "" if extract_field("openfoam_fork", index_content) == "Unknown" else extract_field("openfoam_fork", index_content),
                "version": (extract_field("openfoam_version", index_content)
                    if extract_field("openfoam_version", index_content) != "Unknown" else ""),
                "keywords": extract_field("keywords", index_content),
                "conditions": (extract_field("conditions_json", index_content)
                    if extract_field("conditions_json", index_content) != "Unknown" else "{}"),
                "status": status,
            },
        ))
    return documents


def build_embeddings(provider: str, model: str, config=None):
    """Same provider fan-out as src/utils.py:get_embedding_model.
    
    Args:
        provider: Embedding provider name (openai, huggingface, ollama)
        model: Model name/identifier
        config: Config object with API keys and base URLs (optional)
    """
    if provider == "openai":
        from langchain_openai.embeddings import OpenAIEmbeddings
        import os
        kwargs = {"model": model}
        if config:
            if hasattr(config, "embedding_api_key") and config.embedding_api_key:
                kwargs["api_key"] = config.embedding_api_key
            elif os.getenv("OPENAI_API_KEY"):
                kwargs["api_key"] = os.getenv("OPENAI_API_KEY")
            if hasattr(config, "embedding_base_url") and config.embedding_base_url:
                kwargs["base_url"] = config.embedding_base_url
        return OpenAIEmbeddings(**kwargs)
    if provider == "huggingface":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            raise ImportError("Please install langchain-huggingface")
        return HuggingFaceEmbeddings(model_name=model)
    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=model)
    raise ValueError(f"Unknown provider: {provider}")


def config_defaults():
    """Read the live embedding settings so the index matches what retrieval loads.

    An index built with a different model than ``load_experience_db()`` uses is
    silently useless: the directory name would not match and retrieval would
    find nothing. Defaulting to Config keeps the two in step.
    
    Returns:
        tuple: (provider, model, config_object)
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
        from src.config import Config
    except Exception:
        return "huggingface", "Qwen/Qwen3-Embedding-0.6B", None
    cfg = Config()
    return cfg.embedding_provider, cfg.embedding_model, cfg


def main() -> int:
    default_provider, default_model, config_obj = config_defaults()

    parser = argparse.ArgumentParser(
        description="Build the experience-guidance FAISS index from Markdown entries."
    )
    parser.add_argument(
        "--database_path",
        type=str,
        default=str(Path(__file__).resolve().parent.parent),
        help="Path to the database directory (default: the database/ next to this script)",
    )
    parser.add_argument(
        "--embedding_provider",
        type=str,
        default=default_provider,
        choices=["openai", "huggingface", "ollama"],
        help="Embedding provider (default: from Config)",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default=default_model,
        help="Embedding model name (default: from Config)",
    )
    args = parser.parse_args()

    database_path = Path(args.database_path).resolve()
    experience_dir = database_path / "experience"
    entries_dir = experience_dir / "entries"

    model_dir_name = args.embedding_model.replace("/", "_").replace(":", "_")
    persist_directory = experience_dir / "faiss" / model_dir_name / INDEX_NAME
    if not persist_directory.resolve().is_relative_to((experience_dir / "faiss").resolve()):
        raise ValueError("experience index path escapes its derived directory")
    if not entries_dir.is_dir():
        if persist_directory.is_dir():
            shutil.rmtree(persist_directory)
        print(f"No entries directory at {entries_dir}; nothing to index.")
        return 0

    documents = load_entries(entries_dir)
    if not documents:
        if persist_directory.is_dir():
            shutil.rmtree(persist_directory)
        print("Experience library is empty; no index written.")
        return 0

    embeddings = build_embeddings(args.embedding_provider, args.embedding_model, config_obj)
    vectordb = FAISS.from_documents(documents, embeddings)

    model_dir_name = args.embedding_model.replace("/", "_").replace(":", "_")
    persist_directory = experience_dir / "faiss" / model_dir_name / INDEX_NAME
    persist_directory.parent.mkdir(parents=True, exist_ok=True)
    vectordb.save_local(str(persist_directory))

    print(
        f"{len(documents)} experience entries indexed successfully! "
        f"Saved at: {persist_directory}"
    )
    return 0


if __name__ == "__main__":
    ## python database/script/init_experience_db.py
    raise SystemExit(main())
