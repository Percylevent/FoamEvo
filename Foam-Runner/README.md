# Foam-Runner

Foam-Runner is the execution component of FoamEvo. It turns a natural-language CFD requirement into an OpenFOAM case, retrieves tutorial and experience guidance, writes case files, runs the solver, performs bounded correction, and emits structured evaluation evidence.

The paper calls this module **Foam-Executor**. The repository keeps the name `Foam-Runner` for compatibility with the supplied release. See the [top-level README](../README.md) for installation, configuration, benchmark, and evolution instructions.

## CLI

```bash
python -m pip install -e ".[dev]"
foam-runner --help
foam-runner record-create --root /path/to/records
foam-runner run-case \
  --task /path/to/task.json \
  --output-dir /path/to/records/case-001 \
  --max-review-loop 25
```

`task.json` is a JSON object with a readable `prompt_path`, optional `cfd_context`, and optional `limits`. Relative prompt paths are resolved next to the task file. The case directory stores an immutable task and prompt copy, generated files, logs, correction history, and evaluator output.

## Configuration

Configuration is read from environment variables. The main names are:

| Variable | Purpose |
| --- | --- |
| `FOAM_AGENT_MODEL_PROVIDER` | `openai`, `anthropic`, `openai-codex`, `bedrock`, `ollama`, or `deepseek` |
| `FOAM_AGENT_MODEL_VERSION` | LLM model identifier |
| `FOAM_AGENT_OPENAI_BASE_URL`, `FOAM_AGENT_OPENAI_API_KEY` | OpenAI-compatible endpoint and credential |
| `FOAM_AGENT_ANTHROPIC_BASE_URL`, `FOAM_AGENT_ANTHROPIC_API_KEY` | Anthropic-compatible endpoint and credential |
| `FOAM_AGENT_EMBEDDING_PROVIDER`, `FOAM_AGENT_EMBEDDING_MODEL` | RAG embedding backend and model |
| `FOAM_AGENT_OPENFOAM_FORK` | `foundation` or `esi` |
| `FOAM_AGENT_RUN_ROOT` | Root for external run records |

The upstream aliases `FOAMAGENT_MODEL_PROVIDER` and `FOAMAGENT_MODEL_VERSION` remain accepted. Do not commit credentials, records, FAISS indexes, or Python caches.

## Retrieval data

`database/raw/` contains the tracked tutorial corpus. Run `init_database.py` to generate FAISS indexes using the configured embedding model before executing cases. Experience entries are Markdown files under `database/experience/entries/`; Foam-Refiner rebuilds their ledger and index during candidate validation. An empty experience store is valid.

## Development

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
python -m src.record_cli --help
```

Use a non-root OpenFOAM account for actual cases. Visualization is optional and requires PyVista; HPC execution is optional and requires the provider-specific extras.

## Conditional experience retrieval

Experience source entries declare `solver`, `openfoam_fork`, `openfoam_version`, lifecycle `status` and a one-line JSON object in `conditions_json`. Task JSON supplies the corresponding `cfd_context`; for example:

~~~json
{
  "prompt_path": "/absolute/path/prompt.txt",
  "cfd_context": {
    "flow_regime": "incompressible",
    "turbulence_model": "kOmegaSST",
    "reynolds_number": 100000
  }
}
~~~

An illustrative entry restriction is:

~~~text
solver: simpleFoam
openfoam_fork: foundation
openfoam_version: 10
conditions_json: {"flow_regime":"incompressible","turbulence_model":["kOmegaSST","kEpsilon"],"reynolds_number":{"min":10000,"max":1000000}}
~~~

This demonstrates the schema; choose bounds and guidance appropriate to the actual repair. Set `FOAM_AGENT_OPENFOAM_FORK` and `FOAM_RUNNER_OPENFOAM_VERSION` consistently with the solver environment; the version defaults to `WM_PROJECT_VERSION` when present.

Retrieval recalls nearest summary embeddings, then filters active status, exact solver/fork/version matches, declared conditions and the configured distance threshold (default 0.9). Missing constrained context rejects an entry. Names are case normalized; no solver-family compatibility is inferred. Condition values support equality, explicit alternatives and inclusive numeric bounds, using the same declared units without conversion. Absent restrictions or explicit `any` are unrestricted. Conditions must be valid finite values; malformed entries are rejected.

`record_cli` passes task `cfd_context` into the Runner configuration. Direct callers may supply `experience_context` through the configuration; `FOAM_RUNNER_EXPERIENCE_CONTEXT` is a fallback when no configuration context is supplied. The full retrieved entry, including remaining guidance assumptions, is shown to planning/review. Entry IDs and rejection reasons are logged. Experience edits go into the Markdown source; the Refiner rebuilds the ledger and index before replay. The old `foam-agent` command alias remains available.

