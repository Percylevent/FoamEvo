# FoamEvo

FoamEvo is a simulator-grounded framework for improving OpenFOAM workflows across CFD tasks. It executes a task, preserves structured runtime evidence, develops candidate changes in isolated worktrees, and adopts only changes that pass replay and regression checks.

This repository accompanies the paper *Simulator-Grounded Evolution of Computational Fluid Dynamics Workflows and Domain Experience*. The source release keeps the implementation names used by the code:

| Paper module | Repository component | Role |
| --- | --- | --- |
| Foam-Executor | `Foam-Runner` | Plans and runs an OpenFOAM case, performs bounded task-local recovery, and writes evaluation evidence. |
| Foam-CoEvolver | `Foam-Refiner` (`evolve` / `batch`) | Groups failed cases and develops workflow and experience candidates in isolated worktrees. |
| Foam-Merger | `Foam-Refiner` (`merge` / `adopt`) | Reconciles candidates and performs joint replay before adoption. |

The paper names and the code names are functionally equivalent. The internal Python package is still called `foam_evo`, and the command and branch names remain `foam-refiner` and `foam-evo/...` for compatibility with recorded artifacts.

## What is included

```text
FoamEvo/
├── Foam-Runner/                  execution agent and OpenFOAM evaluator
│   ├── src/                      LangGraph workflow, CLI, evaluator, retrieval
│   ├── database/raw/             tracked OpenFOAM tutorial corpus
│   ├── database/experience/      Markdown experience source entries
│   └── tests/
├── Foam-Refiner/                 evolution and integration control plane
│   ├── foam_evo/                 selection, roles, replay, gates, merge, adoption
│   └── tests/
├── bench/foambench-126/          126 portable task definitions and prompts
├── bench/run_bench.py            multi-worker benchmark harness
└── README.md
```

Generated FAISS indexes, run records, evolution records, Git histories, caches, credentials, and obsolete FoamGPT training-data utilities are excluded. The raw tutorial corpus, 126 benchmark task definitions, `tandem_wing.msh`, and example requirement files are included. Experiment logs, exact evolution/holdout allowlists, additional benchmarks, and baseline-method implementations are not bundled; the example workflow below does not reproduce those experimental splits.

## Requirements

The supported runtime is Linux with:

- Python 3.10 or newer, Git, and Git LFS for the two large source assets.
- OpenFOAM Foundation 10, sourced before running cases. OpenFOAM commands that compile dynamic code must run as a non-root user.
- An LLM endpoint supported by Foam-Runner (`openai`, `anthropic`, `openai-codex`, `bedrock`, `ollama`, or `deepseek`).
- An embedding provider. The default is Hugging Face `Qwen/Qwen3-Embedding-0.6B`; an OpenAI-compatible or Ollama embedding model can also be selected.
- Claude Code CLI for Foam-Refiner role execution.
- Gmsh when a task requests a generated Gmsh mesh; PyVista is needed only for visualization.

`Foam-Runner/environment.yml` records the original Linux Conda environment. The commands below install the Python dependencies; OpenFOAM, Git LFS, and Claude CLI are separate prerequisites.

## Installation

Run these Bash commands from the repository root. Keep the path variables set in the same shell:

```bash
export FOAMEVO_ROOT="$PWD"
export FOAMEVO_RUNNER="$FOAMEVO_ROOT/Foam-Runner"
git lfs install
git lfs pull     # for a Git clone; the source archive already contains the assets
```

**For evolution, prepare an independent Runner Git repository first.** The control plane expects `src/`, `tests/`, and `database/` at the Git worktree root. Use a new, empty location outside this published checkout:

```bash
export FOAMEVO_WORKSPACE="$HOME/foamevo-workspace"
mkdir -p "$FOAMEVO_WORKSPACE"
cp -a "$FOAMEVO_ROOT/Foam-Runner" "$FOAMEVO_WORKSPACE/"
export FOAMEVO_RUNNER="$FOAMEVO_WORKSPACE/Foam-Runner"
git -C "$FOAMEVO_RUNNER" init -b main
git -C "$FOAMEVO_RUNNER" lfs install --local
git -C "$FOAMEVO_RUNNER" add .
git -C "$FOAMEVO_RUNNER" commit -m "FoamEvo execution baseline"
git -C "$FOAMEVO_RUNNER" status --short
```

Configure your Git author identity before committing. The status must be clean before `freeze-suite`. For execution-only use, skip this block and retain the original `FOAMEVO_RUNNER` value.

```bash
cd "$FOAMEVO_ROOT"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e "$FOAMEVO_RUNNER[dev]"
python -m pip install -e "$FOAMEVO_ROOT/Foam-Refiner[dev]"
source /opt/openfoam10/etc/bashrc       # adjust to the local OpenFOAM installation
foamVersion
foam-runner --help
foam-refiner --help
```

Set runtime variables in the shell or explicitly source an untracked local file. `.env` files are not loaded automatically. `Foam-Refiner/.env.example` lists the names without credentials. A minimal OpenAI-compatible setup is:

```bash
export FOAM_AGENT_MODEL_PROVIDER=openai
export FOAM_AGENT_MODEL_VERSION='<model-name>'
export FOAM_AGENT_OPENAI_BASE_URL='https://api.example/v1'
export FOAM_AGENT_OPENAI_API_KEY='<api-key>'
export FOAM_AGENT_EMBEDDING_PROVIDER=huggingface
export FOAM_AGENT_EMBEDDING_MODEL='Qwen/Qwen3-Embedding-0.6B'
export FOAM_AGENT_OPENFOAM_FORK=foundation
export FOAM_AGENT_RUN_ROOT="$FOAMEVO_ROOT/.runs"

export ANTHROPIC_API_KEY='<api-key>'
export ANTHROPIC_MODEL='<claude-model>'
```

Use `FOAM_AGENT_MODEL_PROVIDER=anthropic` with the corresponding `FOAM_AGENT_ANTHROPIC_*` variables for the native Anthropic API. Foam-Refiner invokes the installed `claude` executable and inherits its `ANTHROPIC_*` environment.

`FOAM_EVO_MODEL_VERSION`, if set, overrides `ANTHROPIC_MODEL` for roles. `FOAM_EVO_MODEL_PROVIDER` is not a runtime switch. OpenAI embeddings use the separate `OPENAI_API_KEY` and optional `OPENAI_API_BASE` settings. Keep the embedding model, solver, and runtime configuration fixed across baseline and candidate replay.

## Build retrieval indexes

The release tracks the source corpus, while FAISS indexes are generated locally so they match the selected embedding model. After sourcing OpenFOAM:

```bash
cd "$FOAMEVO_RUNNER"
python init_database.py --openfoam_path "$WM_PROJECT_DIR"
```

The initializer reads the configured embedding provider/model and skips complete indexes for that model. Use `--force` to rebuild the raw corpus and indexes. Initial Hugging Face use downloads the selected model. Foam-Refiner rebuilds experience assets during candidate validation; for a manual rebuild:

```bash
python database/script/init_experience_db.py
```

An empty `database/experience/entries/` directory is valid; experience retrieval then returns no guidance.

## Run one CFD task

Every task JSON contains a `prompt_path` and optional `cfd_context`. Paths in the released benchmark are relative to their task directory, so the repository can be moved freely.

```bash
cd "$FOAMEVO_RUNNER"
foam-runner run-case \
  --task "$FOAMEVO_ROOT/bench/foambench-126/tasks/Cavity_1/task.json" \
  --output-dir "$FOAMEVO_ROOT/.runs/cavity-1" \
  --max-review-loop 25
```

The output directory contains the copied task and prompt, generated OpenFOAM files, workflow logs, correction history, `result.json`, `evaluation.json`, and `failure_packet.json` when the case fails. The evaluator uses stable check identifiers so records can be compared during replay.

## Run FoamBench

Run a small smoke test first. The benchmark harness refuses to run as root because OpenFOAM dynamic-code solvers reject root execution.

```bash
cd "$FOAMEVO_ROOT"
python bench/run_bench.py \
  --cases bench/foambench-126/cases.json \
  --repo-root "$FOAMEVO_RUNNER" \
  --record-root "$FOAMEVO_ROOT/.runs" \
  --workers 1 \
  --limit 1
```

For the complete 126-case run, remove `--limit 1` and choose the worker count for the machine. Useful filters are `--only-family Cavity`, `--only-suite basic`, and `--only-slugs path/to/slugs.json`. Each invocation creates a new record; the record ID and `harness/results.jsonl` identify the exact run used by later evolution.

## Evolve and integrate state

Foam-Refiner expects the independent, clean Runner repository prepared above and a completed benchmark record. The single-case `--output-dir` example does not create a complete benchmark record. Save the development case IDs as a JSON string list in `evolution-cases.json`, including baseline successes to protect; exclude held-out cases. The harness uses task slugs such as `Cavity_1` as record case IDs. This is only a format example:

```json
["Cavity_1", "Cavity_2", "forwardStep_1"]
```

Freeze the baseline suite and refresh its evaluations:

```bash
foam-refiner freeze-suite \
  --repo-root "$FOAMEVO_RUNNER" \
  --base-ref main \
  --record-root "$FOAMEVO_ROOT/.runs" \
  --record-id '<five-character-record-id>' \
  --benchmark-root "$FOAMEVO_ROOT/bench/foambench-126" \
  --evolution-cases "$FOAMEVO_ROOT/evolution-cases.json" \
  --output "$FOAMEVO_ROOT/.evo/round-001/suite.json" \
  --replay-workers 1
```

Develop candidates from that frozen suite:

```bash
foam-refiner evolve \
  --repo-root "$FOAMEVO_RUNNER" \
  --base-ref main \
  --regression-suite "$FOAMEVO_ROOT/.evo/round-001/suite.json" \
  --record-root "$FOAMEVO_ROOT/.runs" \
  --record-id '<five-character-record-id>' \
  --evo-record-root "$FOAMEVO_ROOT/.evo" \
  --max-cases-per-batch 10 \
  --parallel-workers 2 \
  --replay-workers 1
```

Integrate eligible candidates using the `branch_name` and `validation_path` reported by the evolution record:

```bash
foam-refiner merge \
  --repo-root "$FOAMEVO_RUNNER" \
  --base-ref main \
  --branch 'foam-evo/batch/<batch-id>' \
  --candidate-evidence '/absolute/path/to/validation.json' \
  --merge-id round-001-integration \
  --replay-workers 1

foam-refiner adopt \
  --repo-root "$FOAMEVO_RUNNER" \
  --base-ref main \
  --validation '/absolute/path/to/integration-validation.json'
```

The control plane checks common ancestry, evaluator and environment identity, source tests, check-level improvement, protected-task success, candidate provenance, and post-merge replay. A candidate or integrated state is not adopted when any gate fails. The gates apply to the explicit replay suite recorded with the run.

For several candidates, repeat paired `--branch` and `--candidate-evidence` arguments in the same order. Preserve the frozen suite, its `suite-artifacts` directory, candidate evidence, retained inputs, and integration artifacts across rounds. External meshes/data and the solver installation must also remain available. Adoption advances the independent Runner repository.

## Tests

The tests use deterministic providers and synthetic replay fixtures without model calls or CFD solver runs. A tokenizer resource may download on first use:

```bash
cd "$FOAMEVO_RUNNER"
python -m pytest -q -p no:cacheprovider

cd "$FOAMEVO_ROOT/Foam-Refiner"
python -m pytest -q -p no:cacheprovider
```

The test suites cover task recording, error classification, experience filtering, evaluator contracts, candidate replay, non-regression gates, integration, adoption identity checks, and retention of previously successful cases. Full CFD benchmark execution additionally requires the external solver, model endpoint, embedding model, and any referenced mesh/data files.

## License and citation

The upstream MIT notice is preserved in [Foam-Runner/LICENSE](Foam-Runner/LICENSE). OpenFOAM-derived tutorial text retains its embedded notices. The supplied source has no separate license for Foam-Refiner; this release preparation does not assign it a new license.

Please cite *Simulator-Grounded Evolution of Computational Fluid Dynamics Workflows and Domain Experience* when using FoamEvo. Bibliographic metadata can be added when the paper's public citation is available.
