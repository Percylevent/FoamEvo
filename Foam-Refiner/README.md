# Foam-Refiner

Foam-Refiner is the evolution and integration component of FoamEvo. It consumes Foam-Runner records, groups related failures, develops candidate workflow and domain-experience updates in isolated Git worktrees, and adopts only jointly validated changes.

The paper calls these roles **Foam-CoEvolver** and **Foam-Merger**. The source directory and CLI retain the historical names `Foam-Refiner`, `foam_evo`, and `foam-refiner` so existing records and branches remain readable. See the [top-level README](../README.md) for the complete reproducibility workflow.

## Workflow

1. Start from an independent Git repository rooted at Foam-Runner with a clean, committed baseline and a completed benchmark record. The top-level installation instructions prepare this runtime repository from the published source tree.
2. Run `freeze-suite` with an explicit JSON allowlist of evolution cases. Baseline replay establishes the protection set and freezes evaluator and runtime identity.
3. Run `evolve` (or `batch`) to diagnose failures, propose workflow/experience updates, run source tests, and replay each candidate against its failure batch and protected cases.
4. Run `merge` to semantically reconcile eligible candidates from the common baseline and jointly replay the integrated state.
5. Run `adopt` to promote the exact validated commit and derived experience assets to the target branch.

Acceptance is fail-closed: missing or malformed evaluations, infrastructure failures, identity mismatches, lost baseline checks, loss of protected-task success, and zero improvement block promotion. The replay scope and evidence hashes are retained with the evolution record.

## CLI example

```bash
python -m pip install -e ".[dev]"
foam-refiner --help

foam-refiner freeze-suite \
  --repo-root /path/to/Foam-Runner \
  --base-ref main \
  --record-root /path/to/records \
  --record-id '<five-character-record-id>' \
  --benchmark-root /path/to/bench/foambench-126 \
  --evolution-cases /path/to/evolution-cases.json \
  --output /path/to/evo/round-001/suite.json
```

Evolution roles use the installed `claude` executable. Set `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`, and any endpoint variables required by the local Claude CLI. Candidate worktrees are temporary unless `--worktree-root` is supplied. Keep validation artifacts and external meshes available until adoption is complete.

## Tests

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
```

The tests exercise the deterministic control plane, including record selection, Markdown role protocols, replay alignment, protection and non-regression gates, candidate integration, adoption identity checks, and retention carryover. They do not call a model or run OpenFOAM.
