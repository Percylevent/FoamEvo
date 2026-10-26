"""Frozen evolution-side suites and commit-bound regression evidence."""
from __future__ import annotations
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from .matrix import EvaluationMatrix
from ..models import CaseRef
from ..worktree.assets import STATIC_ASSETS, EXPERIENCE_ASSETS, materialize_runtime_assets

class ProtectionError(RuntimeError):
    pass

CONTRACT_FILES = ('src/evaluation.py', 'src/openfoam_runtime.py', 'src/config.py')
DERIVED = ('record/', 'evorecord/', 'runs/', 'output/', 'database/faiss/',
           'database/experience/faiss/', 'database/experience/index.jsonl',
           'database/experience/validations.jsonl', 'database/foamgpt/data/')

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                     separators=(',', ':')).encode()).hexdigest()

def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def git(repo, *args):
    r = subprocess.run(['git', *args], cwd=repo, capture_output=True, text=True)
    if r.returncode:
        raise ProtectionError(r.stderr.strip() or 'git operation failed')
    return r.stdout.strip()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=True, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)

def seal(value):
    return {**value, 'digest': digest({k: v for k, v in value.items() if k != 'digest'})}

def read_sealed(path):
    value = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(value, dict) or value.get('digest') != seal(value)['digest']:
        raise ProtectionError(f'evidence checksum mismatch: {path}')
    return value

def runtime_fingerprint():
    keys = ('WM_PROJECT_VERSION', 'WM_PROJECT_DIR', 'WM_OPTIONS', 'FOAM_RUN',
            'OPENAI_MODEL', 'ANTHROPIC_MODEL', 'OMP_NUM_THREADS')
    values = {k: v for k, v in os.environ.items()
              if k.startswith(('FOAM_AGENT_', 'FOAMAGENT_', 'FOAM_RUNNER_', 'ANTHROPIC_', 'OPENAI_')) or k in keys}
    # Only a digest is persisted; credentials are never written to manifests.
    return digest({'environment': values, 'python': sys.version, 'executable': sys.executable})

def source_fingerprint(repo):
    repo = Path(repo)
    names = git(repo, 'ls-files', '--cached', '--others', '--exclude-standard', '-z').split('\0')
    rows = {}
    for name in sorted(set(names)):
        if not name or name.startswith(DERIVED) or '__pycache__' in name or name.endswith('.pyc'):
            continue
        path = repo / name
        rows[name] = file_hash(path) if path.is_file() else '<missing>'
    return digest(rows)

def assert_committed_source(repo):
    changed = git(repo, 'diff', 'HEAD', '--name-only').splitlines()
    changed += git(repo, 'ls-files', '--others', '--exclude-standard').splitlines()
    if any(name and not name.startswith(DERIVED) and '__pycache__' not in name and not name.endswith('.pyc') for name in changed):
        raise ProtectionError('source changes are not bound to the tested commit')


RUNTIME_ASSETS = EXPERIENCE_ASSETS

def assets_fingerprint(repo, names=(*STATIC_ASSETS, *EXPERIENCE_ASSETS)):
    root = Path(repo)
    rows = {}
    for name in names:
        path = root / name
        if path.is_file():
            rows[name] = file_hash(path)
        elif path.is_dir():
            for item in sorted(path.rglob('*')):
                if item.is_file():
                    rows[item.relative_to(root).as_posix()] = file_hash(item)
        else:
            rows[name] = '<missing>'
    return digest(rows)


def contract_hashes(repo):
    return {name: file_hash(Path(repo) / name) if (Path(repo) / name).is_file() else None
            for name in CONTRACT_FILES}

def retention_registry(repo):
    common = Path(git(repo, 'rev-parse', '--git-common-dir'))
    if not common.is_absolute():
        common = Path(repo) / common
    return common.resolve() / 'foam-refiner-retention.json'

def suite_cases(path):
    value = read_sealed(path)
    if value.get('schema_version') != 1 or value.get('split') != 'evolution':
        raise ProtectionError('a frozen evolution-side suite is required')
    cases = {}
    for row in value['cases']:
        ref = CaseRef.model_validate(row['ref'])
        for key, expected in row['files'].items():
            if file_hash(key) != expected:
                raise ProtectionError(f'frozen input or evaluation changed: {key}')
        if ref.case_id in cases:
            raise ProtectionError('duplicate suite case identifier')
        cases[ref.case_id] = ref
    if set(cases) != set(value['failure_ids']) | set(value['retention_ids']):
        raise ProtectionError('suite coverage does not match case manifest')
    if set(value['failure_ids']) & set(value['retention_ids']):
        raise ProtectionError('failure and retention sets overlap')
    return value, cases

def freeze_suite(*, repo, base, sources, allowed_ids, path, workers=1):
    """Snapshot declared evolution inputs and refresh baseline before development."""
    from ..execution.replay import FoamAgentReplay, _load_evaluation_path
    from ..experience.store import ExperienceStore
    repo, path = Path(repo), Path(path)
    if path.exists():
        raise ProtectionError('suite already exists; choose a new suite path')
    allowed = set(allowed_ids)
    if not allowed or len(allowed) != len(allowed_ids):
        raise ProtectionError('evolution allowlist must be nonempty and unique')
    selected = {k: v for k, v in sources.items() if k in allowed}
    registry = retention_registry(repo)
    registry_digest = None
    if registry.with_suffix('.pending.json').exists():
        raise ProtectionError('an interrupted adoption requires retention reconciliation')
    if registry.exists():
        saved = read_sealed(registry)
        if saved['adopted_commit'] != base:
            raise ProtectionError('retention registry does not match the current baseline')
        registry_digest = saved['digest']
        for row in saved['cases']:
            ref = CaseRef.model_validate(row['ref'])
            if ref.case_id not in allowed:
                raise ProtectionError('previously retained case omitted from evolution allowlist')
            for key, expected in row['files'].items():
                if file_hash(key) != expected:
                    raise ProtectionError('retained evidence has changed')
            if ref.case_id in selected and file_hash(ref.prompt_path) != file_hash(selected[ref.case_id].prompt_path):
                raise ProtectionError('retained case identifier reused for another input')
            selected[ref.case_id] = ref
    if set(selected) != allowed:
        raise ProtectionError('allowlist includes missing or incomplete case records')
    if git(repo, 'rev-parse', 'HEAD') != base or git(repo, 'status', '--porcelain', '--untracked-files=no'):
        raise ProtectionError('freeze requires a clean checkout of declared base')
    root = path.parent / (path.stem + '-artifacts')
    root.mkdir(parents=True, exist_ok=False)
    frozen, previous_successes = {}, set()
    for i, (case_id, source) in enumerate(sorted(selected.items())):
        folder = root / 'inputs' / f'{i:05d}'
        folder.mkdir(parents=True)
        task = json.loads(source.task_path.read_text(encoding='utf-8'))
        if str(task.get('split', 'evolution')).lower() in {'test', 'holdout', 'validation', 'advanced'}:
            raise ProtectionError(f'held-out task cannot enter adaptation: {case_id}')
        shutil.copy2(source.prompt_path, folder / 'prompt.txt')
        task['prompt_path'] = str((folder / 'prompt.txt').resolve())
        write_json(folder / 'task.json', task)
        evaluation_path = source.evaluation_path or source.case_path / 'evaluation.json'
        _, error = _load_evaluation_path(evaluation_path)
        if error:
            raise ProtectionError(f'{case_id}: {error}')
        evaluation = json.loads(evaluation_path.read_text(encoding='utf-8'))
        if evaluation['passed']:
            previous_successes.add(case_id)
        shutil.copy2(evaluation_path, folder / 'evaluation.json')
        frozen[case_id] = source.model_copy(update={
            'case_path': folder.resolve(), 'task_path': (folder / 'task.json').resolve(),
            'prompt_path': (folder / 'prompt.txt').resolve(),
            'evaluation_path': (folder / 'evaluation.json').resolve(), 'baseline_commit': base})
    runtime = runtime_fingerprint()
    static_assets = assets_fingerprint(repo, STATIC_ASSETS)
    worktree = root / 'baseline-worktree'
    git(repo, 'worktree', 'add', '--detach', str(worktree.resolve()), base)
    try:
        materialize_runtime_assets(repo, worktree)
        if assets_fingerprint(worktree, STATIC_ASSETS) != static_assets:
            raise ProtectionError('baseline retrieval assets could not be materialized')
        if (worktree / 'database').is_dir():
            ExperienceStore(worktree).prepare_for_replay()
        assert_committed_source(worktree)
        baseline_source, baseline_assets = source_fingerprint(worktree), assets_fingerprint(worktree)
        baseline_contract = contract_hashes(worktree)
        replay = FoamAgentReplay(source_cases=frozen, replay_workers=workers)
        matrix = replay.run_matrix(case_ids=sorted(frozen), worktree=worktree,
                                   output_root=root / 'baseline-replay', workflow_version=base)
        if not matrix.comparable or matrix.infrastructure_blocked:
            raise ProtectionError('baseline replay unavailable or evaluator contract invalid')
        if any(x.case_id in previous_successes and not x.passed for x in matrix.outcomes):
            raise ProtectionError('previously successful case failed during baseline refresh')
        if (git(worktree, 'rev-parse', 'HEAD') != base or source_fingerprint(worktree) != baseline_source
                or assets_fingerprint(worktree) != baseline_assets):
            raise ProtectionError('baseline state changed during replay')
        if runtime != runtime_fingerprint():
            raise ProtectionError('runtime changed during baseline refresh')
        rows, failures, retention = [], [], []
        for item in matrix.outcomes:
            ref = frozen[item.case_id]
            write_json(ref.evaluation_path, {'schema_version': 2, 'passed': item.passed,
                                           'checks': list(item.candidate_checks)})
            (retention if item.passed else failures).append(item.case_id)
            rows.append({'ref': ref.model_dump(mode='json'), 'files': {
                str(p): file_hash(p) for p in (ref.task_path, ref.prompt_path, ref.evaluation_path)}})
        value = seal({'schema_version': 1, 'split': 'evolution', 'base_commit': base,
                      'runtime': runtime, 'contract': baseline_contract, 'static_assets': static_assets,
                      'retention_registry_digest': registry_digest,
                      'failure_ids': failures, 'retention_ids': retention, 'cases': rows})
        write_json(root / 'baseline-matrix.json', matrix.to_dict())
        write_json(path, value)
        return value
    finally:
        git(repo, 'worktree', 'remove', '--force', str(worktree.resolve()))

def validate_matrix(matrix, failures, retention):
    expected = set(failures) | set(retention)
    actual = [x.case_id for x in matrix.outcomes]
    if not failures or set(actual) != expected or len(actual) != len(expected):
        raise ProtectionError('incomplete or duplicate failure/retention replay coverage')
    if not matrix.gate_passed:
        raise ProtectionError('replay did not improve without regression: ' +
                              '; '.join((*matrix.comparison_errors, *matrix.regressions)))
    if any(not x.passed for x in matrix.outcomes if x.case_id in retention):
        raise ProtectionError('a protected successful case failed')
    if not EvaluationMatrix(tuple(x for x in matrix.outcomes if x.case_id in failures)).improved:
        raise ProtectionError('failure batch did not improve')

def evaluate_suite(*, repo, suite_path, failure_ids, output, workers=1):
    from ..execution.replay import FoamAgentReplay
    repo, suite_path, output = Path(repo), Path(suite_path), Path(output)
    suite, sources = suite_cases(suite_path)
    if not set(failure_ids) <= set(suite['failure_ids']):
        raise ProtectionError('failure batch is outside frozen evolution suite')
    if contract_hashes(repo) != suite['contract']:
        raise ProtectionError('evaluator or fixed runtime configuration changed')
    if assets_fingerprint(repo, STATIC_ASSETS) != suite['static_assets']:
        raise ProtectionError('fixed retrieval assets differ from frozen baseline')
    if runtime_fingerprint() != suite['runtime']:
        raise ProtectionError('runtime differs from frozen baseline')
    assert_committed_source(repo)
    tested_commit = git(repo, 'rev-parse', 'HEAD')
    before = source_fingerprint(repo)
    assets = assets_fingerprint(repo)
    matrix = FoamAgentReplay(source_cases=sources, replay_workers=workers).run_matrix(
        case_ids=list(dict.fromkeys(failure_ids + suite['retention_ids'])),
        worktree=repo, output_root=output / 'cases', workflow_version=tested_commit)
    write_json(output / 'matrix.json', matrix.to_dict())
    if (tested_commit != git(repo, 'rev-parse', 'HEAD') or before != source_fingerprint(repo)
            or assets != assets_fingerprint(repo) or runtime_fingerprint() != suite['runtime']):
        raise ProtectionError('tested state changed during replay')
    validate_matrix(matrix, failure_ids, suite['retention_ids'])
    evidence = seal({'schema_version': 1, 'suite_path': str(suite_path.resolve()),
                     'suite_digest': suite['digest'], 'base_commit': suite['base_commit'],
                     'tested_commit': tested_commit, 'source_fingerprint': before, 'assets_fingerprint': assets,
                     'failure_ids': failure_ids, 'retention_ids': suite['retention_ids'],
                     'matrix': matrix.to_dict(), 'output': str(output.resolve()), 'runtime': suite['runtime']})
    write_json(output / 'validation.json', evidence)
    return matrix, evidence

def verify_evidence(path, *, commit, base):
    evidence = read_sealed(path)
    suite, _ = suite_cases(evidence['suite_path'])
    if (evidence['tested_commit'] != commit or evidence['base_commit'] != base
            or evidence['suite_digest'] != suite['digest'] or suite['base_commit'] != base):
        raise ProtectionError('evidence has a different tested commit, base or suite')
    if evidence['retention_ids'] != suite['retention_ids']:
        raise ProtectionError('candidate omitted protected cases')
    if not set(evidence['failure_ids']) <= set(suite['failure_ids']):
        raise ProtectionError('candidate includes cases outside evolution split')
    if not all(evidence.get(k) for k in ('source_tests_passed', 'plan_approved', 'code_approved')) or evidence.get('iteration_verdict') != 'CONVERGED':
        raise ProtectionError('candidate source checks or reviews are incomplete')
    validate_matrix(EvaluationMatrix.from_dict(evidence['matrix']), evidence['failure_ids'], evidence['retention_ids'])
    return evidence

