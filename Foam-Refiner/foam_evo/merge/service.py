"""Explicit integration, combined-state replay, and guarded fast-forward adoption."""
from __future__ import annotations
import json
import subprocess
import tempfile
import shutil
from pathlib import Path
from ..artifacts.store import ArtifactStore
from ..models import MergeRequest, MergeResult
from ..evaluation.matrix import EvaluationMatrix
from ..evaluation.protection import (ProtectionError, verify_evidence, suite_cases,
    evaluate_suite, source_fingerprint, git, write_json, read_sealed, seal,
    retention_registry, file_hash, validate_matrix, runtime_fingerprint, assets_fingerprint, RUNTIME_ASSETS)
from ..experience.store import ExperienceStore
from ..worktree.assets import STATIC_ASSETS, materialize_runtime_assets
from ..execution.batch_runtime import _commit_all
from .gates import run_full_tests
from .git import MergeGitService


def promote(repo, *, base_ref, expected, commit, validated_worktree=None):
    """Lock the target ref while updating its clean checked-out worktree."""
    repo = Path(repo)
    target = git(repo, 'rev-parse', '--symbolic-full-name', base_ref)
    if not target.startswith('refs/heads/'):
        raise ProtectionError('adoption target must be a local branch')
    git(repo, 'merge-base', '--is-ancestor', expected, commit)
    checked = None
    for block in git(repo, 'worktree', 'list', '--porcelain').split('\n\n'):
        fields = dict(line.split(' ', 1) for line in block.splitlines() if ' ' in line)
        if fields.get('branch') == target:
            checked = Path(fields['worktree'])
            if git(checked, 'status', '--porcelain'):
                raise ProtectionError('target worktree has uncommitted or untracked files')
    # update-ref prepare locks the ref and verifies the expected baseline.
    process = subprocess.Popen(['git', 'update-ref', '--stdin'], cwd=repo,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    updated = False
    assets_installed = False
    backup = tempfile.TemporaryDirectory(prefix='foam-refiner-adopt-')
    try:
        process.stdin.write(f'start\nupdate {target} {commit} {expected}\nprepare\n'.encode('ascii'))
        process.stdin.flush()
        if process.stdout.readline().decode('ascii').strip() != 'start: ok' or process.stdout.readline().decode('ascii').strip() != 'prepare: ok':
            raise ProtectionError('target baseline moved or cannot be locked')
        if checked:
            if validated_worktree and assets_fingerprint(checked, STATIC_ASSETS) != assets_fingerprint(validated_worktree, STATIC_ASSETS):
                raise ProtectionError('target retrieval assets differ from validated state')
            if git(checked, 'status', '--porcelain'):
                raise ProtectionError('target worktree changed during adoption')
            # read-tree -m -u refuses to overwrite conflicting local edits.
            git(checked, 'read-tree', '-u', '-m', expected, commit)
            updated = True
            if validated_worktree:
                for name in RUNTIME_ASSETS:
                    _copy_asset(checked / name, Path(backup.name) / name)
                assets_installed = True
                for name in RUNTIME_ASSETS:
                    _copy_asset(Path(validated_worktree) / name, checked / name)
        process.stdin.write(b'commit\n'); process.stdin.flush()
        if process.stdout.readline().decode('ascii').strip() != 'commit: ok':
            raise ProtectionError('reference transaction failed')
        process.stdin.close()
        if process.wait(timeout=30):
            raise ProtectionError('reference transaction exited unsuccessfully')
    except Exception:
        try:
            process.stdin.write(b'abort\n'); process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass
        if process.poll() is None:
            process.terminate(); process.wait(timeout=30)
        if updated and git(repo, 'rev-parse', target) == expected:
            git(checked, 'read-tree', '-u', '-m', commit, expected)
            if assets_installed:
                for name in RUNTIME_ASSETS:
                    _copy_asset(Path(backup.name) / name, checked / name)
        raise
    finally:
        if process.poll() is None:
            process.terminate(); process.wait(timeout=30)
        backup.cleanup()


def _copy_asset(source, target):
    """Copy only the explicit derived experience paths during a guarded promotion."""
    if target.is_dir():
        shutil.rmtree(target)
    elif target.exists():
        target.unlink()
    if source.is_dir():
        shutil.copytree(source, target)
    elif source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def update_retention(repo, evidence, destination):
    suite, sources = suite_cases(evidence['suite_path'])
    registry = retention_registry(repo)
    current = read_sealed(registry)['digest'] if registry.exists() else None
    if current != suite.get('retention_registry_digest'):
        raise ProtectionError('retention registry changed since the suite was frozen')
    rows = []
    for i, outcome in enumerate(EvaluationMatrix.from_dict(evidence['matrix']).outcomes):
        if not outcome.passed:
            continue
        ref = sources[outcome.case_id]
        evaluation = Path(destination) / 'retained' / f'{i:05d}.json'
        write_json(evaluation, {'schema_version': 2, 'passed': True,
                               'checks': list(outcome.candidate_checks)})
        ref = ref.model_copy(update={'evaluation_path': evaluation.resolve(),
                                    'baseline_commit': evidence['tested_commit']})
        rows.append({'ref': ref.model_dump(mode='json'), 'files': {
            str(p): file_hash(p) for p in (ref.task_path, ref.prompt_path, ref.evaluation_path)}})
    return seal({'schema_version': 1, 'adopted_commit': evidence['tested_commit'], 'cases': rows})


def adopt_validation(repo, base_ref, validation_path):
    """Adopt only an integrated commit with complete replay and source evidence."""
    repo, validation_path = Path(repo), Path(validation_path)
    evidence = read_sealed(validation_path)
    if evidence.get('stage') != 'integration' or not evidence.get('source_tests_passed'):
        raise ProtectionError('integrated-state validation is required')
    suite, _ = suite_cases(evidence['suite_path'])
    if evidence['suite_digest'] != suite['digest'] or evidence['base_commit'] != suite['base_commit']:
        raise ProtectionError('integration suite identity mismatch')
    if evidence['runtime'] != runtime_fingerprint():
        raise ProtectionError('runtime changed since integration validation')
    failures, retention = evidence['failure_ids'], suite['retention_ids']
    if evidence['retention_ids'] != retention or not set(failures) <= set(suite['failure_ids']):
        raise ProtectionError('integration scope mismatch')
    validate_matrix(EvaluationMatrix.from_dict(evidence['matrix']), failures, retention)
    path = Path(evidence['worktree'])
    if git(path, 'rev-parse', 'HEAD') != evidence['tested_commit'] or source_fingerprint(path) != evidence['source_fingerprint'] or assets_fingerprint(path) != evidence['assets_fingerprint']:
        raise ProtectionError('integrated checkout changed since validation')
    for item in evidence['candidates']:
        if git(repo, 'rev-parse', item['branch']) != item['commit']:
            raise ProtectionError('candidate branch changed since integration')
        verify_evidence(item['evidence'], commit=item['commit'], base=evidence['base_commit'])
    if git(repo, 'rev-parse', base_ref) != evidence['base_commit']:
        raise ProtectionError('target baseline moved since integration')
    registry = retention_registry(repo)
    if registry.with_suffix('.pending.json').exists():
        raise ProtectionError('interrupted adoption must be reconciled first')
    lock_path = registry.with_suffix('.lock')
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    # Serializes engine-managed promotions and retention-set updates.
    lock_path.open('x').close()
    try:
        retained = update_retention(repo, evidence, validation_path.parent)
        write_json(validation_path.parent / 'retention-next.json', retained)
        previous = registry.read_bytes() if registry.exists() else None
        pending = registry.with_suffix('.pending.json')
        write_json(pending, seal({'previous_commit': evidence['base_commit'],
            'next_commit': evidence['tested_commit'], 'validation': str(validation_path.resolve()),
            'previous_registry': json.loads(previous) if previous else None, 'next_registry': retained}))
        try:
            write_json(registry, retained)
            promote(repo, base_ref=base_ref, expected=evidence['base_commit'], commit=evidence['tested_commit'], validated_worktree=path)
        except Exception:
            if git(repo, 'rev-parse', base_ref) != evidence['tested_commit']:
                if previous is None:
                    registry.unlink(missing_ok=True)
                else:
                    write_json(registry, json.loads(previous))
                pending.unlink(missing_ok=True)
            raise
        pending.unlink(missing_ok=True)
        write_json(validation_path.parent / 'adoption.json', seal({
            'adopted_commit': evidence['tested_commit'], 'previous_commit': evidence['base_commit'],
            'validation_digest': evidence['digest'], 'retention_digest': retained['digest']}))
    finally:
        lock_path.unlink(missing_ok=True)
    return evidence['tested_commit']


class MergeService:
    def __init__(self, repo_root, *, artifact_root=None, merge_worktree_root=None, semantic_merger=None):
        self.repo = Path(repo_root).resolve()
        self.git = MergeGitService(self.repo, worktree_root=merge_worktree_root)
        self.semantic_merger = semantic_merger
        self.artifact_root = Path(artifact_root or (Path(tempfile.gettempdir()) / 'foam-evo-artifacts/merges')).resolve()

    def run(self, request: MergeRequest, *, merge_id: str) -> MergeResult:
        artifacts = ArtifactStore(self.artifact_root / merge_id)
        base, committed = '', False
        try:
            if len(request.candidate_branches) != len(request.candidate_evidence) or not request.candidate_evidence:
                raise ProtectionError('supply one accepted validation file per candidate branch')
            if len(set(request.candidate_branches)) != len(request.candidate_branches):
                raise ProtectionError('duplicate candidate branches')
            base, diffs = self.git.inspect(request)
            accepted, identities = [], []
            for branch, evidence_path in zip(request.candidate_branches, request.candidate_evidence):
                tip = git(self.repo, 'rev-parse', branch)
                git(self.repo, 'merge-base', '--is-ancestor', base, tip)
                evidence = verify_evidence(evidence_path, commit=tip, base=base)
                accepted.append(evidence)
                identities.append({'branch': branch, 'commit': tip, 'evidence': str(evidence_path.resolve())})
            if len({x['suite_digest'] for x in accepted}) != 1:
                raise ProtectionError('candidate suites differ; fresh paired validation is required')
            artifacts.write_json('request.json', request.model_dump(mode='json'))
            # Integrate immutable commit IDs so branch movement cannot change the patch being merged.
            frozen = request.model_copy(update={'candidate_branches': [x['commit'] for x in identities]})
            result = self.git.merge(frozen, base_commit=base, merge_id=merge_id,
                                    semantic_merger=self.semantic_merger, artifact_dir=artifacts.root / 'merge')
            if result.status != 'READY':
                return self._persist(artifacts, result)
            worktree = self.git.worktree_for(merge_id)
            materialize_runtime_assets(self.repo, worktree)
            changed = git(worktree, 'diff', '--name-only', base).splitlines()
            if any(x.startswith(('database/faiss/', 'database/experience/faiss/')) for x in changed):
                raise ProtectionError('generated indexes cannot be merged as source changes')
            if (worktree / 'database').is_dir():
                ExperienceStore(worktree).prepare_for_replay()
            commit = _commit_all(worktree, f'integration-{merge_id}')
            tested_source, tested_assets = source_fingerprint(worktree), assets_fingerprint(worktree)
            run_full_tests(worktree)
            if tested_source != source_fingerprint(worktree) or tested_assets != assets_fingerprint(worktree) or commit != git(worktree, 'rev-parse', 'HEAD'):
                raise ProtectionError('source tests changed the integrated replay state')
            failure_ids = sorted({x for row in accepted for x in row['failure_ids']})
            matrix, evidence = evaluate_suite(repo=worktree, suite_path=Path(accepted[0]['suite_path']),
                failure_ids=failure_ids, output=artifacts.root / 'replay', workers=request.replay_workers)
            # Preserve each candidate's individually repaired checks as well as baseline successes.
            combined = {x.case_id: {c['check_id']: c['passed'] for c in x.candidate_checks} for x in matrix.outcomes}
            for row in accepted:
                for item in EvaluationMatrix.from_dict(row['matrix']).outcomes:
                    for check in item.candidate_checks:
                        if check['passed'] and not combined[item.case_id][check['check_id']]:
                            raise ProtectionError('integration lost a candidate repair')
            if git(self.repo, 'rev-parse', request.base_ref) != base:
                raise ProtectionError('target baseline moved during integration')
            evidence = seal({**evidence, 'stage': 'integration', 'source_tests_passed': True,
                             'candidates': identities, 'worktree': str(worktree.resolve())})
            validation = artifacts.root / 'integration-validation.json'
            write_json(validation, evidence)
            committed = True
            adopted = adopt_validation(self.repo, request.base_ref, validation) if request.adopt else None
            return self._persist(artifacts, result.model_copy(update={
                'status': 'ADOPTED' if adopted else 'VALIDATED', 'merged_commit': commit,
                'validation_path': str(validation), 'adopted_commit': adopted}))
        except Exception as exc:
            if not committed:
                try:
                    self.git.cleanup(merge_id, keep_branch=False)
                except Exception:
                    pass
            return self._persist(artifacts, MergeResult(status='FAILED', base_commit=base,
                                 error=f'{type(exc).__name__}: {exc}'))

    @staticmethod
    def _persist(artifacts, result):
        path = artifacts.write_json('result.json', result.model_dump(mode='json'))
        final = result.model_copy(update={'result_path': str(path)})
        artifacts.write_json('result.json', final.model_dump(mode='json'))
        return final
