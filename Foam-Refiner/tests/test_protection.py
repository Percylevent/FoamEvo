"""Local deterministic software tests; no CFD solver or language-model requests."""
import json
from pathlib import Path
import pytest
from foam_evo.evaluation.matrix import CaseOutcome, EvaluationMatrix
from foam_evo.evaluation.protection import (ProtectionError, validate_matrix, suite_cases, evaluate_suite,
    read_sealed, write_json, seal, freeze_suite, retention_registry, verify_evidence)
from foam_evo.execution.replay import _load_evaluation_path
from foam_evo.execution.batch_runtime import BatchRuntimeFactory
from foam_evo.models import BatchRequest, BatchStatus, MergeRequest
from foam_evo.merge.service import MergeService, adopt_validation
from test_end_to_end import _fixture, _limits, _git, FakeMarkdownProvider

def outcome(case, before, after, **kw):
    return CaseOutcome(case_id=case, passed=all(after), reason='test', baseline_passed=all(before),
        baseline_checks=tuple({'check_id':str(i),'passed':x} for i,x in enumerate(before)),
        candidate_checks=tuple({'check_id':str(i),'passed':x} for i,x in enumerate(after)), **kw)

@pytest.mark.parametrize('kind', ['check_regression','overall_regression','missing','duplicate','infrastructure','no_improvement'])
def test_fail_closed_matrix(kind):
    repaired=outcome('failure',[False,True],[True,True])
    protected=outcome('protected',[True,True],[True,True])
    rows=[repaired,protected]
    if kind=='check_regression': rows[1]=outcome('protected',[True,True],[True,False])
    if kind=='overall_regression': rows[1]=CaseOutcome(**{**protected.__dict__,'passed':False})
    if kind=='missing': rows=rows[:1]
    if kind=='duplicate': rows.append(protected)
    if kind=='infrastructure': rows[1]=outcome('protected',[True,True],[True,True],infrastructure_blocked=True)
    if kind=='no_improvement': rows[0]=outcome('failure',[False,True],[False,True])
    with pytest.raises(ProtectionError): validate_matrix(EvaluationMatrix(tuple(rows)),['failure'],['protected'])

@pytest.mark.parametrize('checks,overall', [([],True),([{'check_id':'x','passed':1}],True),
    ([{'check_id':'x','passed':True},{'check_id':'x','passed':True}],True),
    ([{'check_id':'x','passed':True}],False)])
def test_malformed_evaluations_rejected(tmp_path, checks, overall):
    p=tmp_path/'evaluation.json'; write_json(p,{'schema_version':2,'checks':checks,'passed':overall})
    assert _load_evaluation_path(p)[1]

def test_frozen_manifest_and_contract(tmp_path):
    repo,base,_,_=_fixture(tmp_path,('case-a',)); suite=tmp_path/'suite.json'
    data,refs=suite_cases(suite)
    assert data['retention_ids']==['protected'] and data['failure_ids']==['case-a']
    (repo/'src/evaluation.py').write_text('# changed contract')
    with pytest.raises(ProtectionError,match='configuration changed'):
        evaluate_suite(repo=repo,suite_path=suite,failure_ids=['case-a'],output=tmp_path/'replay')
    refs['case-a'].prompt_path.write_text('changed')
    with pytest.raises(ProtectionError,match='frozen input'): suite_cases(suite)

class RegressingProvider(FakeMarkdownProvider):
    def run(self,**kwargs):
        result=super().run(**kwargs)
        if 'implementer' in kwargs['prompt'].splitlines()[0].lower():
            (Path(kwargs['cwd'])/'src/regress.py').write_text('REGRESSION=True')
        return result

def candidate(tmp_path,repo,base,sources,name='candidate',provider=None):
    return BatchRuntimeFactory(repo_root=repo,provider=provider or FakeMarkdownProvider(),source_cases=sources,
        artifact_root=tmp_path/'batches',worktree_root=tmp_path/'worktrees').run(
            BatchRequest(batch_id=name,repo_root=repo,base_commit=base,case_ids=['case-a'],
                         regression_suite=tmp_path/'suite.json',limits=_limits()))

def test_candidate_retention_regression_is_rejected(tmp_path):
    repo,base,_,sources=_fixture(tmp_path,('case-a',))
    result=candidate(tmp_path,repo,base,sources,provider=RegressingProvider())
    assert result.status==BatchStatus.FAILED and result.candidate_commit is None
    assert _git(repo,'rev-parse','main')==base

@pytest.fixture
def integrated(tmp_path):
    repo,base,_,sources=_fixture(tmp_path,('case-a',))
    c=candidate(tmp_path,repo,base,sources)
    assert c.status==BatchStatus.CANDIDATE_READY, c
    service=MergeService(repo,artifact_root=tmp_path/'integrations',merge_worktree_root=tmp_path/'merge-worktrees')
    request=MergeRequest(repo_root=repo,candidate_branches=[c.branch_name],candidate_evidence=[Path(c.validation_path)])
    result=service.run(request,merge_id='first')
    assert result.status=='VALIDATED',result
    return repo,base,c,result

def test_validated_adoption_accumulates_retention(tmp_path, integrated):
    repo,base,c,result=integrated
    commit=adopt_validation(repo,'main',result.validation_path)
    assert _git(repo,'rev-parse','main')==commit==result.merged_commit
    assert (repo/'src/mesh_fix.py').exists()
    assert (repo/'database/experience/index.jsonl').is_file()
    evidence=read_sealed(result.validation_path)
    assert (Path(evidence['worktree'])/'database/faiss/fixture.index').read_bytes()==b'frozen-tutorial-index'
    retained=read_sealed(retention_registry(repo))
    assert {r['ref']['case_id'] for r in retained['cases']}=={'case-a','protected'}
    _,sources=suite_cases(tmp_path/'suite.json')
    with pytest.raises(ProtectionError,match='omitted'):
        freeze_suite(repo=repo,base=commit,sources=sources,allowed_ids=['case-a'],path=tmp_path/'next.json')
    fresh=freeze_suite(repo=repo,base=commit,sources=sources,allowed_ids=['case-a','protected'],path=tmp_path/'next.json')
    assert fresh['retention_ids']==['case-a','protected'] and fresh['failure_ids']==[]

@pytest.mark.parametrize('change',['dirty','baseline','candidate','integrated','index','static_index'])
def test_adoption_rejects_changed_state(tmp_path,integrated,change):
    repo,base,c,result=integrated
    if change=='static_index': (repo/'database/faiss/fixture.index').write_bytes(b'changed')
    if change=='dirty': (repo/'untracked.txt').write_text('local work')
    if change=='baseline': _git(repo,'commit','--allow-empty','-m','concurrent baseline')
    if change=='candidate': _git(repo,'update-ref','refs/heads/'+c.branch_name,base)
    if change=='index':
        e=read_sealed(result.validation_path);(Path(e['worktree'])/'database/experience/index.jsonl').write_text('tampered')
    if change=='integrated':
        e=read_sealed(result.validation_path);(Path(e['worktree'])/'src/mesh_fix.py').write_text('FIXED=False')
    with pytest.raises(ProtectionError): adopt_validation(repo,'main',result.validation_path)
    assert _git(repo,'rev-parse','main')!=result.merged_commit
    assert not retention_registry(repo).exists()

def test_evidence_needs_review_and_exact_commit(tmp_path,integrated):
    repo,base,c,result=integrated
    evidence=read_sealed(c.validation_path)
    with pytest.raises(ProtectionError,match='different tested commit'):
        verify_evidence(c.validation_path,commit=base,base=base)
    evidence['iteration_verdict']='NEED_MORE_WORK'; p=tmp_path/'unreviewed.json';write_json(p,seal(evidence))
    with pytest.raises(ProtectionError,match='reviews are incomplete'):
        verify_evidence(p,commit=c.candidate_commit,base=base)

def test_combined_state_regression_blocks_integration(tmp_path):
    repo,base,_,sources=_fixture(tmp_path,('case-a',));c=candidate(tmp_path,repo,base,sources)
    class Integrator:
        def run(self,*,cwd,**kw):
            _git(cwd,'merge','--no-ff','-m','integrate',c.candidate_commit)
            (Path(cwd)/'src/regress.py').write_text('REGRESSION=True')
    service=MergeService(repo,artifact_root=tmp_path/'integrations',merge_worktree_root=tmp_path/'merge-worktrees',semantic_merger=Integrator())
    result=service.run(MergeRequest(repo_root=repo,candidate_branches=[c.branch_name],candidate_evidence=[Path(c.validation_path)]),merge_id='regresses')
    assert result.status=='FAILED' and 'regression' in result.error
    assert _git(repo,'rev-parse','main')==base
