"""Declared CFD applicability and serialized metadata round-trip tests."""
import importlib.util
import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from experience_conditions import applicability

def test_explicit_conditions_and_unknown_values():
    meta={'solver':'simpleFoam','fork':'foundation','version':'10',
          'conditions':{'regime':['laminar','turbulent'],'reynolds':{'min':100,'max':200}}}
    kwargs=dict(solver='simpleFoam',fork='foundation',version='10',context={'regime':'LAMINAR','reynolds':100})
    assert applicability(meta,**kwargs)[0]
    for key,value in [('solver','pimpleFoam'),('version',''),('fork','esi'),('context',{'regime':'laminar','reynolds':201}),('context',{})]:
        assert not applicability(meta,**{**kwargs,key:value})[0]

@pytest.mark.parametrize('conditions',[[], 'broken', {'x':{'min':2,'max':1}}, {'x':{'min':'nan'}}, {'x':[]}, {'x':[None,'laminar']}, {'x':{'min':'inf'}}, {'x':{'min':True}}, {'x':None}])
def test_malformed_or_unsatisfied_condition_is_not_retrieved(conditions):
    assert not applicability({'conditions':conditions},context={'x':1})[0]

def test_any_is_explicit_and_retired_is_blocked():
    assert applicability({'solver':'any','fork':'any','conditions':{'regime':'any'}})[0]
    assert not applicability({'solver':'simpleFoam'})[0]
    assert not applicability({'status':'retired'})[0]

def test_store_indexer_retrieval_round_trip(tmp_path,monkeypatch):
    root=Path(__file__).resolve().parents[1]
    refiner=root.parent/'Foam-Refiner'
    if not (refiner/'foam_evo').is_dir(): pytest.skip('companion Foam-Refiner checkout is required')
    sys.path.insert(0,str(refiner))
    from foam_evo.experience.store import ExperienceStore
    (tmp_path/'database').mkdir();store=ExperienceStore(tmp_path)
    conditions={'regime':'laminar','reynolds':{'max':1000}}
    key=store.add({'trigger_type':'flow','solver':'simpleFoam','fork':'foundation','conditions':conditions,'guidance':'Use only for the declared setup.'},source_batch='batch-test')
    assert store.rebuild_ledger()[0]['conditions']==conditions
    spec=importlib.util.spec_from_file_location('experience_indexer',root/'database/script/init_experience_db.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    docs=module.load_entries(store.entries)
    assert docs[0].metadata['version']==''
    import utils
    class Db:
        def similarity_search_with_score(self,q,k): return [(docs[0],0.1)]
    monkeypatch.setattr(utils,'EXPERIENCE_DB_CACHE',None)
    monkeypatch.setattr(utils,'_EXPERIENCE_DB_LOADED',False)
    monkeypatch.setattr(utils,'load_experience_db',lambda config=None:Db())
    kwargs=dict(solver='simpleFoam',fork='foundation',context={'regime':'laminar','reynolds':100})
    assert utils.retrieve_experience('flow',**kwargs)[0]['entry_id']==key
    assert utils.retrieve_experience('flow',**{**kwargs,'context':{'regime':'turbulent','reynolds':100}})==[]
