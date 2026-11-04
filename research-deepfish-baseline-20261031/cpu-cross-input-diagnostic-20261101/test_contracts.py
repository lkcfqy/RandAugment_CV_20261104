"""Artificial identity/persistence/metric tests and static check; no models built."""
import ast
import contextlib
import copy
import errno
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
from unittest import mock
import diagnostic_contracts as c

passed=[]
def rejects(call,kind=c.ContractError):
    try:call()
    except kind:return
    raise AssertionError('Expected rejection absent')

models=[{'seed':s,'arm':a} for s in (20261031,20261101) for a in ('A','B')]
ids=c.expected_ids(models,list(range(64)))
assert len(ids)==len(set(ids))==512
c.require_complete_ids(list(reversed(ids)),ids)
rejects(lambda:c.require_complete_ids(ids[:-1],ids))
rejects(lambda:c.require_complete_ids(ids+[ids[0]],ids))
wrong=ids.copy();wrong[-1]=c.identity(20261101,'B','c080',9999)
rejects(lambda:c.require_complete_ids(wrong,ids))
passed.append('512 unique matrix; missing/duplicate/extra identities rejected')

key=c.identity(20261031,'A','c100',0)
spec={'seed':20261031,'arm':'A','condition':'c100','factor':1.,'image_id':0,
      'checkpoint_sha256':'0'*64,'config_sha256':'1'*64,'image_sha256':'2'*64}
run={'identity_map':{key:spec},'dev_annotation':{'sha256':'3'*64},'environment_lock_sha256':'4'*64}
payload={**spec,'identity':key,'annotation_sha256':'3'*64,'environment_lock_sha256':'4'*64,
         'predictions':[{'image_id':0,'category_id':1,'bbox':[1.,2.,3.,4.],'score':.5} for _ in range(300)]}
c.check_payload(payload,key,run)
bad=copy.deepcopy(payload);bad['checkpoint_sha256']='f'*64
rejects(lambda:c.check_payload(bad,key,run))
bad2=copy.deepcopy(payload);bad2['predictions'][0]['score']=float('nan')
rejects(lambda:c.check_payload(bad2,key,run))
bad3=copy.deepcopy(payload);bad3['predictions'].pop()
rejects(lambda:c.check_payload(bad3,key,run))
passed.append('Checkpoint/image/condition provenance and300 finite unfiltered predictions enforced')

with tempfile.TemporaryDirectory(prefix='synthetic-',dir=c.P/'tmp') as temp:
    folder=pathlib.Path(temp);target=folder/'atomic.json'
    old={'synthetic':True,'version':1};new={'synthetic':True,'version':2}
    c.atomic_json(target,old,new=True)
    assert c.atomic_json(target,old,new=True)['existing_exact']
    rejects(lambda:c.atomic_json(target,new,new=True))
    with mock.patch.object(c.os,'replace',side_effect=OSError(errno.EIO,'synthetic publication interruption')):
        rejects(lambda:c.atomic_json(target,new),OSError)
    assert json.loads(target.read_text())==old
    assert target.with_name(target.name+'.partial').exists()
    c.atomic_json(target,new)
    assert json.loads(target.read_text())==new
    childfile=folder/'child.json'
    command='import pathlib;from diagnostic_contracts import atomic_json;atomic_json(pathlib.Path('+repr(str(childfile))+'),{"synthetic":True,"child_exit_persistence":True},new=True)'
    subprocess.run([sys.executable,'-B','-c',command],cwd=c.P,check=True,timeout=30)
    assert json.loads(childfile.read_text())['child_exit_persistence']
passed.append('Atomic fsync/replace/readback; injected I/O failure preserves prior state; exact retry and child-exit persistence')

rejects(lambda:c.Guard({'deadline_monotonic':time.monotonic()-1}).check(),c.ResourceStop)
with mock.patch.object(c,'tree_bytes',return_value=100*1024**2+1):
    rejects(lambda:c.Guard({'deadline_monotonic':time.monotonic()+60}).check(),c.ResourceStop)
with mock.patch.object(c,'tree_bytes',side_effect=lambda path:1024**2 if path==c.P else 6*1024**3-79*1024**2):
    rejects(lambda:c.Guard({'deadline_monotonic':time.monotonic()+60}).check(),c.ResourceStop)
passed.append('Wall-time,100MiB output cap and retained80MiB research margin stop before computation')

import numpy as np
from pycocotools.cocoeval import COCOeval
ev=COCOeval(iouType='bbox');ev.params.maxDets=[100,300,1000]
precision=np.zeros((10,101,1,4,3));recall=np.zeros((10,1,4,3))
for i,value in enumerate([.1,.3,.9]):precision[...,i]=value;recall[...,i]=value
ev.eval={'precision':precision,'recall':recall}
with contextlib.redirect_stdout(io.StringIO()):ev.summarize()
mapped=c.mapped_stats(ev.stats)
assert np.isclose(mapped['AP']['value'],.1) and mapped['AP']['maxDets']==100
assert np.isclose(mapped['AP50']['value'],.9) and mapped['AP50']['maxDets']==1000
for name,value,maxdets in [('AR100',.1,100),('AR300',.3,300),('AR1000',.9,1000)]:
    assert np.isclose(mapped[name]['value'],value) and mapped[name]['maxDets']==maxdets
assert c.array_digest(precision)==c.array_digest(precision.copy())
altered=precision.copy();altered.flat[0]+=.001;assert c.array_digest(precision)!=c.array_digest(altered)
passed.append('Artificial COCO arrays with distinct maxDets values verify original AP and all AR mappings')

from mmdet.evaluation.metrics.coco_metric import CocoMetric
assert CocoMetric.xyxy2xywh(None,np.array([1.125,2.5,7.,12.25],dtype=np.float32))==[1.125,2.5,5.875,9.75]
passed.append('Pinned official bbox conversion arity and arithmetic verified without an evaluator/model run')

tree=ast.parse((c.P/'formal_cpu.py').read_text())
calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
assert not any(isinstance(n.func,ast.Attribute) and n.func.attr in ('train','backward','cuda','mps') for n in calls)
loads=[n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='load' and isinstance(n.func.value,ast.Name) and n.func.value.id=='torch']
assert len(loads)==1
kw={x.arg:x.value for x in loads[0].keywords}
assert isinstance(kw['weights_only'],ast.Constant) and kw['weights_only'].value is True
assert isinstance(kw['map_location'],ast.Constant) and kw['map_location'].value=='cpu'
assert len([n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='test_step'])==1
assert json.loads((c.P/'RUN_PROTOCOL.json').read_text())['expected_identities']
passed.append('Static: one inference call site; explicit safeCPU weights; no training/backward/GPU/MPS call')

run=c.read_execution()
for name,expected in run['code_sha256'].items():assert c.sha(c.P/name)==expected
result={'status':'PASS','checks':passed,'check_count':len(passed),'model_instances':0,'model_forward_calls':0,
        'real_dev_predictions_read':False,'artificial_test_data_only':True,
        'run_protocol_sha256':c.sha(c.P/'RUN_PROTOCOL.json')}
c.atomic_json(c.P/'TEST_RESULTS.json',result,new=True)
print(json.dumps(result))
