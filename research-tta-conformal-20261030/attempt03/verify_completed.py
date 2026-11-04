"""Post-release local durability and identity checks; no model execution."""
from pathlib import Path
import datetime,hashlib,json,pickle,tarfile,time
import numpy as np
from state_io import atomic_json,sha_file,validate_checkpoint

R=Path(__file__).resolve().parent
def read(name):return json.loads((R/name).read_text())
start=time.time();release=read('released.json')
assert release['verified_zero'] and release['active_assignments']==0 and release['rate']==0
identity=read('results/environment.json')
assert identity['protocol_sha256']==sha_file(R.parent/'PROTOCOL_ZH.md')
assert identity['science_source_sha256']==sha_file(R.parent/'remote_experiment.py')
assert identity['split_manifest_sha256']==sha_file(R/'results/split-manifest.jsonl')
checked=[]
for epoch in range(0,81,5):
    p=R/'commits'/f'formal{epoch:03d}.pt';receipt=read(f'commits/formal{epoch:03d}.receipt.json')
    data=validate_checkpoint(p,receipt,identity);s=data['state']
    assert s['phase']=='formal' and s['step']==epoch*313 and s['epoch']==epoch and s['cursor']==0
    assert s['presentations']==epoch*40000 and s['scheduler']['next_global_step']==epoch*313
    assert s['permutation'] is None and s['training_mode'] is True
    checked.append({'epoch':epoch,'step':s['step'],'sha256':receipt['sha256'],'state_sha256':receipt['state_content_sha256']})
    del data,s
ids=np.load(R/'results/heldout-image-ids.npy');labels=np.load(R/'results/heldout-labels.npy')
split=np.load(R/'results/split-ids.npz');expected=np.concatenate([split[k] for k in ('pseudocal','validation','cal','eval')])
assert np.array_equal(ids,expected) and len(np.unique(ids))==20000
assert np.array_equal(np.load(R/'results/view-ids.npy'),np.arange(10))
archive=Path('/Users/lkc/Documents/Codex/2026-10-31/task/tta-conformal/recovered-data/cifar-100-python.tar.gz')
assert sha_file(archive)==identity['dataset_sha256']
with tarfile.open(archive,'r:gz') as tf:
    tr=pickle.load(tf.extractfile('cifar-100-python/train'),encoding='bytes')
    truth=list(tr[b'fine_labels']);del tr
    te=pickle.load(tf.extractfile('cifar-100-python/test'),encoding='bytes')
    truth+=list(te[b'fine_labels']);del te
assert np.array_equal(labels,np.asarray(truth,dtype=np.int64)[ids])
raw=np.load(R/'results/logits-fp32.npy',mmap_mode='r')
assert raw.shape==(20000,10,100) and raw.dtype==np.float32 and np.isfinite(raw).all()
for index in range(40):
    receipt=read(f'commits/infer{index:03d}.receipt.json');p=R/'commits'/f'infer{index:03d}.npy'
    assert receipt['identity']==identity and receipt['checkpoint_sha256']==checked[-1]['sha256']
    assert sha_file(p)==receipt['sha256'] and p.stat().st_size==receipt['bytes']
    a=np.load(p,mmap_mode='r');lo,hi=receipt['row_start'],receipt['row_end']
    assert lo==index*512 and hi==min((index+1)*512,20000)
    assert receipt['image_ids']==ids[lo:hi].tolist() and receipt['view_ids']==list(range(10)) and receipt['class_ids']==list(range(100))
    assert hashlib.sha256(a.tobytes()).hexdigest()==receipt['logits_bytes_sha256']
    assert np.array_equal(raw[lo:hi],a)
assert np.array_equal(np.load(R/'results/baseline-logits-fp32.npy'),raw[:,4,:])
assert np.array_equal(np.load(R/'results/mean-logits-f64.npy'),raw.mean(1,dtype=np.float64))
old={}
for line in (R.parent/'monitor/remote-process-019.log').read_text().splitlines():
    try:o=json.loads(line)
    except json.JSONDecodeError:continue
    if o.get('event')=='epoch_done':old[o['epoch']]=o
epochs=[]
for target in range(5,81,5):epochs.extend(read(f'commits/train{target:03d}.done.json')['epochs'])
assert [e['epoch'] for e in epochs]==list(range(1,81))
comparison=[]
for e in epochs:
    if e['epoch'] in old:
        comparison.append({'epoch':e['epoch'],'steps':e['formal_steps'],
          'step_equal':e['formal_steps']==old[e['epoch']]['formal_steps'],
          'lr_equal':e['lr']==old[e['epoch']]['lr'],'loss_equal':e['train_loss']==old[e['epoch']]['train_loss'],
          'retry_train_loss':e['train_loss'],'first_train_loss':old[e['epoch']]['train_loss']})
atomic_json(R/'trajectory-continuity.json',{'comparison_scope':'Training logs only; no scientific result selection.',
 'compared_original_epochs':len(comparison),'all_recorded_steps_lr_losses_exact':all(x['step_equal'] and x['lr_equal'] and x['loss_equal'] for x in comparison),
 'rows':comparison,'adopted_trajectory':'attempt03_formal_designated regardless of first run later recovery',
 'final_epoch80':epochs[-1]})
atomic_json(R/'durability-final-verification.json',{'passed':True,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'GPU_already_released':True,'new_training_steps':0,'new_inference_views':0,'checkpoints':checked,
 'official_archive_and_labels_verified':True,'full_split_identity_verified':True,'all40_shards_and_aggregate_bitwise_equal':True,
 'baseline_center_and_float64_mean_exact':True,'raw_logits_sha256':sha_file(R/'results/logits-fp32.npy'),
 'CPU_seconds':time.time()-start})
print(json.dumps({'post_release_durability':'PASS','checkpoints':len(checked),'shards':40,'prefix_compared_epochs':len(comparison),
 'prefix_all_exact':all(x['step_equal'] and x['lr_equal'] and x['loss_equal'] for x in comparison),'seconds':time.time()-start}),flush=True)
