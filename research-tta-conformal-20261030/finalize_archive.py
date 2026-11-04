from portable_dev.paths import project_path as _portable_project_path
import datetime,hashlib,json,shutil,time
from pathlib import Path
R=Path(__file__).resolve().parent
DEST=_portable_project_path(__file__, 'research-tta-conformal-20261030')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
release=json.loads((R/'released.json').read_text());assert release['release_confirmed']
collection=json.loads((R/'collection-verified.json').read_text());assert collection['verified']
decision=json.loads((R/'analysis/decision.json').read_text())
frozen=json.loads((R/'protocol-lock.json').read_text())
assert all(sha(R/name)==digest for name,digest in frozen['files'].items()),'Frozen protocol/code changed'
download=next(json.loads(l) for l in (R/'results/events.jsonl').read_text().splitlines() if json.loads(l)['event']=='download_started')
allocation=json.loads((R/'cli-ledger/allocate-t4.json').read_text())['started_epoch']
download_epoch=allocation+download['allocation_elapsed_seconds']
wall=time.time()-download_epoch
assert wall<6*3600,'Six hour wall deadline exceeded'
(R/'STATUS_ZH.md').write_text(f'''# 完成状态

更新时间：{datetime.datetime.now(datetime.timezone.utc).isoformat()}。

独立seed0 ResNet20完成80epoch、25040正式步；测速200步；205120推理views。CPU数学验收9项通过，缓存/顺序/均值与真实rankonly一致性验收通过。999次配对分组bootstrap已完成。

本候选决定：{decision['candidate_status']}。A门={decision['A_pass']}；B主门={decision['B_primary_frontier_and_restore_pass']}，反证后B={decision['B_pass_after_controls']}。这只关闭/审查本候选，不停止整体探索；没有自动追加seed、模型或视图。

T4已释放并验证。{release['sessions_output'].strip()}

{release['usage_output'].strip()}

完整中文报告见REPORT_ZH.md，资源账本见RESOURCE_LEDGER.json。一次Jupyter只读状态查询超时后，经会话核对和官方文件下载确认训练未中断，已恢复监控，未重跑实验。没有待处理执行阻塞。
''')
skip={'__pycache__','matplotlib-cache'}
copied=[]
for src in sorted(R.rglob('*')):
    if not src.is_file():continue
    rel=src.relative_to(R)
    if any(part in skip for part in rel.parts) or rel.name in ('results.tar','archive-receipt.json','SHA256SUMS.json'):continue
    dst=DEST/rel;dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists() and sha(src)!=sha(dst) and rel.as_posix()!='STATUS_ZH.md':
        raise RuntimeError('Refusing to overwrite different historical artifact '+str(dst))
    if not dst.exists() or sha(src)!=sha(dst):shutil.copy2(src,dst)
    copied.append(rel.as_posix())
archive_bytes=sum(p.stat().st_size for p in DEST.rglob('*') if p.is_file())
staging_bytes=sum(p.stat().st_size for p in R.rglob('*') if p.is_file())
assert archive_bytes+staging_bytes<2*1024**3,'Combined local additions exceed 2GiB'
receipt={'archived_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'destination':str(DEST),
         'copied_relative_paths':copied,'archive_bytes_before_receipt_and_hashes':archive_bytes,
         'staging_bytes':staging_bytes,'combined_local_bytes':archive_bytes+staging_bytes,
         'first_download_to_archive_seconds':time.time()-download_epoch,
         'frozen_hashes_verified':True,'release_confirmed':True,'no_commit_push':True,
         'candidate_status':decision['candidate_status'],'new_experiments_started_after_candidate':False}
(DEST/'archive-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
hashes={p.relative_to(DEST).as_posix():sha(p) for p in sorted(DEST.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.json'}
(DEST/'SHA256SUMS.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2)+'\n')
assert all(sha(DEST/n)==h for n,h in hashes.items())
print(json.dumps({'archive':str(DEST),'status':decision['candidate_status'],'files':len(hashes),
                  'combined_local_MiB':(archive_bytes+staging_bytes)/1024**2,
                  'wall_minutes':receipt['first_download_to_archive_seconds']/60},ensure_ascii=False,indent=2))
