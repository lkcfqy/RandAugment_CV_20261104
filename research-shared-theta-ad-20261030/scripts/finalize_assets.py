"""Archive final provenance and costs. No experiment or model computation."""
import calendar,collections,hashlib,json,pathlib,shutil,stat,subprocess,time,urllib.request
ROOT=pathlib.Path(__file__).resolve().parents[1]
extra=json.loads((ROOT/"logs/final-source-fetch.json").read_text()) if (ROOT/"logs/final-source-fetch.json").exists() else []
def small_get(url,dest,cap=4*1024*1024):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'AssetVerification/1.0'}),timeout=30) as r:
        data=r.read(cap+1);assert len(data)<=cap
        (ROOT/dest).write_bytes(data);entry={'url':url,'path':dest,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'http_status':r.status};extra.append(entry);return data
if not (ROOT/'sources/dinov2-upstream-commit.json').exists():
    small_get('https://api.github.com/repos/facebookresearch/dinov2/commits/main','sources/dinov2-upstream-commit.json')
upstream=json.loads((ROOT/'sources/dinov2-upstream-commit.json').read_text())['sha']
if not (ROOT/'models/dinov2-small/LICENSE-UPSTREAM').exists():
    small_get(f'https://raw.githubusercontent.com/facebookresearch/dinov2/{upstream}/LICENSE','models/dinov2-small/LICENSE-UPSTREAM')
(ROOT/'logs/final-source-fetch.json').write_text(json.dumps(extra,indent=2))
plan=json.loads((ROOT/'logs/dependency-install.json').read_text())
wheels=json.loads((ROOT/'reports/dependency-wheel-ledger.json').read_text()) if (ROOT/'reports/dependency-wheel-ledger.json').exists() else []
for entry in ([] if wheels else plan['install']):
    url=entry['download_info']['url']
    with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=30) as r:
        wheels.append({'name':entry['metadata']['name'],'version':entry['metadata']['version'],'url':url,'bytes':int(r.headers['Content-Length']),'sha256':entry['download_info']['archive_info']['hashes']['sha256']})
(ROOT/'reports/dependency-wheel-ledger.json').write_text(json.dumps(wheels,indent=2))
check=subprocess.run([str(ROOT/'.asset-venv/bin/python'),'-m','pip','check','--no-cache-dir'],capture_output=True,text=True)
assert check.returncode==0,check.stderr
(ROOT/'logs/pip-check.txt').write_text(check.stdout+check.stderr)
freeze=subprocess.run([str(ROOT/'.asset-venv/bin/python'),'-m','pip','freeze','--all','--no-cache-dir'],capture_output=True,text=True,check=True)
(ROOT/'reports/environment-freeze.txt').write_text(freeze.stdout)
commit=json.loads((ROOT/'sources/mmr-commit.json').read_text())
if not (ROOT/'sources/mmr-tree-pinned.json').exists():
    small_get(commit['commit']['tree']['url']+'?recursive=1','sources/mmr-tree-pinned.json')
(ROOT/'logs/final-source-fetch.json').write_text(json.dumps(extra,indent=2))
tree=json.loads((ROOT/'sources/mmr-tree-pinned.json').read_text())
assert tree['sha']==commit['commit']['tree']['sha']
checklist={'local_checkout':'/Users/lkc/Downloads/projects/RandAugment_CV_20261104','instructions':'No AGENTS.md or .agents/skills/SKILL.md found in checkout/ancestors or searched project/task trees','applicable_skill':'Dispatched asset task: using-superpowers SKILL.md contains SUBAGENT-STOP; no creative implementation or scientific protocol authored','existing_python_environment':str(ROOT.parent/'.venv'),'new_isolated_environment':str(ROOT/'.asset-venv'),'environment_reuse':'New venv includes read-only import path to existing project packages for torch/numpy/Pillow; additional official PyPI wheels installed only in new venv','existing_experiments_modified':False,'git_commit_or_push':False,'private_data_upload':False,'new_credentials':False,'purchased_compute':False,'colab_allocation':False,'gpu_operations':0,'no_safety_allowlist_added':True,'torchhub_calls':0,'pickle_checkpoints_loaded':0,'repository_commit':commit['sha'],'repository_tree_sha':tree['sha'],'repository_tree_matches_commit':True,'hf_revision':json.loads((ROOT/'sources/hf-model-info.json').read_text())['sha'],'dinov2_upstream_license_revision':upstream}
(ROOT/'reports/scope-and-environment.json').write_text(json.dumps(checklist,indent=2))
shutil.copyfile(ROOT/'sources/mmr-LICENSE-DATASET',ROOT/'data/LICENSE-DATASET')
(ROOT/'data/README-SOURCE.md').write_text('''# AeBAD-S official subset\n\nSource: https://github.com/zhangzilongc/MMR (commit '''+commit['sha']+''').\nOriginal publication: Zilong Zhang, Zhibin Zhao, Xingwu Zhang, Chuang Sun, Xuefeng Chen, “Industrial Anomaly Detection with Domain Shift: A Real-world Dataset and Masked Multi-scale Reconstruction”, https://arxiv.org/abs/2304.02216v2.\n\nDataset license: CC BY 4.0; the original LICENSE-DATASET is retained alongside this file.\nOfficial README download: https://drive.google.com/file/d/14wkZAFFeudlg0NMFLsiGwS0E593b-lNo/view .\n\nOnly AeBAD_S train/good and test/ground_truth same + illumination were extracted, using public HTTP Range requests from the original AeBAD.zip. Original image and mask bytes and names are unchanged. macOS dotfiles, resource forks and unrequested domains/video data are not extracted. 521 train images, 962 test images, 657 anomaly masks. The train set contains 520 distinct raw/decoded images because two published paths contain the same image; no file was removed.\n\nThe .png-named image files are actually JPEG-coded RGB; masks are PNG. Use byte-aware image decoding. No entity/batch mapping is supplied; filenames must not be treated as independent specimens.\n''')
source_files=[]
for folder in ('sources','models/dinov2-small'):
    for p in sorted((ROOT/folder).rglob('*')):
        if p.is_file():source_files.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(ROOT/'reports/source-artifact-manifest.json').write_text(json.dumps(source_files,indent=2))
acq=json.loads((ROOT/'reports/data-acquisition.json').read_text())
known_bytes=0
for logfile in ('initial-fetch.json','source-fetch.json','asset-probe-downloads.json','final-source-fetch.json'):
    known_bytes+=sum(x.get('bytes',0) for x in json.loads((ROOT/'logs'/logfile).read_text()))
known_bytes+=acq['downloaded_range_bytes']+sum(x['bytes'] for x in wheels)+(ROOT/'sources/aebad-drive-download-form.html').stat().st_size
size=0;allocated=0
for p in ROOT.rglob('*'):
    s=p.lstat()
    if stat.S_ISREG(s.st_mode):size+=s.st_size;allocated+=s.st_blocks*512
now=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());start='2026-10-30T21:47:23Z'
start_seconds=calendar.timegm(time.strptime(start,'%Y-%m-%dT%H:%M:%SZ'))
wall=time.time()-start_seconds
cost={'started_utc':start,'reported_utc':now,'elapsed_wall_seconds':wall,'wall_limit_seconds':1800,'data_archive_full_bytes':1616195913,'data_archive_full_GiB':1616195913/1024**3,'selected_http_range_bytes':acq['downloaded_range_bytes'],'selected_extracted_image_and_mask_bytes':1051850985,'model_weight_bytes':88249960,'installed_wheel_download_bytes':sum(x['bytes'] for x in wheels),'known_download_payload_bytes':known_bytes,'unmetered_browser_index_metadata_allowance_bytes':50*1024**2,'conservative_download_upper_bound_bytes':known_bytes+50*1024**2,'download_limit_bytes':2*1024**3,'retained_regular_file_bytes_before_final_report':size,'allocated_disk_bytes_before_final_report':allocated,'disk_peak_conservative_bound_bytes':max(size,allocated)+100*1024**2,'disk_peak_bound_reason':'All dataset ranges and extracted files retained; no full ZIP or second model copy. Add 100 MiB allowance for temporary package install/model partial files and final reports. Temporary model/wheel downloads precede data growth. No cache retained.','disk_limit_bytes':3*1024**3,'gpu_seconds':0,'colab_allocations':0,'data_acquisition_seconds':acq['elapsed_seconds'],'file_audit_seconds':json.loads((ROOT/'reports/data-integrity-summary.json').read_text())['elapsed_seconds'],'model_cpu_structural_check_seconds':json.loads((ROOT/'reports/model-cpu-verification.json').read_text())['wall_seconds'],'future_feature_bank_arithmetic_only':{'training_images_as_published':521,'patches_per_image':256,'feature_dim':384,'float32_bytes_per_scalar':4,'patch_vectors_per_transform':521*256,'float32_feature_bank_bytes_per_transform':521*256*384*4,'distance_matrix_bytes_per_query_image_per_transform_if_materialized':256*521*256*4,'not_a_measured_runtime_forecast':True,'augmentation_count_and_compute_protocol':'pending independent review'}}
assert cost['elapsed_wall_seconds']<1800
assert cost['conservative_download_upper_bound_bytes']<2*1024**3
assert cost['disk_peak_conservative_bound_bytes']<3*1024**3
(ROOT/'reports/cost-ledger.json').write_text(json.dumps(cost,indent=2))
print(json.dumps(cost,indent=2))
