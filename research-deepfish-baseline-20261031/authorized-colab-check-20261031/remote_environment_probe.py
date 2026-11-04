"""Read-only prerequisites: no framework import, install, model or user-data access."""
import json,os,pathlib,platform,shutil,subprocess,sys,time
candidate=pathlib.Path(sys.executable) if sys.version_info[:2]==(3,10) else pathlib.Path('/usr/bin/python3.10')
python310=None
if candidate.is_file():
 r=subprocess.run([str(candidate),'-B','-c','import sys,json;print(json.dumps(list(sys.version_info[:3])))'],capture_output=True,text=True,timeout=5)
 if r.returncode==0:python310=json.loads(r.stdout)
mounts=[]
for line in pathlib.Path('/proc/mounts').read_text().splitlines():
 parts=line.split()
 if len(parts)>=4 and parts[2] in ('fuse.drivefs','fuse.gcsfuse','gcsfuse','nfs','nfs4','cifs'):
  mounts.append({'mountpoint':parts[1],'fstype':parts[2],'read_write':'rw' in parts[3].split(',')})
gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],capture_output=True,text=True,timeout=10)
result={'epoch':time.time(),'platform':platform.system(),'machine':platform.machine(),'kernel_python':platform.python_version(),'python310_candidate_exists':candidate.is_file(),'python310_version':python310,'python310_compatible':python310 is not None and python310[:2]==[3,10],'free_bytes':shutil.disk_usage('/content').free,'persistent_mount_candidates':mounts,'persistent_storage_verified':False,'native_management_env_present':bool(os.environ.get('TBE_RUNTIME_ADDR')),'gpu_query_returncode':gpu.returncode,'gpu':gpu.stdout.strip(),'model_imported':False,'dependencies_installed':False,'package_uploaded':False,'test_loaded':False}
print('DEEPFISH_ENVIRONMENT '+json.dumps(result,sort_keys=True),flush=True)
