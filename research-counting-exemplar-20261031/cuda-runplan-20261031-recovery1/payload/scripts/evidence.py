"""Atomic remote/local evidence with a content-addressed stdout copy."""
import hashlib,json,os,tempfile
from pathlib import Path
from contracts import make_packet,validate_artifact_path

def owned_size(root):
    total=0
    for path in Path(root).rglob('*'):
        try:
            if path.is_file():total+=path.stat().st_size
        except FileNotFoundError:pass  # Another thread atomically renamed a small state file.
    return total

def atomic_bytes(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix=path.name+'.',delete=False) as stream:
        tmp=Path(stream.name);stream.write(data);stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path)

class Evidence:
    def __init__(self,root,owned_root=None,disk_cap=6*1024**3):
        self.root=Path(root);self.index={};self.owned_root=Path(owned_root) if owned_root is not None else None;self.disk_cap=disk_cap
    def write(self,path,value,emit=True):
        path=validate_artifact_path(path)
        data=(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
        if self.owned_root is not None:
            used=owned_size(self.owned_root)
            if used+len(data)>self.disk_cap:raise RuntimeError('owned disk cap would be exceeded by evidence write')
        atomic_bytes(self.root/path,data)
        self.index[path]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
        if emit:print(make_packet(path,data),flush=True)
        return self.index[path]
    def finish_index(self):
        self.write('artifact_index.json',{'files':dict(self.index)})
