import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import importlib.metadata
from pathlib import Path
from stheta.protocol import WORK,SPEC,save_json,file_sha
from stheta.data import build_split_manifest

if __name__=='__main__':
    before={'protocol':SPEC,'data_code_sha256':file_sha(Path(__file__).parent/'stheta/data.py'),
      'protocol_code_sha256':file_sha(Path(__file__).parent/'stheta/protocol.py'),
      'versions':{n:importlib.metadata.version(n) for n in ('ImageHash','scikit-image','Pillow','numpy','scipy')},
      'purpose':'Fixed before any near-duplicate result; no image model computations'}
    target=WORK/'pre-data-manifest.json'
    if target.exists():
        import json
        assert json.loads(target.read_text())==json.loads(json.dumps(before)),'Data definition changed after freeze; stop for review'
    else:save_json(target,before)
    build_split_manifest()
