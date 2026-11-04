"""Independent process: derive reference input hashes from the frozen first-run code."""
from portable_dev.paths import project_path as _portable_project_path
import sys,json,hashlib,zipfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OLD=_portable_project_path(__file__, 'augmentation-selection-pilot-20261102')
sys.path.insert(0,str(OLD))
import common as reference
freeze=json.loads((OLD/"CODE_FREEZE.json").read_text())
assert reference.sha(OLD/"common.py")==freeze["code"]["common.py"]
m=json.loads((ROOT/"IDENTITIES.json").read_text());t=time.perf_counter();rows=[]
with zipfile.ZipFile(reference.ZIP) as z:
    for entry in m["support_union"]:
        views=reference.make_views(reference.read_image(z,entry),entry["rgb_sha256"])
        rows.append({"member":entry["member"],"weak_ra_sha256":[hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in views[1:]]})
payload={"old_common_sha256":freeze["code"]["common.py"],"rows":rows,"seconds":time.perf_counter()-t,"model_forwards":0}
(ROOT/"REFERENCE_VIEW_HASHES.json").write_text(json.dumps(payload,indent=2)+"\n")
print(json.dumps({"originals":len(rows),"view_hashes":len(rows)*8,"seconds":payload["seconds"]}))

