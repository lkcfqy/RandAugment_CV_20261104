"""Bounded public asset fetcher; no remote code execution or model inference."""
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
DATA_REV = '761a8e0320fa8e818e5929e0cba269015cd60e06'
MODEL_REV = '51481ef2546f870060c43eb6d6525399f5b3d2d3'
DATA_SHA = 'cf14eaf7ac287a30822fd626cba76b255f39e416b95bcc089e48c7aa91503112'
MODEL_SHA = '2291407125e809cc9c0614cc2d010d21d309a66eb3da33e1ee2386a68fa05894'
BUDGET = 495 * 1024**2  # reserve 5 MiB for connector/web source inspection
LOG = ROOT / 'logs/downloads.json'
records = json.loads(LOG.read_text()) if LOG.exists() else []
downloaded = sum(r.get('bytes_received', 0) for r in records)

def stamp():
    return datetime.now(timezone.utc).isoformat()

def save_log():
    LOG.write_text(json.dumps(records, indent=2) + '\n')

def fetch(url, rel, limit, expected=None, optional=False):
    global downloaded
    out = ROOT / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        if expected and hashlib.sha256(out.read_bytes()).hexdigest() != expected:
            raise RuntimeError('Existing file hash mismatch: ' + rel)
        return out
    assert url.startswith('https://')
    for attempt in (1, 2):
        rec = dict(url=url, path=rel, attempt=attempt, started_utc=stamp(), bytes_received=0)
        t0 = time.monotonic()
        tmp = out.with_name(out.name + '.partial')
        try:
            req = urllib.request.Request(url, headers={'User-Agent':'MILL-proxy-asset-verification/1.0'})
            with urllib.request.urlopen(req, timeout=40) as response, tmp.open('xb') as f:
                rec['final_url_origin'] = urllib.request.urlparse(response.url).netloc if hasattr(urllib.request, 'urlparse') else response.url.split('/')[2]
                rec['content_length'] = response.headers.get('Content-Length')
                if rec['content_length'] and int(rec['content_length']) > limit:
                    raise RuntimeError('Content-Length exceeds per-file budget')
                h = hashlib.sha256()
                while True:
                    b = response.read(1024 * 1024)
                    if not b:
                        break
                    downloaded += len(b)
                    rec['bytes_received'] += len(b)
                    if downloaded > BUDGET or rec['bytes_received'] > limit:
                        raise RuntimeError('Download budget exceeded')
                    h.update(b)
                    f.write(b)
                rec['sha256'] = h.hexdigest()
                if expected and rec['sha256'] != expected:
                    raise RuntimeError('SHA256 mismatch: ' + rel)
            tmp.rename(out)
            rec['status'] = 'verified' if expected else 'downloaded'
            rec['wall_seconds'] = round(time.monotonic()-t0, 3)
            records.append(rec); save_log()
            print(json.dumps({k:rec[k] for k in ['path','bytes_received','sha256','status','wall_seconds']}), flush=True)
            return out
        except Exception as e:
            rec['status'] = 'failed'
            rec['error'] = repr(e)
            rec['wall_seconds'] = round(time.monotonic()-t0, 3)
            records.append(rec); save_log()
            print(json.dumps(rec), flush=True)
            # Only remove this attempt's incomplete temporary file.
            if tmp.exists(): tmp.unlink()
            ordinary = isinstance(e, (TimeoutError, urllib.error.URLError)) and not isinstance(e, urllib.error.HTTPError)
            ordinary = ordinary or (isinstance(e, urllib.error.HTTPError) and e.code in (408,429,500,502,503,504))
            denied = isinstance(e, urllib.error.HTTPError) and e.code in (401, 403)
            if ordinary and not denied and attempt == 1: continue
            if optional: return None
            raise

def main():
    sm = 4 * 1024**2
    mill_meta = json.loads(fetch('https://api.github.com/repos/CVC-Color/MILL/commits/main','sources/mill_commit.json',sm).read_text())
    mill_commit = mill_meta['sha']
    mill_readme = fetch(f'https://raw.githubusercontent.com/CVC-Color/MILL/{mill_commit}/README.md','sources/MILL_README.md',sm)
    assert 'huggingface.co/datasets/davidserra9/MILL' in mill_readme.read_text()
    fetch('https://color.cvc.uab.cat/mill/','sources/mill_project.html',sm,optional=True)
    fetch(f'https://huggingface.co/api/datasets/davidserra9/MILL/revision/{DATA_REV}','sources/mill_hf_metadata.json',sm)
    fetch(f'https://huggingface.co/api/datasets/davidserra9/MILL/tree/{DATA_REV}','sources/mill_hf_tree.json',sm)
    card=fetch(f'https://huggingface.co/datasets/davidserra9/MILL/raw/{DATA_REV}/README.md','sources/MILL_DATA_CARD.md',sm)
    assert 'license: mit' in card.read_text()
    hvi_meta=json.loads(fetch('https://api.github.com/repos/Fediory/HVI-CIDNet/commits/master','sources/hvi_commit.json',sm).read_text())
    hvi_commit=hvi_meta['sha']
    fetch(f'https://api.github.com/repos/Fediory/HVI-CIDNet/git/trees/{hvi_commit}?recursive=1','sources/hvi_tree.json',sm)
    fetch(f'https://raw.githubusercontent.com/Fediory/HVI-CIDNet/{hvi_commit}/Readme.md','sources/HVI_README.md',sm)
    fetch(f'https://raw.githubusercontent.com/Fediory/HVI-CIDNet/{hvi_commit}/LICENSE','sources/HVI_LICENSE',sm)
    for name in ['net/CIDNet.py','net/HVI_transform.py','net/LCA.py','net/transformer_utils.py','eval_hf.py','eval.py','measure.py','data/options.py','requirements.txt']:
        fetch(f'https://raw.githubusercontent.com/Fediory/HVI-CIDNet/{hvi_commit}/{name}','vendor/HVI-CIDNet/'+name,sm)
    fetch('https://arxiv.org/html/2511.15496v1','sources/mill_paper_v1.html',sm,optional=True)
    fetch(f'https://huggingface.co/api/models/Fediory/HVI-CIDNet-Generalization/revision/{MODEL_REV}','sources/hvi_hf_metadata.json',sm)
    fetch(f'https://huggingface.co/api/models/Fediory/HVI-CIDNet-Generalization/tree/{MODEL_REV}','sources/hvi_hf_tree.json',sm)
    fetch(f'https://huggingface.co/Fediory/HVI-CIDNet-Generalization/raw/{MODEL_REV}/README.md','sources/HVI_MODEL_CARD.md',sm)
    fetch(f'https://huggingface.co/Fediory/HVI-CIDNet-Generalization/resolve/{MODEL_REV}/config.json','models/config.json',sm)
    pins={'data_revision':DATA_REV,'model_revision':MODEL_REV,'mill_github_commit':mill_commit,'hvi_github_commit':hvi_commit,'dataset_sha256':DATA_SHA,'model_sha256':MODEL_SHA,'prepared_utc':stamp()}
    (ROOT/'reports/source_pins.json').write_text(json.dumps(pins,indent=2)+'\n')
    fetch(f'https://huggingface.co/datasets/davidserra9/MILL/resolve/{DATA_REV}/MILLs-DSLR.zip','downloads/MILLs-DSLR.zip',230*1024**2,DATA_SHA)
    fetch(f'https://huggingface.co/Fediory/HVI-CIDNet-Generalization/resolve/{MODEL_REV}/model.safetensors','models/model.safetensors',12*1024**2,MODEL_SHA)
    print('TOTAL_DOWNLOADED_BYTES',downloaded,flush=True)

if __name__ == '__main__': main()
