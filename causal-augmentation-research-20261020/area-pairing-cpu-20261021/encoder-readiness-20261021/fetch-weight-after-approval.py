"""Proposed official weight fetch. NOT executed during readiness inspection."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import sys
import urllib.request

URL='https://download.pytorch.org/models/resnet18-f37072fd.pth'
EXPECTED_BYTES=46830571


def main():
    assert len(sys.argv)==2
    root=Path(sys.argv[1]).resolve()
    directory=root/'torch-cache/hub/checkpoints'
    directory.mkdir(parents=True,exist_ok=True)
    target=directory/'resnet18-f37072fd.pth'
    partial=directory/'resnet18-f37072fd.pth.partial'
    assert not target.exists() and not partial.exists(),'Preserving existing or uncertain outcome'
    checksum=hashlib.sha256();count=0
    with urllib.request.urlopen(urllib.request.Request(URL,headers={'User-Agent':'RandAugment-CPU-encoder/1.0'}),timeout=30) as response:
        assert response.status==200 and int(response.headers['Content-Length'])==EXPECTED_BYTES
        with partial.open('xb') as f:
            while True:
                block=response.read(1024*1024)
                if not block:break
                count+=len(block);assert count<=EXPECTED_BYTES
                checksum.update(block);f.write(block)
    digest=checksum.hexdigest()
    assert count==EXPECTED_BYTES and digest.startswith('f37072fd')
    # Atomic no-overwrite publication of our own newly created download.
    os.link(partial,target);partial.unlink()
    receipt={'status':'approved_official_weight_download_verified','url':URL,'bytes':count,'sha256':digest,
             'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'subsequent_load':'torch.load(...,map_location="cpu",weights_only=True); no automatic fallback download'}
    with (root/'weight-download-receipt.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
