"""Reconstruct every delivered image hash and independently recompute predictions."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,math
import numpy as np
from PIL import Image,ImageOps

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'encoder-resnet18-round2-20261021'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    protocol=json.loads((ROOT/'protocol.json').read_text())
    freeze=json.loads((ROOT/'development-freeze.json').read_text())
    assert freeze['protocol_sha256']==sha(ROOT/'protocol.json')
    assert freeze['code_sha256']==sha(ROOT/'run_equalize_audit.py')
    assert freeze['development_results_sha256']==sha(ROOT/'fit-results.json')
    assert sha(Path(protocol['head_file']))==protocol['head_file_sha256']
    head=json.loads(Path(protocol['head_file']).read_text())['protected']['coupled']
    old=json.loads((OLD/'main-frames.json').read_text())['records']
    raw=np.memmap(OLD/'main-frames.rgba',dtype=np.uint8,mode='r',shape=(len(old),128,128,4))
    image_hash_checks=0;LL_error=0.;source_sets={};part_marginal_checks=0
    for split in ['fit','heldout']:
        result=json.loads((ROOT/(split+'-results.json')).read_text())
        desc=json.loads((ROOT/(split+'-image-manifest.json')).read_text())
        rows=desc['records'];x=np.load(ROOT/(split+'-features.npy'),allow_pickle=False)
        assert x.shape==(len(rows),512) and sha(ROOT/(split+'-features.npy'))==result['feature_sha256']
        selected=[r for r in protocol['selection_rows'] if r['split']==split]
        source_sets[split]={r['foreground_id'] for r in selected}
        reconstructed={}
        for s in selected:
            source_rows=[r for r in old if r['case']==s['case'] and r['layout']=='protected']
            guard=np.asarray(Image.open(ROOT/('guards-'+split)/('case'+str(s['case'])+'.png')).convert('L'))>0
            base={r['actual_background_id']:np.array(raw[r['index'],:,:,:3],copy=True) for r in source_rows}
            equal={key:np.array(ImageOps.equalize(Image.fromarray(im))) for key,im in base.items()}
            solar={key:np.array(ImageOps.solarize(Image.fromarray(im),128)) for key,im in base.items()}
            for r in rows:
                if r['case']!=s['case']:continue
                j,k=r['background_id'],r['tone_background_id'];variant=r['variant']
                if variant=='no_op':im=base[j]
                elif variant=='solarize':im=solar[j]
                elif variant=='equalize':im=equal[j]
                else:
                    assert variant=='cross_equalize'
                    im=equal[j].copy();im[guard]=equal[k][guard]
                assert hashlib.sha256(im.tobytes()).hexdigest()==r['RGB_sha256']
                reconstructed[r['index']]=im;image_hash_checks+=1
            if split=='heldout':
                rs=[r for r in rows if r['case']==s['case'] and r['variant']=='cross_equalize']
                G=[r for r in rows if r['case']==s['case'] and r['variant']=='equalize'];m=len(G)
                for region in [guard,~guard]:
                    a=Counter(hashlib.sha256(reconstructed[r['index']][region].tobytes()).hexdigest() for r in G)
                    b=Counter(hashlib.sha256(reconstructed[r['index']][region].tobytes()).hexdigest() for r in rs)
                    assert b==Counter({key:count*m for key,count in a.items()});part_marginal_checks+=1
        for r,vec in zip(rows,x):
            norm=math.sqrt(math.fsum(float(z)**2 for z in vec))
            score=10*math.fsum((float(z)/norm)*(v1-v0) for z,v0,v1 in zip(vec,head['centers'][0],head['centers'][1]))
            loss=max(0,score)+math.log1p(math.exp(-abs(score)))-r['Y']*score
            assert ((score>0)==bool(r['Y']))==r['correct']
            LL_error=max(LL_error,abs(loss-r['loss_nats']))
        assert result['threads']==1 and result['CPU_only'] and not result['GPU'] and not result['ARIS']
    assert len(source_sets['fit'])==len(source_sets['heldout'])==16 and not source_sets['fit']&source_sets['heldout']
    assert LL_error<1e-12
    # Prior delivered results are immutable.
    previous=json.loads((OLD/'delivery-verification.json').read_text())
    for r in previous['files']:
        f=OLD/r['relative_path'];assert f.stat().st_size==r['bytes'] and sha(f)==r['sha256']
    verify=dict(status='passed',reconstructed_RGB_image_hashes=image_hash_checks,
        heldout_exact_processed_part_marginal_checks=part_marginal_checks,
        independent_math_fsum_loss_max_abs_error=LL_error,all_predictions_read_back=True,
        fixed_protocol_head_and_code_unchanged=True,fit_and_heldout_sources_disjoint=True,
        prior_round2_delivery_files_unchanged=len(previous['files']),CPU_only=True,ARIS=False,GPU=False)
    with (ROOT/'verification.json').open('x') as f:json.dump(verify,f,indent=2);f.write('\n')
    print(json.dumps(verify,indent=2))

if __name__=='__main__':main()
