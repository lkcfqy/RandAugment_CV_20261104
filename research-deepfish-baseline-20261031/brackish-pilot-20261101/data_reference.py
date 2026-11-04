"""Data-only deterministic reference and passive instrumentation equivalence; no model."""
from runtime_common import *
import collections
def main():
    import torch,numpy as np
    setup();o=observer();manifest=read(P/'IMAGE_MANIFEST.json');proto=read(P/'PROTOCOL.json');selected={};summaries={}
    for arm,group in [('A0','support0'),('A1','support1')]:
        ds=dataset(group);assert len(ds)==5
        shot=read(P/f'data/annotations/{group}.json');aids={a['image_id']:a['id'] for a in shot['annotations']}
        assert len(aids)==5
        for im in shot['images']:assert sha(P/'data/images'/im['file_name'])==manifest[im['file_name']]['image_sha256']
        order=training_order(ds);out=P/f'data_reference_{arm}';out.mkdir()
        reset_seed();obs=o.Observer();pipeline=o.make_pipeline('A',obs);references=[]
        for u,index in enumerate(order,1):
            info=ds.get_data_info(index);obs.begin(u,info['img_id'],aids[info['img_id']],(info['height'],info['width']))
            before=o.rng();t=time.monotonic();item=pipeline(copy.deepcopy(info));duration=time.monotonic()-t
            sig=o.signature(item);row={'update':u,'dataset_index':index,'image_id':info['img_id'],'signature':sig,'RNG_before':before,'pipeline_seconds':duration,'events':copy.deepcopy(obs.events),'final_GT_lineage':copy.deepcopy(obs.boxes)}
            atomic(out/f'{u:03}.json',row);references.append(row)
        # Independent uninstrumented output must agree at all200 fixed identities.
        reset_seed();plain=o.make_pipeline('A')
        for u,index in enumerate(order,1):
            before=o.rng();item=plain(copy.deepcopy(ds.get_data_info(index)));sig=o.signature(item);ref=references[u-1]
            assert sig==ref['signature'] and before==ref['RNG_before'],'Passive observer changed image/GT/RNG'
        positive=[x for x in references if len(x['signature']['gt']['gt_instances']['boxes'])>0];assert positive
        largest=max(x['signature']['shape_CHW'][1]*x['signature']['shape_CHW'][2] for x in positive)
        chosen=next(x for x in positive if x['signature']['shape_CHW'][1]*x['signature']['shape_CHW'][2]==largest)
        selected[arm]=chosen['update']
        summaries[arm]={'dataset_index_order':order,'image_order':[x['image_id'] for x in references],'each_image_exposures':dict(collections.Counter(x['image_id'] for x in references)),'positive_updates':len(positive),'empty_GT_updates':200-len(positive),'largest_positive_shape':chosen['signature']['shape_CHW'],'preflight_iteration':chosen['update'],'largest_any_shape':max((x['signature']['shape_CHW'] for x in references),key=lambda s:s[1]*s[2]),'observer_untracked_equivalence_updates':200,'pipeline_seconds':sum(x['pipeline_seconds'] for x in references)}
    assert summaries['A0']['dataset_index_order']==summaries['A1']['dataset_index_order']
    proto['preflight_iterations']=selected;proto['status']='FROZEN_BEFORE_ANY_MODEL_OUTPUTS_OR_UPDATES'
    proto['data_reference_summary_sha256']=digest(summaries)
    atomic(P/'DATA_REFERENCE_SUMMARY.json',summaries);atomic(P/'PROTOCOL.json',proto)
    atomic(P/'DATA_REFERENCE_MANIFEST.json',{str(f.relative_to(P)):sha(f) for folder in ['data_reference_A0','data_reference_A1'] for f in sorted((P/folder).glob('*.json'))})
    print(json.dumps(summaries),flush=True)
if __name__=='__main__':main()
