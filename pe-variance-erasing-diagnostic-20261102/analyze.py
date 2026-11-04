"""Recompute every frozen diagnostic cell from persisted tensors. No models."""
from common import *
import csv

def describe(values):
    a=np.asarray(values,dtype='float64');assert np.isfinite(a).all()
    return {'n':len(a),'min':float(a.min()),'q25':float(np.quantile(a,.25)),'median':float(np.median(a)),'q75':float(np.quantile(a,.75)),'max':float(a.max()),'mean':float(a.mean())}
def cosine_distance(a,b):return float(1-np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
def top5(query,reference,exclude):
    scores=reference@query/(np.linalg.norm(reference,axis=1)*np.linalg.norm(query));scores[exclude]=-np.inf
    return np.argsort(-scores,kind='stable')[:5].tolist()
def jaccard(a,b):return 1-len(set(a)&set(b))/len(set(a)|set(b))
def output_slope(fs):return float(np.linalg.norm((fs[2].astype('float64')-fs[0])/.02)/np.linalg.norm(fs[1]))
def ordinary_mismatch(pred,actual):
    lp=math.log(pred[0]/pred[1]);la=math.log(actual[0]/actual[1])
    return abs(lp)>=math.log(1.01) and abs(la)>=math.log(1.01) and lp*la<0
def write_csv(name,rows):
    with (P/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def analyze():
    proto=read('PROTOCOL.json');receipt=read('RUN_RECEIPT.json');frozen=read('RUN_FROZEN.json')
    assert receipt['status']=='COMPLETE' and receipt['counts']=={'core':768,'halfstep':16,'identity':2}
    assert sha(P/'PROTOCOL.json')==frozen['protocol_sha256']==receipt['protocol_sha256']
    for n,h in frozen['code'].items():assert sha(P/n)==h,('code changed',n)
    expected={(n,r['id'],m) for n in proto['models'] for r in proto['images'] for m in proto['modes']}
    assert len(receipt['identities'])==256 and set(map(tuple,receipt['identities']))==expected
    summaries={};pairrows=[];conditionrows=[];max_check=0.;recomputed=0
    for name in proto['models']:
        compath=P/f'raw/{name}/MODEL_COMPONENTS.npz';mr=read(f'raw/{name}/MODEL_RECEIPT.json');assert sha(compath)==mr['component_npz_sha256']
        with np.load(compath,allow_pickle=False) as c:
            pos=c['position'][0,1:].astype('float64');gamma=c['norm_weight'].astype('float64');eps=float(c['norm_eps'])
        entries=[]
        for ix,row in enumerate(proto['images']):
            stem=f'raw/{name}/{ix:02d}_{row["id"]}';rec=read(stem+'.json');assert rec['image_id']==row['id'] and rec['model']==name
            assert sha(P/(stem+'.npz'))==rec['npz_sha256']
            with np.load(P/(stem+'.npz'),allow_pickle=False) as z:
                raw={k:z[k] for k in z.files}
            for mi,mode in enumerate(proto['modes']):
                x=raw['patches'][mi];s=rec['modes'][mode]['tokens'];assert hashlib.sha256(x.tobytes()).hexdigest()==rec['modes'][mode]['patch_tensor_sha256']
                xd=x.astype('float64');a=xd+pos;a-=a.mean(-1,keepdims=True);q=pos-pos.mean(-1,keepdims=True);scale=np.sqrt((a*a).mean(-1,keepdims=True)+eps)
                d=gamma*(q/scale-a*(a*q).mean(-1,keepdims=True)/scale**3)
                checks=[(float(xd.var()),s['patch_flat_var']),(float(np.sqrt(np.mean(d*d))),s['regions']['all']['LN_PE_sensitivity_RMS'])]
                for v,w in checks:
                    er=abs(v-w)/(abs(w)+1e-30);assert er<1e-11,(name,ix,mode,v,w);max_check=max(max_check,er)
                assert np.allclose(xd.var(-1),raw['token_feature_variance'][mi],rtol=1e-11,atol=1e-12)
                assert np.allclose(np.sqrt((d*d).mean(-1)),raw['token_LN_PE_sensitivity_RMS'][mi],rtol=1e-11,atol=1e-12)
                recomputed+=1
            # Independently recreate the input/multiset contract, no network or model.
            xs=variants(load_image(row),row)
            for mode in proto['modes']:assert th(xs[mode])==rec['modes'][mode]['input']['input_tensor_sha256']
            for fld in ('flat_mean','flat_var'):
                a=rec['modes']['pixel']['input'][fld];b=rec['modes']['matched_correlated']['input'][fld]
                assert abs(a-b)<1e-12
            entries.append((rec,raw))
        clean=np.stack([raw['features'][0,1] for rec,raw in entries]).astype('float64')
        ordinary_errors=[];mismatches=[];tokenlogs=[];lnlogs=[];outputlogs=[];cosines=[];neighbour_dist=[]
        bymode={mode:[] for mode in proto['modes']}
        for ix,((rec,raw),row) in enumerate(zip(entries,proto['images'])):
            stats=rec['modes'];vi0=stats['none']['input']['flat_var'];s0=stats['none']['tokens']['regions']['all']['LN_PE_sensitivity_RMS']
            pred=[];actual=[]
            for mi,mode in enumerate(proto['modes']):
                ins=stats[mode]['input'];tok=stats[mode]['tokens'];S=tok['regions']['all']['LN_PE_sensitivity_RMS'];V=tok['patch_flat_var']
                fc=raw['features'][mi,1].astype('float64');fn=raw['features'][0,1].astype('float64')
                pr=math.sqrt(vi0/ins['flat_var']);ar=S/s0;logerror=math.log(ar/pr)
                line={'model':name,'image_id':row['id'],'mode':mode,'input_flat_var':ins['flat_var'],'patch_flat_var':V,'LN_PE_sensitivity_RMS':S,'scalar_proxy_relative_S':pr,'actual_relative_S':ar,'proxy_log_error':logerror,
                  'CLS_cosine_distance_from_none':cosine_distance(fc,fn),'CLS_relative_L2_from_none':float(np.linalg.norm(fc-fn)/np.linalg.norm(fn)),
                  'output_PE_relative_slope':output_slope(raw['features'][mi]),'erase_lag1_mean':float(np.mean([v for v in ins['erase_lag1_correlations'] if v is not None])) if all(v is not None for v in ins['erase_lag1_correlations']) else None,
                  'input_channel_var_mean':float(np.mean(ins['channel_var'])),'erase_channel_var_mean':float(np.mean(ins['erase_channel_var'])),'erase_flat_var':ins['erase_flat_var']}
                for region in ('fully_erased','unaffected'):
                    line[region+'_mean_feature_var']=tok['regions'][region]['mean_feature_var'];line[region+'_LN_PE_sensitivity_RMS']=tok['regions'][region]['LN_PE_sensitivity_RMS']
                if name=='deit_tiny':
                    lg=raw['logits'][mi,1];l0=raw['logits'][0,1]
                    line['DeiT_top1_changed_vs_none']=int(lg.argmax()!=l0.argmax());line['DeiT_top5_Jaccard_distance_vs_none']=jaccard(np.argsort(-lg)[:5],np.argsort(-l0)[:5])
                conditionrows.append(line);bymode[mode].append(line)
                if mode in ('pixel','rand'):ordinary_errors.append(abs(logerror));pred.append(pr);actual.append(ar)
            mismatch=ordinary_mismatch(pred,actual);mismatches.append(mismatch)
            pix,corr=stats['pixel'],stats['matched_correlated'];vp=pix['tokens']['patch_flat_var'];vc=corr['tokens']['patch_flat_var'];sp=pix['tokens']['regions']['all']['LN_PE_sensitivity_RMS'];sc=corr['tokens']['regions']['all']['LN_PE_sensitivity_RMS']
            fp=raw['features'][1,1].astype('float64');fc=raw['features'][3,1].astype('float64')
            nrp=top5(fp,clean,ix);nrc=top5(fc,clean,ix);cd=cosine_distance(fp,fc);jd=jaccard(nrp,nrc)
            vp_ratio=vc/vp;s_ratio=sc/sp;o_ratio=output_slope(raw['features'][3])/output_slope(raw['features'][1])
            tokenlogs.append(abs(math.log(vp_ratio)));lnlogs.append(abs(math.log(s_ratio)));outputlogs.append(abs(math.log(o_ratio)));cosines.append(cd);neighbour_dist.append(jd)
            pair={'model':name,'image_id':row['id'],'ordinary_order_disagreement':int(mismatch),'matched_vs_pixel_input_var_ratio':corr['input']['flat_var']/pix['input']['flat_var'],'matched_vs_pixel_token_var_ratio':vp_ratio,'matched_vs_pixel_LN_S_ratio':s_ratio,'matched_vs_pixel_output_PE_slope_ratio':o_ratio,'matched_vs_pixel_CLS_cosine_distance':cd,'matched_vs_pixel_neighbour_Jaccard_distance':jd,'pixel_clean_neighbours':json.dumps(nrp),'matched_clean_neighbours':json.dumps(nrc)}
            if name=='deit_tiny':pair['matched_vs_pixel_top1_changed']=int(raw['logits'][1,1].argmax()!=raw['logits'][3,1].argmax())
            pairrows.append(pair)
        # The exact pre-output decision rule; none of these gates is a significance test.
        threshold=math.log(1.10)
        gates={'ordinary':float(np.median(ordinary_errors))>=threshold and np.mean(mismatches)>=.25,
          'matched':float(np.median(tokenlogs))>=threshold and float(np.median(lnlogs))>=threshold,
          'output':float(np.median(outputlogs))>=threshold and (np.median(cosines)>=.01 or np.median(neighbour_dist)>=.20)}
        gates={k:bool(v) for k,v in gates.items()}
        summaries[name]={'n_images':32,'gates':gates,'ordinary_proxy_abs_log_error':describe(ordinary_errors),'ordinary_order_disagreements':int(sum(mismatches)),'ordinary_order_disagreement_fraction':float(np.mean(mismatches)),
          'matched_abs_log_token_var_ratio':describe(tokenlogs),'matched_abs_log_LN_S_ratio':describe(lnlogs),'matched_abs_log_output_PE_slope_ratio':describe(outputlogs),'matched_CLS_cosine_distance':describe(cosines),'matched_neighbour_Jaccard_distance':describe(neighbour_dist),'by_mode':{}}
        for mode,rows in bymode.items():
            keys=[k for k in rows[0] if k not in ('model','image_id','mode') and rows[0][k] is not None]
            summaries[name]['by_mode'][mode]={k:describe([r[k] for r in rows]) for k in keys}
        if name=='deit_tiny':summaries[name]['matched_vs_pixel_top1_changes']=sum(r['matched_vs_pixel_top1_changed'] for r in pairrows if r['model']==name)
    # Uniform CSV columns, including unavailable classifier statistics for DINOv2.
    for rows,filename in ((conditionrows,'ALL_CONDITIONS.csv'),(pairrows,'ALL_PAIRED_DIFFERENCES.csv')):
        keys=list(dict.fromkeys(k for r in rows for k in r));write_csv(filename,[{k:r.get(k) for k in keys} for r in rows])
    eligible=all(all(s['gates'].values()) for s in summaries.values())
    dump('SUMMARY.json',{'UTC':utc(),'protocol_sha256':sha(P/'PROTOCOL.json'),'models':summaries,'decision':'FUTURE_TRAINING_PROPOSAL_ELIGIBLE_NOT_AUTHORIZED' if eligible else 'CLOSE_PROBE_NO_TRAINING','all_identities_complete':True,'independently_recomputed_conditions':recomputed,'max_numpy_vs_saved_stat_relative_error':max_check,'all_input_hashes_recreated':True,'all_channel_multisets_exact':True,'threshold_abs_log':math.log(1.10),'no_accuracy_or_novelty_claim':True})
    print(json.dumps({'status':'COMPLETE','conditions':recomputed,'decision': 'eligible' if eligible else 'close','model_gates':{n:s['gates'] for n,s in summaries.items()}}))

if __name__=='__main__':analyze()
