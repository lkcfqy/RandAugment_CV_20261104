"""Independent saved-pixel, provenance, split and result readback checks."""
import collections
import hashlib
import json
import math
import pathlib
import struct
import zlib

ROOT=pathlib.Path(__file__).resolve().parent


def png(path):
    data=path.read_bytes();assert data[:8]==b'\x89PNG\r\n\x1a\n'
    pos=8;blocks=[]
    while pos<len(data):
        n=struct.unpack('>I',data[pos:pos+4])[0];kind=data[pos+4:pos+8];body=data[pos+8:pos+8+n]
        assert zlib.crc32(kind+body)&0xffffffff==struct.unpack('>I',data[pos+8+n:pos+12+n])[0]
        if kind==b'IHDR':
            w,h,depth,color,compression,filtering,interlace=struct.unpack('>IIBBBBB',body)
            assert depth==8 and color in [0,6] and compression==filtering==interlace==0
            channels={0:1,6:4}[color]
        if kind==b'IDAT':blocks.append(body)
        pos+=12+n
        if kind==b'IEND':break
    assert pos==len(data)
    raw=zlib.decompress(b''.join(blocks));stride=w*channels;assert len(raw)==h*(stride+1)
    prior=bytearray(stride);pixels=bytearray();offset=0
    for _ in range(h):
        f=raw[offset];row=bytearray(raw[offset+1:offset+1+stride]);offset+=stride+1
        assert f in range(5)
        for i in range(stride):
            a=row[i-channels] if i>=channels else 0;b=prior[i];c=prior[i-channels] if i>=channels else 0
            if f==0:p=0
            elif f==1:p=a
            elif f==2:p=b
            elif f==3:p=(a+b)//2
            else:
                estimate=a+b-c;dist=[abs(estimate-a),abs(estimate-b),abs(estimate-c)]
                p=[a,b,c][dist.index(min(dist))]
            row[i]=(row[i]+p)&255
        pixels.extend(row);prior=row
    return w,h,channels,pixels


def features(pixels):
    assert len(pixels)==128*128*4
    lum=[(.2126*pixels[i]+.7152*pixels[i+1]+.0722*pixels[i+2])/255 for i in range(0,len(pixels),4)]
    out=[0.]*64
    for y in range(128):
        for x in range(128):
            i=y*128+x;p=i*4;cell=(y//32*4+x//32)*4
            for c in range(3):out[cell+c]+=pixels[p+c]/(255*1024)
            dx=abs(lum[i]-lum[i+1]) if x<127 else 0
            dy=abs(lum[i]-lum[i+128]) if y<127 else 0
            out[cell+3]+=(dx+dy)/(2*1024)
    return out


def score(vector,model):
    z=[max(-5,min(5,(v-m)/s)) for v,m,s in zip(vector,model['mean'],model['scale'])]
    coefficients=model['difference'];value=sum(x*c for x,c in zip(z,coefficients[:64]));offset=64
    for i in range(64):
        for j in range(i,64):value+=z[i]*z[j]*coefficients[offset];offset+=1
    assert offset==2144
    return model['intercept']+2*value/2144


def main():
    plan=json.loads((ROOT/'pixel-photo-plan.json').read_text())['rows'];ids={r['source_id']:r for r in plan}
    assert len(ids)==64
    receipts=[json.loads((ROOT/(kind+'-fetch-receipt.json')).read_text()) for kind in ['mask','photo']]
    for receipt in receipts:
        assert receipt['succeeded']==receipt['requested']
        for row in receipt['files']:
            path=ROOT/('masks' if row['kind']=='mask' else 'photos')/(row['source_id']+('.png' if row['kind']=='mask' else '.jpg'))
            b=path.read_bytes();assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']
    for row in plan:
        w,h,c,p=png(ROOT/'masks'/(row['source_id']+'.png'));assert c==1 and (w,h)==(row['width'],row['height'])
        counts=collections.Counter(p);assert set(counts)<=set([1,2,3])
        assert counts[1]==row['mask_label_counts']['foreground'] and counts[2]==row['mask_label_counts']['background']
        if row['role']=='background':
            x,y,rw,rh=row['largest_background_rectangle']
            assert all(p[j*w+i]==2 for j in range(y,y+rh) for i in range(x,x+rw))
    rows=[json.loads(line) for line in (ROOT/'rgb-features.jsonl').open()]
    cases=json.loads((ROOT/'render-cases.json').read_text())['rows'];lookup={r['case']:r for r in cases}
    by_key={(r['case'],r['layout'],r['policy'],r['actual_background_id']):r for r in rows}
    assert len(rows)==len(by_key)==9344
    for case in cases:
        for layout in ['protected','ordinary']:
            values=[r for r in rows if r['case']==case['case'] and r['layout']==layout]
            independent=[r for r in values if r['policy']=='independent'];coupled=[r for r in values if r['policy']=='coupled']
            assert len(coupled)==1 and len(independent)==(16 if case['split']=='fit' else 4)
            assert collections.Counter(r['C'] for r in independent)=={0:len(independent)//2,1:len(independent)//2}
            same=next(r for r in independent if r['actual_background_id']==case['base_background_id'])
            assert same['features']==coupled[0]['features']
            for r in values:
                assert len(r['features'])==64 and all(math.isfinite(v) and 0<=v<=1.00000001 for v in r['features'])
                assert r['Y']==ids[r['foreground_id']]['species'] and r['C']==ids[r['actual_background_id']]['species']
    block_ids={b:{r['source_id'] for r in plan if r['split']=='heldout' and r['macroblock']==b} for b in range(4)}
    assert all(len(s)==8 for s in block_ids.values())
    assert all(not block_ids[a]&block_ids[b] for a in range(4) for b in range(a+1,4))
    assert not {r['source_id'] for r in plan if r['split']=='fit'}&set.union(*block_ids.values())
    receipt=json.loads((ROOT/'render-receipt.json').read_text())
    max_feature_error=0.;qa_pairs=[]
    for q in receipt['QA']:
        case=lookup[q['case']]
        if q['variant']=='coupled':actual=case['base_background_id']
        else:
            actual=next(r['source_id'] for r in plan if r['split']=='heldout' and r['role']=='background' and
                        r['macroblock']==case['block'] and r['species']!=case['base_C'])
        _,_,c,p=png(ROOT/q['file']);assert c==4
        wanted=by_key[(q['case'],q['layout'],'independent',actual)]['features']
        error=max(abs(a-b) for a,b in zip(features(p),wanted));assert error<1e-12
        max_feature_error=max(max_feature_error,error)
    for case_id in sorted({q['case'] for q in receipt['QA']}):
        _,_,_,g=png(ROOT/'qa'/('case%d-guard.png'%case_id))
        _,_,_,a=png(ROOT/'qa'/('case%d-protected-coupled.png'%case_id))
        _,_,_,b=png(ROOT/'qa'/('case%d-protected-changed.png'%case_id))
        inside=outside_changes=0
        for i in range(16384):
            if g[i*4]>0:
                assert a[i*4:i*4+3]==b[i*4:i*4+3];inside+=1
            elif a[i*4:i*4+3]!=b[i*4:i*4+3]:outside_changes+=1
        assert outside_changes>0
        qa_pairs.append(dict(case=case_id,identical_foreground_and_guard_pixels=inside,changed_outside_guard_pixels=outside_changes))
    models=json.loads((ROOT/'readout-models.json').read_text());result=json.loads((ROOT/'pilot-results.json').read_text())
    max_loss_error=0.
    for layout in ['protected','ordinary']:
        for trained in ['coupled','independent']:
            for policy in ['coupled','independent']:
                group=[r for r in rows if r['split']=='heldout' and r['layout']==layout and r['policy']==policy]
                losses=[];correct=0
                for r in group:
                    s=score(r['features'],models[layout][trained])
                    losses.append(max(0,s)+math.log1p(math.exp(-abs(s)))-r['Y']*s)
                    correct+=((s>0)==bool(r['Y']))
                got=sum(losses)/len(losses);want=result['layouts'][layout]['overall']['table'][trained][policy]
                assert abs(got-want['mean_log_loss_nats'])<1e-12 and correct/len(group)==want['accuracy']
                max_loss_error=max(max_loss_error,abs(got-want['mean_log_loss_nats']))
    verification=dict(status='passed_saved_pixels_sources_weights_and_result_replay',source_photographs=64,
                      lossless_selected_masks_independently_decoded=64,background_rectangles_all_raw_label2=True,
                      all_192_downloaded_payload_file_byte_and_SHA256_readbacks_passed=True,
                      independent_test_macroblocks=4,photo_sources_per_test_block=8,
                      no_source_photo_overlap_between_roles_or_splits=True,
                      feature_rows=9344,independent_policy_C_balanced_for_every_F_T_case=True,
                      coupled_and_independent_F_T_marginal_preserved=True,QA_saved_RGB_images_recomputed=32,
                      max_RGB_feature_replay_error=max_feature_error,protected_QA_pairs=qa_pairs,
                      all_protected_foreground_local_edge_pixels_identical=True,
                      max_saved_model_loss_replay_error=max_loss_error,
                      manual_contact_sheet_review='Viewed 8 cat/dog foreground cases across 4 heldout blocks; mask/photo alignment upright, changing backgrounds visible. Not an audit of every segmentation or animal identity.',
                      cautious_primary_screen_passed=False,
                      interpretation='No positive primary RGB shortcut signal in this small weak-readout transfer; not proof ForNet or stronger models lack the mechanism.')
    with (ROOT/'pilot-verification.json').open('x') as f:json.dump(verification,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in verification.items() if k!='protected_QA_pairs'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
