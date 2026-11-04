"""Recompute descriptive data-pipeline accounting from saved JSON, no models."""
import collections
import csv
import hashlib
import json
import os
from pathlib import Path

P = Path(__file__).resolve().parent
proto = json.loads((P/'PROTOCOL.json').read_text())
ids = {x['image_id'] for x in proto['support_images']}
ann_ids = {930:2197,1892:8431,1900:8439,2244:8913,2363:9032}
checks = 0
summary = {}
csv_rows = []
for arm in ('A','M'):
    paths = sorted((P/('tracked_'+arm)).glob('*.json'))
    assert len(paths) == 200
    selected, visible, retained = collections.Counter(), collections.Counter(), collections.Counter()
    unique_hist, selected_unique_hist, gt_hist, visible_hist = (collections.Counter() for _ in range(4))
    applied, discarded, clip_events, clip_rows = (collections.Counter() for _ in range(4))
    selected_total = visible_total = gt_total = empty = repeated = repeated_selected = repeated_origin = 0
    introduced = 200
    widths, heights, pixels = [], [], []
    for i,path in enumerate(paths,1):
        x=json.loads(path.read_text());ref=json.loads((P/('reference_'+arm)/path.name).read_text())
        assert x['iteration']==i and x['RNG_before']==ref['RNG_before'] and x['signature']==ref['signature']
        checks += 1
        all_sources=x['final_source_regions'];vs=[s for s in all_sources if s['region_xyxy'][2]>s['region_xyxy'][0] and s['region_xyxy'][3]>s['region_xyxy'][1]]
        assert len(vs)==x['final_nonempty_source_occurrences']
        assert {s['image_id'] for s in all_sources}<=ids and all(s['origin_iteration']<=i and s['annotation_id']==ann_ids[s['image_id']] for s in all_sources)
        assert len(all_sources)<=8
        for s in all_sources:selected[s['image_id']]+=1
        for s in vs:visible[s['image_id']]+=1
        for s in x['final_GT_lineage']:retained[s['image_id']]+=1
        n=len(x['final_GT_lineage']);assert n==len(x['signature']['gt']['gt_instances']['boxes'])
        selected_total+=len(all_sources);visible_total+=len(vs);gt_total+=n;empty+=n==0
        repeated+=len(vs)>len({s['image_id'] for s in vs});repeated_selected+=len(all_sources)>len({s['image_id'] for s in all_sources})
        repeated_origin+=len(vs)>len({(s['image_id'],s['origin_iteration']) for s in vs})
        unique_hist[len({s['image_id'] for s in vs})]+=1;selected_unique_hist[len({s['image_id'] for s in all_sources})]+=1;gt_hist[n]+=1;visible_hist[len(vs)]+=1
        for event in x['events']:
            name=event['type'];applied[name]+=bool(event.get('applied'))
            discarded[name]+=event['discarded_GT_count']
            introduced+=len(event['candidate_GT_lineage'])-event['input_GT_count']
            clip_events[name]+=sum(bool(c['changed_rows']) for c in event['clip_events'])
            clip_rows[name]+=sum(len(c['changed_rows']) for c in event['clip_events'])
            if name in ('CachedMosaic','CachedMixUp'):
                assert event['cache_anchor_iterations'][-1]==i
                assert event['cache_size']<= (40 if name=='CachedMosaic' else 10)
                assert event['cache_size']==len(event['cache_anchor_iterations'])
                if i <= (4 if name=='CachedMosaic' else 1): assert not event['applied']
                for j in event.get('selected_cache_indices',[]): assert j<event['cache_size']-1
                if 'selected_cache_index' in event:assert event['selected_cache_index']<event['cache_size']-1
        _,h,w=x['signature']['shape_CHW'];heights.append(h);widths.append(w);pixels.append(h*w)
        csv_rows.append({'arm':arm,'iteration':i,'anchor_image_id':x['anchor_image_id'],'selected_sources':len(all_sources),'final_nonempty_regions':len(vs),'unique_final_image_ids':len({s['image_id'] for s in vs}),'GT':n,'discarded_GT':x['removed_GT_total'],'height':h,'width':w,'reference_seconds':ref['seconds']})
    assert introduced-gt_total==sum(discarded.values())
    summary[arm]={'iterations':200,'anchor_counts':{k:40 for k in sorted(ids)},'selected_source_occurrences':selected_total,'final_nonempty_geometric_source_occurrences':visible_total,'selected_unique_ID_histogram':dict(sorted(selected_unique_hist.items())),'final_unique_ID_histogram':dict(sorted(unique_hist.items())),'final_source_count_histogram':dict(sorted(visible_hist.items())),'iterations_with_repeated_selected_image_ID':repeated_selected,'iterations_with_repeated_final_image_ID':repeated,'iterations_with_repeated_same_cached_origin':repeated_origin,'per_source':{k:{'selected':selected[k],'final_region':visible[k],'final_GT':retained[k]} for k in sorted(ids)},'GT_presentations_introduced_including_composition':introduced,'GT_final_presentations':gt_total,'GT_final_histogram':dict(sorted(gt_hist.items())),'empty_GT_iterations':empty,'discarded_GT_by_stage':{k:v for k,v in discarded.items() if v},'clip_operations_with_changed_boxes':{k:v for k,v in clip_events.items() if v},'clip_changed_row_events_not_unique_boxes':{k:v for k,v in clip_rows.items() if v},'applied_counts':{k:v for k,v in applied.items() if v},'output_width_range':[min(widths),max(widths)],'output_height_range':[min(heights),max(heights)],'mean_output_pixels':sum(pixels)/200,'min_max_output_pixels':[min(pixels),max(pixels)],'reference_timing':json.loads((P/('reference_'+arm+'_summary.json')).read_text()),'tracked_timing':json.loads((P/('tracked_'+arm+'_summary.json')).read_text())}

result={'status':'COMPLETE','scientific_data_records':400,'independent_reference_comparisons':checks,'reference_and_completed_tracked_pipeline_calls':800,'additional_engineering_attempt_calls':16,'total_pipeline_calls_including_failed_and_debug':816,'model_calls':0,'optimizer_updates':0,'new_CU':0,'arms':summary,'interpretation':'Counts are data-use facts, not detector efficacy. Selected sources and surviving geometric regions are distinct; regions do not claim exact pixel attribution after interpolation/quantization.'}
(P/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
with (P/'ITERATIONS.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)

# Conditional scenario only: historical conservative formula adapted to ONE
# candidate. No allocation, current fee-rate observation or training is done.
W=692175962;B=4453855.0958829;I=2*20.244484609999972;pred=.4143788810000615;prep=554.525631070137
constant=prep+I+1.5*64*pred+2*W/B+120+180
remaining=7200-5443.899788473129
cost={'scope':'one new candidate U200 plus one candidate preflight; prior A background only; not an execution GO','formula_seconds':'P + I + 1.5*200*t_M + 1.5*64*t_pred + 2*W/B + 120 + 180','P_prepare_and_preflight_allowance_seconds':prep,'I_two_model_loads_seconds':I,'t_pred_historical_p90_seconds':pred,'W_complete_checkpoint_bytes':W,'B_historical_return_bytes_per_second':B,'overhead_seconds':120,'release_reserve_seconds':180,'constant_seconds':constant,'rate_CU_per_hour_historical_not_live':1.15,'remaining_prior_GPU_seconds':remaining,'remaining_prior_CU':2.5-1.8088330815795528,'candidate_GPU_update_p90_not_measured':True,'scenarios':[{'assumed_t_M_seconds':t,'total_seconds':constant+300*t,'CU_at_historical_rate':(constant+300*t)*1.15/3600,'GPU_seconds_margin':remaining-(constant+300*t)} for t in (1.6807091929999842,2.0)],'largest_t_M_fitting_old_GPU_seconds_if_all_other_assumptions_hold':(remaining-constant)/300,'Mac_augmentation_seconds_not_substituted_for_GPU_update_seconds':True,'notes':['P is the unchanged earlier conservative preparation/preflight allowance; it is not newly measured or reduced for one arm.','Transfer factor2 and fixed overhead/release allowances preserved from prior conservative accounting.','Includes complete checkpoint return; no lossless-delta saving assumed.','New runtime bandwidth, rate, GPU update time and available storage remain unverified.']}
(P/'ONE_CANDIDATE_COST.json').write_text(json.dumps(cost,indent=2)+'\n')
print(json.dumps({'A':{k:v for k,v in summary['A'].items() if k not in ('reference_timing','tracked_timing')},'M':{k:v for k,v in summary['M'].items() if k not in ('reference_timing','tracked_timing')},'cost':cost},indent=2))
