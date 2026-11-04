import base64
import hashlib
import json
import math
from pathlib import PurePosixPath

PREFIX='CUEVIDENCE1:'
CONTROL_PREFIX='CUCONTROL1:'

def parse_control_frame(output):
    """Require one explicit control frame; CLI diagnostics remain in the raw log."""
    frames=[line[len(CONTROL_PREFIX):] for line in output.splitlines() if line.startswith(CONTROL_PREFIX)]
    if len(frames)!=1:raise ValueError('exactly one Colab control JSON frame required')
    value=json.loads(frames[0])
    if not isinstance(value,dict):raise ValueError('Colab control frame must be a JSON object')
    return value

def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

def compute_gate(cumulative_seconds,extra_startup_seconds,three_mode_seconds,io_reserve_seconds):
    values=[cumulative_seconds,extra_startup_seconds,io_reserve_seconds,*three_mode_seconds]
    if len(three_mode_seconds)!=3 or any(not math.isfinite(x) or x<0 for x in values) or io_reserve_seconds<=0 or min(three_mode_seconds)<=0:
        raise ValueError('finite complete three-mode timings and positive I/O reserve required')
    forecast=cumulative_seconds+extra_startup_seconds+2*492*max(three_mode_seconds)+io_reserve_seconds
    return {'allowed':forecast<=1800,'forecast_seconds':forecast,'full_queries':492,'multiplier':2,'io_reserve_seconds':io_reserve_seconds,'max_mode_seconds':max(three_mode_seconds),'cumulative_seconds':cumulative_seconds,'extra_startup_seconds':extra_startup_seconds}

def credit_gate(rate_per_hour,spent,remaining_allocation_seconds,cap):
    if any(not math.isfinite(x) or x<0 for x in [rate_per_hour,spent,remaining_allocation_seconds,cap]) or cap<=0:raise ValueError('invalid credit budget')
    projected=spent+rate_per_hour*remaining_allocation_seconds/3600+.05
    return {'allowed':projected<cap,'projected_with_rounding_buffer':projected,'rounding_buffer_CU':.05,'cap_CU':cap}

def validate_artifact_path(path):
    p=PurePosixPath(path)
    if not isinstance(path,str) or p.is_absolute() or not p.parts or any(x in ('.','..','') for x in p.parts) or '\\' in path:
        raise ValueError('unsafe evidence path')
    return str(p)

def make_packet(path,data):
    return PREFIX+json.dumps({'path':validate_artifact_path(path),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'base64':base64.b64encode(data).decode()},separators=(',',':'))

def decode_packet(line):
    if not line.startswith(PREFIX):raise ValueError('not an evidence packet')
    row=json.loads(line[len(PREFIX):]);path=validate_artifact_path(row['path']);data=base64.b64decode(row['base64'],validate=True)
    if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('evidence content/hash mismatch')
    return path,data

def validate_go(authorization,freeze_sha256):
    if not isinstance(authorization,dict) or authorization.get('allocation_allowed') is not True or authorization.get('freeze_sha256')!=freeze_sha256 or authorization.get('scope')!='one_T4_precheck_then_conditional_492' or authorization.get('max_allocations')!=1:
        raise ValueError('Explicit GO matching this exact freeze is required before allocation')

def complete_result_set(rows,expected):
    ids=[x['query_id'] for x in rows]
    if len(ids)!=len(set(ids)) or set(ids)!=set(expected):raise ValueError('missing, extra, or duplicate query result')
    if any(x['query_sha256']!=expected[x['query_id']] for x in rows):raise ValueError('result query identity mismatch')
    return True

def validate_query(q,source,label):
    payload={k:v for k,v in q.items() if k!='query_sha256'}
    if hashlib.sha256(canonical(payload)).hexdigest()!=q['query_sha256']:raise ValueError('query hash mismatch')
    target=q['target'];mode=q['mode']
    if target not in ('A','B') or mode not in ('F','C','V') or not label['eligible']:raise ValueError('invalid target/mode/eligibility')
    truth_rows=[x for x in label['selected_images'] if x['filename']==source['filename']]
    if label['pair_id']!=source['pair_id'] or len(truth_rows)!=1 or any(truth_rows[0][target+'_count']!=source[target+'_count'] for target in ('A','B')):
        raise ValueError('truth must agree with frozen labels and selected image identity')
    other='B' if target=='A' else 'A';texts=label['frozen_input_texts']
    positive=texts['F_'+target] if mode=='F' else texts['C'] if mode=='C' else ''
    negative=texts['F_'+other] if mode=='F' else ''
    expected={'query_id':source['filename']+'/'+mode+'/'+target,'filename':source['filename'],'pair_id':source['pair_id'],
              'selection_rank_sha256':source['selection_sha256'],'text':positive,'negative_texts':[negative],
              'exemplar_boxes':source[target+'_boxes_xyxy'],'negative_exemplar_boxes':[source[other+'_boxes_xyxy']],
              'target_count_from_official_annotation':source[target+'_count'],'negative_count_from_official_annotation':source[other+'_count'],
              'caption':positive+' . '+negative+' . '}
    if any(q.get(k)!=v for k,v in expected.items()):raise ValueError('frozen image/name/box/target/mode identity mismatch')
    if len(q['exemplar_boxes'])!=3 or len(q['negative_exemplar_boxes'])!=1 or len(q['negative_exemplar_boxes'][0])!=3:raise ValueError('three positive and three negative exemplars required')
    if source['A_count']+source['B_count']<=0:raise ValueError('positive official total required')
    return True

def bound_metric_inputs(predictions,labels):
    """No caller-supplied truth: labels are the hashed, frozen authoritative roster."""
    expected={im['filename'] for pair in labels if pair['eligible'] for im in pair['selected_images']}
    if set(predictions)!=expected:raise ValueError('prediction images do not match frozen eligible labels')
    policy=[];paired=[]
    for label in labels:
        policy.append({'pair_id':label['pair_id'],'eligible':label['eligible'],'reason':label['reason'],'image_ids':[x['filename'] for x in label['selected_images']]})
        if label['eligible']:
            paired.append({'pair_id':label['pair_id'],'images':[{'image_id':im['filename'],'truth':{'A':im['A_count'],'B':im['B_count']},'predictions':predictions[im['filename']]} for im in label['selected_images']]})
    return paired,policy
