"""Compile measured resource evidence. No model import or real utility."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,math,os,shutil,time

ROOT=Path(__file__).resolve().parent
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(name,x):(ROOT/name).write_text(json.dumps(x,sort_keys=True,indent=2)+'\n')

identity=json.loads((ROOT/'IDENTITIES.json').read_text())
audit=json.loads((ROOT/'ASSET_AUDIT.json').read_text())
bench={m:json.loads((ROOT/(m+'_BENCHMARK.json')).read_text()) for m in ['dinov2_small','deit_tiny']}
assert all(x['status']=='PASS_NO_QUERY_UTILITY' and x['query_image_forwards']==0 and x['real_head_fits']==0 for x in bench.values())
a,b=[bench[k]['views'] for k in ['dinov2_small','deit_tiny']]
assert [x['id'] for x in a]==[x['id'] for x in b]
assert all(x['weak_ra']==y['weak_ra'] for x,y in zip(a,b))

# Re-save only these task-created synthetic fixtures atomically with fsync,
# checking exact bytes. No source asset or actual experimental result is modified.
io=[]
for model,x in bench.items():
    path=ROOT/(model+'_synthetic_persistence.npz');payload=path.read_bytes()
    assert hashlib.sha256(payload).hexdigest()==x['synthetic_persistence_sha256']
    tmp=path.with_suffix('.atomic.tmp');t=time.perf_counter()
    with tmp.open('wb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    assert sha(path)==x['synthetic_persistence_sha256']
    seconds=time.perf_counter()-t
    io.append({'encoder':model,'bytes':len(payload),'atomic_write_fsync_replace_and_hash_seconds':seconds,
               'bytes_per_second':len(payload)/seconds,'hash_unchanged':True})
dump('ATOMIC_IO_CHECK.json',{'status':'PASS','checks':io,'old_assets_modified':False})

S,Q=len(identity['support_union']),len(identity['query']);C=100;D=384+192
per_encoder=9*S+Q
formal={
    'support_and_query_features_fp32':per_encoder*D*4,
    'all_720_heads_fp64':360*(D+2)*C*8,
    'all_query_argmax_uint8':2*360*Q,
    'all_cv_records_compact':2*12*(800+1600)*3*8,
    'query_labels_uint8_and_ids_uint32':Q*5,
    'headers_seals_logs_plots_metric_tables_allowance':12*2**20}
preflight_bytes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
reserve=40*2**20
storage_total=sum(formal.values())+preflight_bytes+reserve
io_speed=min(min(x['bytes_per_second'] for x in io),100*2**20)
io_seconds=3*sum(formal.values())/io_speed # write, read/verify, later reprojection read
times={}
for model,x in bench.items():
    times[model]={
        'encoder_batches':math.ceil(per_encoder/16),
        'forward_seconds':math.ceil(per_encoder/16)*x['timed_batch_p90_seconds'],
        # Deliberately charge query clean preprocessing as if all9 views were made.
        'transform_seconds_upper':(S+Q)*x['transform_seconds_8_originals']/8,
        # Both encoders must seal fits before queries, so reload for query phase.
        'two_loads_seconds':2*x['load_seconds'],
        'all360_head_fits_max_shape_seconds':360*x['synthetic_ridge_timed_p90_seconds'],
        # Both original scores and persisted-score recomputation counted; OOF reserve included.
        'query_score_and_reprojection_seconds':2*360*x['synthetic_query_score_one_head_seconds'],
        'cv_scoring_seconds_upper':2*360*x['synthetic_query_score_one_head_seconds']}
numerical=sum(sum(v for k,v in x.items() if k!='encoder_batches') for x in times.values())+io_seconds
projection=1.5*numerical+600
now=datetime.now(timezone.utc);start=datetime.fromisoformat('2026-11-02T05:36:42+00:00')
elapsed=(now-start).total_seconds()
free=shutil.disk_usage(ROOT).free
resource={'status':'RESOURCE_GATE_PASS' if projection+elapsed<=5400 and storage_total<=500*2**20 and free>storage_total+2*2**30 else 'RESOURCE_GATE_FAIL',
 'utc':now.isoformat(),'formal_run_not_started':True,'query_utility_observed':False,
 'support_union':S,'query_originals':Q,'formal_forward_images_per_encoder':per_encoder,
 'formal_forward_images_both_encoders':2*per_encoder,'head_fits':720,'cv_choices':48,
 'full_query_predictions_both_encoders':2*360*Q,'cv_original_predictions_both_encoders':2*12*(800+1600)*3,
 'time_components_seconds':times,'conservative_io_bytes_per_second':io_speed,'io_seconds':io_seconds,
 'numeric_and_io_seconds_before_margin':numerical,'margin_multiplier':1.5,'fixed_setup_verify_archive_reserve_seconds':600,
 'future_full_run_projection_seconds':projection,'preflight_elapsed_seconds_at_projection':elapsed,
 'combined_elapsed_plus_projection_seconds':elapsed+projection,'combined_budget_seconds':5400,
 'projection_is_estimate_not_guarantee':True,'five_batch_percentile_is_not_a_population_bound':True,
 'formal_uncompressed_bytes_inventory':formal,'current_preflight_artifact_bytes':preflight_bytes,
 'extra_storage_reserve_bytes':reserve,'combined_storage_budget_bytes':storage_total,'artifact_hard_limit_bytes':500*2**20,
 'local_free_bytes':free,'peak_measured_preflight_rss_bytes':max(x['peak_rss_bytes'] for x in bench.values()),
 'full_run_peak_rss_not_measured':True,'preflight_weak_ra_bitwise_equal_views':64,
 'forbidden_full_query_score_cube_bytes':2*360*Q*C*8,
 'full_class_scores_persistence':'Not stored redundantly. All class scores are reproducible from exact FP32 features + FP64 W,b using frozen 256-query blocks; all argmax predictions retained.',
 'new_CU':0,'user_global_cutoff_utc':'2026-11-02T10:00:00Z'}
dump('RESOURCE_PROJECTION.json',resource)
print(json.dumps(resource,indent=2))
