import hashlib
import json
import os
from pathlib import Path

ASSETS = Path(os.environ.get('STHETA_ASSETS', '/Users/lkc/Downloads/projects/RandAugment_CV_20261104/research-shared-theta-ad-20261030'))
WORK = ASSETS / 'implementation-v1'
THETAS = [(1.0, 1.0)] + [(a,b) for a in (.8,1.,1.2) for b in (.8,1.,1.2) if (a,b)!=(1.,1.)]
THETA_IDS = [f'a={a:g},b={b:g}' for a,b in THETAS]
SEEDS = [11,22,33]
SHUFFLES = [9001,9002,9003,9004,9005]
METHOD_NAMES=tuple(sorted(['identity','pool9','primary','robust_color','selector/all256','selector/min_top13']+
    [f'shuffle/{p}' for p in SHUFFLES]+
    [f'refselect/{selector}/K{k}/{bank}' for selector in ('spatial_bow_KL','trimmed_chamfer') for k in (1,4,10) for bank in ('selected_view','pool9')]+
    [f'projection/{matrix}/r{rank}/{bank}/{readout}' for matrix in ('S0','Sc') for rank in (8,16,32) for bank in ('identity','pool9') for readout in ('pre_sq','post_sq','post_sqrt')]))
BASELINE_NAMES=tuple(n for n in METHOD_NAMES if n in ('identity','pool9','robust_color') or n.startswith(('refselect/','projection/')))
assert len(METHOD_NAMES)==59 and len(BASELINE_NAMES)==51
REVISION = '150c8e7bb7cef2d30ec31b13a517af14840ee3f7'
WEIGHT_SHA = 'ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1'
SPEC = {
 'version':'stheta-v1', 'status':'implementation/preflight only; formal scoring awaits parent confirmation',
 'theta_order':THETA_IDS, 'theta_parameters':THETAS, 'reference_seeds':SEEDS,
 'shuffle_seeds':SHUFFLES,'primary_shuffle':9001,'cal_clusters':64,'refs_per_seed':64,
 'image_geometry':'EXIF transpose -> RGB -> Pillow BICUBIC direct 224 square; no crop',
 'numeric':'FP32 eager, eval, no AMP; ImageNet mean/std; post-final-LN patch L2 eps=1e-8',
 'photometric':'sRGB -> linear L; per-image per-channel spatial mean mu; clip(a*(b*L+(1-b)*mu),0,1) -> sRGB; no uint8; identity exact shortcut',
 'identity_meaning':'theta index 0; preserve original floating array and reuse original feature',
 'near_duplicate':{'phash':'ImageHash.phash(hash_size=8,highfreq_factor=4)', 'hamming_max':4,
   'ssim_gray':'EXIF transpose -> RGB -> L -> Pillow LANCZOS 256 square', 'ssim_min':.995,
   'ssim_parameters':{'data_range':255,'win_size':7,'gaussian_weights':False,'use_sample_covariance':True,'K1':.01,'K2':.03},
   'train_clustering':'connected components on exactRGB-unique nodes',
   'cluster_members':'all original paths including exact aliases; sorted (pixelhash,path)',
   'cluster_id':'SHA256(newline-joined pixelhash + TAB + relative path), no terminal newline',
   'canonical':'minimum (pixelhash,relative path)',
   'cross_split':'same-label exact duplicates remove all train members; conflict stops; near duplicates reported only'},
 'distance':'FP32 exhaustive squared L2 on unit patches; no approximate retrieval',
 'ties':'theta order, then canonical ref (pixelhash,path), then row-major patch index; strict less update',
 'primary':{'theta_selection':'mean lowest 204 of 256 distances','score':'mean largest 13 squared distances'},
 'mechanism_controls':['all256_mean_theta_selection','minimum_top13_mean_over_theta'],
 'reference_selection':{'selectors':['spatial_bow_KL','trimmed_chamfer'],'K':[1,4,10],'bank_modes':['selected_view','pool9'],
   'kmeans':{'n_clusters':12,'init':'k-means++','random_state':0,'n_init':10,'algorithm':'lloyd','max_iter':300,'tol':1e-4},
   'bow':'5x5 grid by floor(5*(patch_index+.5)/16); 12 counts +1e-6 then normalize; KL(h_ref||h_query), lowest20/25 mean',
   'chamfer':'lowest204/256 mean of A[:,r,theta]; choose theta per ref, then K distinct refs'},
 'robust_color':{'statistics':'per-channel median and IQR in linear RGB, numpy quantile method=linear',
   'floor':1e-4,'target':'cross-reference medians of per-image median and floored IQR',
   'transform':'S(clip(mbar+sbar*(L-m)/max(IQR,1e-4),0,1))','feature_reuse':False},
 'projection':{'fit':'CPU float64 E(delta), S0=E(delta delta^T), Sc=S0-mu mu^T',
   'delta':'unit augmented patch minus unit identity at same position; all64x8x256',
   'ranks':[8,16,32],'matrices':['S0','Sc'],'top32':'single eigh per matrix; nested first8/16/32',
   'inference_centering':False,'pre':'Qz on query and entire bank, L2 eps1e-8, new exhaustive NN',
   'post':'Q(original-space winning residual), no renormalization; squared and sqrt top13 means',
   'banks':['identity','pool9'],'configuration_count':36},
 'calibration':{'n':64,'threshold_ascending_order':62,'positive':'score > threshold','ties':'normal','alpha':.05,'target_FPR_guarantee':False},
 'evaluation':{'test_counts':{'same':{'normal':230,'anomaly':459},'illumination':{'normal':75,'anomaly':198}},
   'equal_domain_AUROC':'(same + illumination)/2','baseline_oracle':'investment gate only; never deployment selection or significance',
   'gates':{'eq_delta':.02,'illumination_delta':.03,'same_delta_min':-.01,'all_seed_eq_delta_strictly_positive':True,'primary_minus_shuffle9001_eq':.01},
   'CI':'none; no physical entity IDs; only official image-split acquisition-condition generalization',
   'masks_used':False},
 'budgets':{'preflight_MPS_window_seconds':300,'future_total_from_first_GPU_wall_seconds':7200,'stage_wall_seconds':14400,'disk_bytes':8*1024**3,'forecast_margin':1.2},
 'method_count':59,'baseline_count':51,
}

def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()

def file_sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def save_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temporary.replace(path)

def query_manifest_records(manifest):
    calibration=[{'query_id':'cal/'+c['cluster_id'],'path':c['canonical']['path'],'kind':'calibration','domain':'calibration','label':0,'pixelhash':c['canonical']['pixelhash']} for c in manifest['calibration']]
    tests=[{'query_id':'test/'+r['path'],'path':r['path'],'kind':'test','domain':r['domain'],'label':r['label'],'pixelhash':r['pixelhash']} for r in manifest['tests']]
    return calibration+tests
