"""Fixed two-shift/two-model CPU analysis. --synthetic is explicitly nonempirical."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import io
import json
import math
import os
import sys
import zipfile

import numpy as np
from cns_core import MODELS, SHIFTS, SCALES, assemble, align_models, SchemaError
from cns_assets import (sha256, audit_annotations, load_prediction_zip,
                        check_predictions_against_annotations, verify_local_baseline_manifest)
from cns_statistics import (make_design, bootstrap, resource_gate, summaries, STAT_NAMES)


def clean_json(value):
    if isinstance(value, dict): return {str(k):clean_json(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [clean_json(v) for v in value]
    if isinstance(value, np.generic): return clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(clean_json(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def write_csv(path, rows):
    rows=list(rows)
    if not rows: return
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else clean_json(v) for k,v in row.items()})


def make_synthetic_assets(output):
    """Never use this fixture as evidence about CNS, classifiers, or nuisance shifts."""
    annotation={'info':{'description':'SYNTHETIC NONEMPIRICAL pipeline fixture',
                        'version':'SYNTHETIC-1','license':'Test fixture; not CNS release data'},
                'scales':list(SCALES),'classes':[2,4,7],'shifts':list(SHIFTS),'images':[]}
    image_records=[]
    for shift in SHIFTS:
        for c in annotation['classes']:
            for seed in range(1,13):
                if shift==SHIFTS[0] and c==7 and seed==12: continue
                for scale in SCALES:
                    name=f'seed_{seed:04d}_scale_{round(scale*1000):04d}.jpg'
                    image_records.append({'image':f'{shift}/{c:03d}_synthetic/{name}',
                                          'seed':seed,'scale':scale,'shift':shift,'class':c})
    annotation['images']=image_records
    metadata=output/'SYNTHETIC_NONEMPIRICAL_annotations.json'
    write_json(metadata,annotation)
    archive_path=output/'SYNTHETIC_NONEMPIRICAL_predictions.zip'
    with zipfile.ZipFile(archive_path,'x',zipfile.ZIP_DEFLATED) as archive:
        archive.comment=b'SYNTHETIC NONEMPIRICAL TEST DATA -- NOT CNS BENCH RESULTS'
        for m,model in enumerate(MODELS):
            for s,shift in enumerate(SHIFTS):
                buffer=io.StringIO()
                fields=['pred_class','gt_class','shift','class_str','img_name','class_name','seed','scale',model]
                writer=csv.DictWriter(buffer,fieldnames=fields); writer.writeheader()
                records=[r for r in image_records if r['shift']==shift]
                for record in reversed(records):  # Deliberate order adversary.
                    c,seed,scale=record['class'],record['seed'],record['scale']
                    if m==0 and s==1 and c==4 and seed==10 and scale==2.5: continue
                    y=[True]*6
                    if seed%2==0 and s==0: y=[True,True,True,False,True,False]
                    if seed%3==0 and s==1: y=[True,False,True,False,False,False]
                    if m==1 and s==1 and seed==1: y=[False,True,True,True,True,True]
                    if m==1 and s==0 and seed==7: y=[True,True,False,False,False,False]
                    correct=y[SCALES.index(scale)]
                    writer.writerow({'pred_class':c if correct else c+1,'gt_class':c,'shift':shift,
                                     'class_str':f'{c:03d}_synthetic','img_name':Path(record['image']).name,
                                     'class_name':'synthetic','seed':seed,'scale':scale,model:correct})
                archive.writestr(f'SYNTHETIC/{model}__{shift}.csv',buffer.getvalue())
    return archive_path,metadata


def plot_figures(output, tables, boot, synthetic, identity_verified):
    os.environ.setdefault('MPLCONFIGDIR',str(output/'.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    label='SYNTHETIC / NONEMPIRICAL: pipeline check' if synthetic else (
        'Verified identical zero-scale source files' if identity_verified else 'Exploratory class/seed matching; source identity unverified')
    cohort='common_first_observed_correct'
    colors=('#c85524','#247ba0')
    short=('ResNet-18','ViT-B AugReg')
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True)
    for m,model in enumerate(MODELS):
        for e,endpoint in enumerate(('F','T')):
            ax=axes[m,e]
            for s,shift in enumerate(SHIFTS):
                rows=[r for r in tables['event_proportions'] if r['model']==model and r['shift']==shift and r['cohort']==cohort and r['endpoint']==endpoint]
                ax.plot(SCALES,[r['proportion'] for r in rows],marker='o',color=colors[s],label=shift)
            ax.set_title(f'{short[m]}: '+('first error F' if endpoint=='F' else 'terminal wrong suffix T'))
            ax.set_ylabel('Fraction of all eligible trajectories'); ax.set_xlabel('Observed shift scale')
            ax.set_xticks(SCALES); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle(label,fontsize=12,color='#9c2525' if synthetic else 'black')
    fig.tight_layout(rect=(0,0,1,.95))
    fig.savefig(output/'failure_distributions.png',dpi=170); fig.savefig(output/'failure_distributions.svg'); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for m,model in enumerate(MODELS):
        for s,shift in enumerate(SHIFTS):
            rows=[r for r in tables['accuracy'] if r['model']==model and r['shift']==shift and r['cohort']==cohort]
            axes[m].plot(SCALES,[r['accuracy'] for r in rows],marker='o',label=shift,color=colors[s])
        axes[m].set(title=short[m],xlabel='Observed shift scale',ylabel='Accuracy',ylim=(0,1.03),xticks=SCALES)
        axes[m].grid(alpha=.2); axes[m].legend(fontsize=8)
    fig.suptitle(label,fontsize=11,color='#9c2525' if synthetic else 'black')
    fig.tight_layout(rect=(0,0,1,.91)); fig.savefig(output/'accuracy.png',dpi=170); fig.savefig(output/'accuracy.svg'); plt.close(fig)
    if boot.get('simultaneous_primary_intervals') is not None:
        coords=boot['primary_coordinate_indices']; point=np.array(boot['point'])[coords]
        q=boot['simultaneous_half_width']
        fig,ax=plt.subplots(figsize=(9,4))
        ax.errorbar(point*100,np.arange(4),xerr=q*100,fmt='o',capsize=4,color='#214e6f')
        ax.axvspan(-5,5,color='#e8e8e8'); ax.axvline(0,color='#777',linewidth=.8)
        ax.set(yticks=range(4),yticklabels=['ResNet-18 D_C','ResNet-18 D_E','ViT-B D_C','ViT-B D_E'],
               xlabel='Change in cartoon-minus-snow event proportion (percentage points)',title=label)
        ax.grid(axis='x',alpha=.2); fig.tight_layout()
        fig.savefig(output/'contrast_changes.png',dpi=170); fig.savefig(output/'contrast_changes.svg'); plt.close(fig)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--synthetic',action='store_true')
    source.add_argument('--predictions-zip',type=Path)
    parser.add_argument('--annotations',type=Path)
    parser.add_argument('--asset-receipt',type=Path,help='Verified prediction provenance/checksum receipt; real data only')
    parser.add_argument('--identity-manifest',type=Path,help='Optional local baseline image evidence; never downloads images')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--synthetic-replicates',type=int,default=400)
    args=parser.parse_args(argv)
    root=Path(__file__).resolve().parent
    protocol=json.loads((root/'protocol.json').read_text())
    if not args.synthetic:
        if args.annotations is None or args.asset_receipt is None:
            parser.error('Real data require --annotations and --asset-receipt; no implicit download')
        lock=json.loads((root/'protocol-lock.json').read_text())
        if any(sha256(root/name)!=digest for name,digest in lock['files'].items()):
            raise SchemaError('Locked protocol was changed')
        receipt=json.loads(args.asset_receipt.read_text())
        if receipt.get('sha256')!=sha256(args.predictions_zip) or not receipt.get('source_url') or not receipt.get('license'):
            raise SchemaError('Prediction receipt lacks matching SHA-256, source URL, or verified license')
    else:
        receipt={'evidence_type':'SYNTHETIC_NONEMPIRICAL','benchmark_results':False}
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    try:
        predictions,annotations=(make_synthetic_assets(output) if args.synthetic else (args.predictions_zip,args.annotations))
        metadata,annotation_keys,metadata_composition=audit_annotations(annotations)
        rows_by_model,zip_audit=load_prediction_zip(predictions)
        matching=check_predictions_against_annotations(rows_by_model,annotation_keys)
        alignment=align_models(rows_by_model)
        trajectories={}; exclusions=[]
        for model,rows in rows_by_model.items():
            trajectories[model],missing=assemble(rows,model); exclusions.extend(missing)
        key_design=make_design(trajectories)
        verified=set(); identity={'status':'unverified; no baseline image hashes available'}
        if args.identity_manifest:
            verified,identity=verify_local_baseline_manifest(args.identity_manifest,annotation_keys)
        # All common complete source keys must be covered for this first-pass primary analysis.
        required_sources=set()
        for model in MODELS:
            t=trajectories[model]
            required_sources|={key for sh,key in t if all((s,key) in t for s in SHIFTS)}
        identity_verified=bool(required_sources) and required_sources.issubset(verified)
        identity.update(required_common_complete_sources=len(required_sources),verified_sources=len(verified),
                        primary_identity_coverage_complete=identity_verified,
                        equal_predictions_are_not_identity_proof=True)
        # Without full identity evidence, all estimates are explicitly key-matched exploratory estimates.
        design=make_design(trajectories,verified if identity_verified else None)
        tables=summaries(design)
        replicates=args.synthetic_replicates if args.synthetic else protocol['bootstrap_replicates']
        b=bootstrap(design,replicates,protocol['bootstrap_seed'],'stratified_source')
        cluster=bootstrap(design,replicates,protocol['bootstrap_seed']+1,'class_cluster')
        gate=resource_gate(b,identity_verified,args.synthetic)
        cluster_gate=resource_gate(cluster,identity_verified,args.synthetic)
        if gate['decision']!=cluster_gate['decision']:
            gate={'decision':'undetermined_inference_target_sensitivity','source_bootstrap_decision':gate,
                  'class_cluster_decision':cluster_gate}
        evidence='SYNTHETIC_NONEMPIRICAL' if args.synthetic else ('EMPIRICAL_VERIFIED_SOURCE' if identity_verified else 'EMPIRICAL_KEY_MATCHED_EXPLORATORY')
        for name,rows in tables.items():
            if isinstance(rows,list) and rows and isinstance(rows[0],dict):
                write_csv(output/f'{name}.csv',[{'evidence_type':evidence,**r} for r in rows])
        features=[]
        for model,t in trajectories.items():
            for (shift,key),tr in sorted(t.items()):
                features.append({'evidence_type':evidence,'model':model,'shift':shift,'class_id':key[0],
                                 'seed':key[1],**tr.features})
        write_csv(output/'trajectory_features.csv',features)
        write_csv(output/'excluded_incomplete_trajectories.csv',exclusions)
        write_csv(output/'metadata_class_composition.csv',metadata_composition)
        write_json(output/'source_bootstrap.json',b); write_json(output/'class_cluster_bootstrap.json',cluster)
        write_json(output/'asset_audit.json',{'metadata':metadata,'prediction_archive':zip_audit,
                                            'prediction_provenance':receipt,'prediction_metadata_match':matching,
                                            'model_key_alignment':alignment,'identity':identity})
        plot_figures(output,tables,b,args.synthetic,identity_verified)
        result={'status':'completed','evidence_type':evidence,'completed_utc':datetime.now(timezone.utc).isoformat(),
                'models':list(MODELS),'shifts':list(SHIFTS),'source_key_count':len(design.sources),
                'fixed_class_support_K':list(design.fixed_classes),'cohort_audit':design.audits,
                'incomplete_trajectories_excluded':len(exclusions),'source_identity':identity,
                'resource_gate':gate,'strict_same_source_conclusion_available':identity_verified and not args.synthetic,
                'training_performed':False,'model_inference_performed':False,'GPU_requested':False,
                'limitations':['Already-filtered published cohort only; unmatched and eligibility selection can bias comparisons.',
                               'Different shifts have no equal-physical-strength guarantee.',
                               'T describes only the final observed wrong suffix, not permanent or irreversible failure.',
                               'Zero-event or zero-width bootstrap intervals do not prove absence.',
                               'Class-cluster resampling concerns a different inferential target.',
                               'First-observed correctness does not by itself verify a clean source image.']}
        write_json(output/'status.json',result)
        print(json.dumps({'status':'completed','evidence_type':evidence,'output':str(output),'resource_gate':gate},ensure_ascii=False))
    except Exception as exc:
        write_json(output/'status.json',{'status':'failed','error_type':type(exc).__name__,'error':str(exc),
                                        'training_performed':False})
        raise


if __name__=='__main__': main()
