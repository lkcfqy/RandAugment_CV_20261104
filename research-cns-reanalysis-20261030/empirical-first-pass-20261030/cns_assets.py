"""Read-only local asset checks. This module never downloads anything."""
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
import csv
import hashlib
import io
import json
import zipfile

from cns_core import SHIFTS, MODELS, SCALES, SchemaError, read_csv_rows


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def audit_annotations(path):
    content=json.loads(Path(path).read_text())
    if tuple(content.get('scales',[])) != SCALES:
        raise SchemaError('Annotation scales differ from fixed protocol')
    keys={}; trajectories=defaultdict(list)
    for row in content['images']:
        if set(row)!={'image','seed','scale','shift','class'}:
            raise SchemaError('Unexpected annotation image schema')
        if row['shift'] not in SHIFTS:
            continue
        key=(row['shift'],row['class'],row['seed'],float(row['scale']))
        if key in keys:
            raise SchemaError('Duplicate annotation key')
        if key[3] not in SCALES:
            raise SchemaError('Unexpected annotation scale')
        image=PurePosixPath(row['image'])
        if len(image.parts)!=3 or image.parts[0]!=row['shift'] or '..' in image.parts:
            raise SchemaError('Unexpected annotation image path')
        if not image.parts[1].startswith(f"{row['class']:03d}_"):
            raise SchemaError('Annotation class path disagrees with class ID')
        expected_stem=f"seed_{row['seed']:04d}_scale_{round(row['scale']*1000):04d}"
        if image.stem!=expected_stem:
            raise SchemaError('Annotation file stem disagrees with scale or seed')
        keys[key]=row
        trajectories[(key[0],key[1],key[2])].append(key[3])
    result={'annotation_info':content['info'],'sha256':sha256(path),
            'total_published_image_records':len(content['images']),
            'selected_image_records':len(keys),'by_shift':{},
            'identity_hash_fields_present':False,
            'strict_source_identity':'unconfirmed: keys and paths contain no common baseline digest',
            'license_verified_from_annotation_info':content['info'].get('license'),
            'version_verified_from_annotation_info':content['info'].get('version')}
    complete={s:set() for s in SHIFTS}
    composition=[]
    for shift in SHIFTS:
        groups={key:scales for key,scales in trajectories.items() if key[0]==shift}
        complete[shift]={(k[1],k[2]) for k,v in groups.items() if tuple(sorted(v))==SCALES}
        counts=Counter(k[0] for k in complete[shift])
        result['by_shift'][shift]={'images':sum(k[0]==shift for k in keys),
                                  'trajectories':len(groups),'complete_trajectories':len(complete[shift]),
                                  'incomplete_trajectories':len(groups)-len(complete[shift]),
                                  'classes':len(counts)}
        for c,n in sorted(counts.items()):
            composition.append({'shift':shift,'class_id':c,'complete_trajectories':n})
    common=complete[SHIFTS[0]] & complete[SHIFTS[1]]
    result['common_class_seed_keys']=len(common)
    result['common_key_classes']=len({c for c,_ in common})
    result['these_counts_are_metadata_not_prediction_results']=True
    return result, keys, composition


def load_prediction_zip(path):
    rows_by_model={m:[] for m in MODELS}
    selected=[]
    with zipfile.ZipFile(path) as archive:
        for model in MODELS:
            for shift in SHIFTS:
                basename=f'{model}__{shift}.csv'
                files=[info for info in archive.infolist() if PurePosixPath(info.filename).name==basename]
                if len(files)!=1:
                    raise SchemaError(f'Expected exactly one release file named {basename}; no model substitution')
                info=files[0]
                if info.file_size>50_000_000:
                    raise SchemaError('Selected CSV exceeds safety size bound')
                with archive.open(info) as binary:
                    rows=read_csv_rows(io.TextIOWrapper(binary,encoding='utf-8-sig',newline=''),model)
                if any(row.shift!=shift for row in rows):
                    raise SchemaError('Shift column disagrees with chosen release filename')
                rows_by_model[model].extend(rows)
                selected.append({'model':model,'shift':shift,'archive_member':info.filename,
                                 'uncompressed_bytes':info.file_size,'rows':len(rows)})
    return rows_by_model, {'zip_sha256':sha256(path),'selected_files':selected,
                           'file_selection':'exact predeclared model and shift filenames'}


def check_predictions_against_annotations(rows_by_model, keys, allow_png_jpg_release_mapping=False):
    audit={}
    for model,rows in rows_by_model.items():
        unknown=[row.key for row in rows if row.key not in keys]
        if unknown:
            raise SchemaError(f'{model}: prediction keys absent from official annotations')
        mismatches=[]; mapped_extensions=0
        for row in rows:
            image=PurePosixPath(keys[row.key]['image'])
            prediction=PurePosixPath(row.img_name)
            exact=row.img_name==image.name
            extension_mapping=(allow_png_jpg_release_mapping and len(prediction.parts)==1
                               and prediction.suffix=='.png' and image.suffix=='.jpg'
                               and prediction.stem==image.stem)
            if (not exact and not extension_mapping) or row.class_str!=image.parts[1]:
                mismatches.append(row.key)
            elif not exact:
                mapped_extensions+=1
        if mismatches:
            raise SchemaError(f'{model}: image/class filename mapping disagrees with metadata; inspect before effects')
        audit[model]={'prediction_rows':len(rows),'annotation_rows':len(keys),
                      'annotation_keys_without_prediction':len(set(keys)-{r.key for r in rows}),
                      'class_and_filename_stems_match_metadata':True,
                      'exact_image_filenames_match_metadata':mapped_extensions==0,
                      'png_to_jpg_extension_mapped_rows':mapped_extensions,
                      'extension_mapping_explicitly_enabled':allow_png_jpg_release_mapping,
                      'image_byte_identity_proven':False}
    return audit


def verify_local_baseline_manifest(path, annotation_keys):
    """Verify actual local bytes. A declared hash alone is never enough."""
    path=Path(path).resolve(); base=path.parent
    hashes={}; evidence=[]
    with path.open(newline='') as handle:
        for record in csv.DictReader(handle):
            key=(record['shift'],int(record['gt_class']),int(record['seed']),0.0)
            if key not in annotation_keys or key[:3] in hashes:
                raise SchemaError('Unknown or duplicate baseline manifest key')
            local=(base/record['local_path']).resolve()
            if not local.is_relative_to(base):
                raise SchemaError('Baseline manifest path escapes its local evidence directory')
            if local.name!=PurePosixPath(annotation_keys[key]['image']).name:
                raise SchemaError('Baseline file name disagrees with official scale-zero image')
            if not local.is_file() or local.stat().st_size>20_000_000:
                raise SchemaError('Baseline evidence file is missing or unexpectedly large')
            digest=sha256(local)
            if record.get('sha256') and record['sha256']!=digest:
                raise SchemaError('Baseline file digest differs from supplied digest')
            hashes[key[:3]]=digest
            evidence.append({'shift':key[0],'class_id':key[1],'seed':key[2],'sha256':digest})
    common={key[1:] for key in hashes if all((s,*key[1:]) in hashes for s in SHIFTS)}
    verified={key for key in common if hashes[(SHIFTS[0],*key)]==hashes[(SHIFTS[1],*key)]}
    aliases=defaultdict(set)
    for key in verified:
        aliases[hashes[(SHIFTS[0],*key)]].add(key)
    duplicate_sources=[sorted(keys) for keys in aliases.values() if len(keys)>1]
    if duplicate_sources:
        raise SchemaError('Same source bytes have multiple class/seed aliases; variant weighting needs a separate protocol')
    return verified, {'verified_source_keys':len(verified),'mismatched_baseline_pairs':len(common-verified),
                      'hash_evidence':evidence,'method':'actual local baseline file SHA-256 equality'}
