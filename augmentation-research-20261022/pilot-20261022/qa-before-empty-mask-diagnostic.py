"""Source/duplicate/visibility preparation. No model fitting or test pixels."""
from common import *
from PIL import ImageDraw,ImageOps
from collections import defaultdict
from unittest.mock import patch
from torchvision.transforms import autoaugment as AA
import zlib

def mask_visibility(im,mask,source_id):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(keyseed('weak-v1',20261022,0,source_id))
        i,j,h,w=T.RandomResizedCrop.get_params(im,scale=(.8,1.),ratio=(.9,1.1))
        cropped=np.asarray(mask)[i:i+h,j:j+w]
        weak=TF.resized_crop(mask,i,j,h,w,(224,224),InterpolationMode.NEAREST)
        if torch.rand(())<.5:weak=TF.hflip(weak)
    ops=[];original=AA._apply_op
    def capture(image,op_name,magnitude,interpolation,fill):
        ops.append({'operation':op_name,'magnitude':magnitude})
        return original(image,op_name,magnitude,interpolation,fill)
    with patch.object(AA,'_apply_op',side_effect=capture):
        traced=augmented_pil(im,source_id,0,20261022,'ra')
    # Verify the observer did not alter the actual training transform.
    assert traced.tobytes()==augmented_pil(im,source_id,0,20261022,'ra').tobytes()
    transported=weak.copy()
    for op in ops:
        if op['operation'] in {'ShearX','ShearY','TranslateX','TranslateY','Rotate'}:
            transported=original(transported,op['operation'],op['magnitude'],InterpolationMode.NEAREST,0)
    fg=int((np.asarray(mask)==1).sum());weak_fg=int((np.asarray(weak)==1).sum())
    ra_fg=int((np.asarray(transported)==1).sum())
    assert fg>0
    return {'source_id':source_id,'epoch':0,'seed':20261022,'RA_operations':ops,
        'weak_original_foreground_retained_fraction':float((cropped==1).sum()/fg),
        'RA_to_weak_foreground_pixel_count_ratio':float(ra_fg/max(1,weak_fg)),
        'RA_visible_foreground_pixels':ra_fg,
        'geometric_proxy_only_not_a_breed_semantic_certificate':True}

def main():
    out=ROOT/'qa';out.mkdir(exist_ok=False)
    plan=json.loads(PLAN_PATH.read_text())
    rows=[dict(r) for r in plan['rows'] if r['candidate_partition']!='reserved_confirmation']
    source_metadata=Path('/Users/lkc/Downloads/RandAugment_CV_20261104/causal-augmentation-research-20261020/area-pairing-cpu-20261021/mechanism-transfer-pets-20261021/archive-text-3.txt')
    source_labels={}
    for line in source_metadata.read_text().splitlines():
        fields=line.split()
        if fields and not line.startswith('#'):source_labels[fields[0]]=tuple(map(int,fields[1:]))
    breed_prefix={}
    for r in rows:
        assert source_labels[r['source_id']][:2]==(r['class_id']+1,r['species']+1)
        breed=r['source_id'].rsplit('_',1)[0]
        breed_prefix.setdefault(r['class_id'],breed)
        assert breed_prefix[r['class_id']]==breed
    receipts=json.loads((ROOT/'data/photo-mask-receipt.json').read_text())
    assert receipts['photos']==receipts['masks']==1184
    exact=defaultdict(list);hashes=[];issues=[]
    for r in rows:
        path=ROOT/'data/images'/(r['source_id']+'.jpg')
        mask_path=ROOT/'data/masks'/(r['source_id']+'.png')
        with Image.open(path) as f:
            orientation=f.getexif().get(274,1)
            im=f.convert('RGB');im.load()
        with Image.open(mask_path) as f:mask=np.asarray(f)
        assert mask.shape==(im.height,im.width)
        assert set(np.unique(mask)).issubset({1,2,3})
        if orientation!=1:issues.append({'source_id':r['source_id'],'issue':'EXIF_orientation_requires_resolution'})
        rgb_hash=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
        exact[rgb_hash].append(r['source_id'])
        small=np.asarray(im.convert('L').resize((9,8)))
        bits=(small[:,1:]>small[:,:-1]).flatten()
        dh=sum(int(b)<<i for i,b in enumerate(bits))
        hashes.append((r['source_id'],dh))
        r.update(image_sha256=sha(path),mask_sha256=sha(mask_path),decoded_RGB_sha256=rgb_hash,
            width=im.width,height=im.height,foreground_fraction=float((mask==1).mean()),
            unknown_boundary_fraction=float((mask==3).mean()),dhash64=f'{dh:016x}')
    near=[]
    for i,(a,ha) in enumerate(hashes):
        for b,hb in hashes[i+1:]:
            distance=(ha^hb).bit_count()
            if distance<=4:near.append({'a':a,'b':b,'distance':distance})
    exact_groups=[v for v in exact.values() if len(v)>1]
    visibility=[]
    for r in rows:
        if r['candidate_partition']=='train':
            im=Image.open(ROOT/'data/images'/(r['source_id']+'.jpg')).convert('RGB')
            mask=Image.open(ROOT/'data/masks'/(r['source_id']+'.png')).copy()
            visibility.append(mask_visibility(im,mask,r['source_id']))
    dump(out/'geometry-visibility.json',{'scope':'Every candidate training photo, predeclared epoch0/seed20261022; later epochs not certified.',
        'rows':visibility,'zero_foreground_source_ids':[r['source_id'] for r in visibility if r['RA_visible_foreground_pixels']==0],
        'photometric_breed_cue_preservation_proven':False})
    dump(out/'source-qa.json',{'photos':1184,'masks':1184,'all_image_mask_dimensions_and_trimap_values_valid':True,
        'all_source_breed_and_species_metadata_match':True,'class_names':breed_prefix,
        'trainval_metadata_sha256':sha(source_metadata),
        'exact_decoded_duplicate_groups':exact_groups,'dhash_distance4_candidate_pairs':near,
        'issues':issues,'rows':rows,'reserved_confirmation_decoded':False,
        'identity_or_pretraining_overlap_verified':False})
    # One preselected training photograph per class, before model outputs exist.
    examples=[]
    for c in range(37):
        choices=[r for r in rows if r['candidate_partition']=='train' and r['class_id']==c]
        examples.append(choices[0])
    for page in range(4):
        group=examples[page*10:(page+1)*10]
        sheet=Image.new('RGB',(672,len(group)*250),(250,250,250));draw=ImageDraw.Draw(sheet)
        for j,r in enumerate(group):
            im=Image.open(ROOT/'data/images'/(r['source_id']+'.jpg')).convert('RGB')
            views=[ImageOps.pad(im,(224,224),color=(245,245,245)),augmented_pil(im,r['source_id'],0,20261022,'weak'),augmented_pil(im,r['source_id'],0,20261022,'ra')]
            for k,v in enumerate(views):sheet.paste(v,(224*k,250*j+22))
            draw.text((3,250*j+2),f"class{r['class_id']} {r['source_id']}  original | weak | RA",fill='black')
        sheet.save(out/f'visibility-{page}.png')
    if near:
        lookup={r['source_id']:r for r in rows}
        for page in range((len(near)+9)//10):
            group=near[10*page:10*(page+1)]
            sheet=Image.new('RGB',(448,len(group)*248),'white');draw=ImageDraw.Draw(sheet)
            for j,pair in enumerate(group):
                for k,key in enumerate(('a','b')):
                    sid=pair[key]
                    im=Image.open(ROOT/'data/images'/(sid+'.jpg')).convert('RGB')
                    sheet.paste(ImageOps.pad(im,(224,224),color=(245,245,245)),(224*k,248*j+22))
                    draw.text((224*k+2,248*j+2),sid,fill='black')
            sheet.save(out/f'near-duplicates-{page}.png')
    (out/'attribution.md').write_text('Oxford-IIIT Pet: Parkhi, Vedaldi, Zisserman, Jawahar, Cats and Dogs (CVPR2012). Official current license CC BY-SA4.0; original rights retained. Source: https://www.robots.ox.ac.uk/~vgg/data/pets/ . Fixed public mirror revision003aeb4bc80d584ef307dc24e8b38d26cf9f9e43. Local research only. Contact sheets contain resizing, cropping, flips and RandAugment transformations; derivatives retain attribution and CC BY-SA4.0. Older archive research-use notice remains preserved in original project.\n')
    print(json.dumps({'status':'source_and_contact_sheet_QA_prepared','photos':1184,'duplicate_groups':exact_groups,'near_candidates':near,'issues':issues}),flush=True)

if __name__=='__main__':main()
