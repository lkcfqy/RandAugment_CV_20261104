"""All weak/RA tensors equal the first run; native DeiT clean transform contract."""
from common import *
import zipfile,datetime
t=time.perf_counter();m=json.loads((ROOT/"IDENTITIES.json").read_text())
ref=json.loads((ROOT/"REFERENCE_VIEW_HASHES.json").read_text());lookup={r["member"]:r["weak_ra_sha256"] for r in ref["rows"]}
assert sha(MODEL)=="a1311bcf4f24e3c95adaa75535db67bc4412d95535b98f7c1dfd1164dda41c97"
matched=0
with zipfile.ZipFile(ZIP) as z:
    for entry in m["support_union"]:
        view=make_views(read_image(z,entry),entry["rgb_sha256"])
        hashes=[hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in view[1:]]
        assert hashes==lookup[entry["member"]]
        matched+=len(hashes)
timm=timm_module()
cfg=json.loads((PROJECT/"research-view-route-20261030/sources/timm-config.json").read_text())["pretrained_cfg"]
native=timm.data.create_transform(input_size=tuple(cfg["input_size"]),is_training=False,mean=tuple(cfg["mean"]),std=tuple(cfg["std"]),interpolation=cfg["interpolation"],crop_pct=cfg["crop_pct"])
im=Image.fromarray(np.arange(64*64*3,dtype=np.uint8).reshape(64,64,3))
assert torch.equal(native(im),TENSOR(CLEAN(im)))
report={"status":"PASS","matched_originals":len(m["support_union"]),"weak_ra_bitwise_equal_views":matched,"clean_resize":248,"dinov2_clean_resize":256,"native_timm_clean_transform_bitwise_equal":True,"weight_sha256":sha(MODEL),"weight_bytes":MODEL.stat().st_size,"timm_version":timm.__version__,"timm_file":timm.__file__,"model_forwards":0,"seconds":time.perf_counter()-t,"utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}
atomic_json("SECOND_ENCODER_CONTRACT_TEST.json",report);print(json.dumps(report,indent=2))

