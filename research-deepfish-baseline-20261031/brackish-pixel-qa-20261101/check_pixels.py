"""Bounded local decode QA: eight pre-frozen identities, no model or label modification."""
from portable_dev.paths import project_path as _portable_project_path
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import json, csv, hashlib, collections, datetime, time, sys
import cv2
import numpy as np

P = Path(__file__).resolve().parent
R = _portable_project_path(__file__, 'research-deepfish-baseline-20261031')
A = R / 'brackish-asset-check-20261101'
D = A / 'original_v4'
protocol = json.loads((P / 'FROZEN_QA_PROTOCOL.json').read_text())
deadline = datetime.datetime.fromisoformat(protocol['deadline_UTC']).timestamp()
cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)

def guard():
    assert time.time() < deadline, '15 minute deadline reached'

def sha(f):
    h = hashlib.sha256()
    with f.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()

def capture(path):
    x = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG,
                         [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE,
                          cv2.CAP_PROP_N_THREADS, 1])
    assert x.isOpened(), str(path)
    return x

vs = list(csv.DictReader((A / 'reports/VIDEO_COVERAGE.csv').open()))
video_info = {}
for v in vs:
    guard()
    f = D / v['archive_path']
    with f.open('rb') as stream: magic = stream.read(12)
    assert magic[:4] == b'RIFF' and magic[8:12] == b'AVI '
    cap = capture(f)
    code = int(cap.get(cv2.CAP_PROP_FOURCC))
    info = {
        'video': v['video'], 'archive_path': v['archive_path'], 'bytes': f.stat().st_size,
        'RIFF_AVI_header_hex': magic.hex(), 'backend': cap.getBackendName(),
        'codec_FOURCC': ''.join(chr((code >> (8*i)) & 255) for i in range(4)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'container_reported_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'fps': cap.get(cv2.CAP_PROP_FPS), 'YOLO_filename_count': int(v['listed_frame_count']),
        'all_frames_decoded': False, 'pixel_samples_retrieved': 0
    }
    cap.release()
    video_info[v['video']] = info
(P / 'VIDEO_CONTAINER_INVENTORY.json').write_text(json.dumps(list(video_info.values()), indent=2)+'\n')

anns = collections.defaultdict(list)
categories = {}
for split in ('train', 'valid', 'test'):
    x = json.loads((D / f'annotations/annotations_COCO/{split}_groundtruth.json').read_text())
    images = {i['id']: i for i in x['images']}
    categories.update({i['id']: i['name'] for i in x['categories']})
    for a in x['annotations']:
        anns[images[a['image_id']]['file_name']].append({**a, 'original_split': split})
ledger = {x['filename']: x for x in csv.DictReader((A / 'reports/ALL_FRAME_QUALIFICATION.csv').open())}
by_video = collections.defaultdict(list)
for row in protocol['samples']: by_video[row['video']].append(row)
result = []
(P / 'frames_960x540').mkdir()
(P / 'overlays').mkdir()
for video, samples in by_video.items():
    guard()
    info = video_info[video]
    f = D / info['archive_path']
    original = next(v for v in vs if v['video'] == video)
    assert sha(f) == original['sha256']
    cap = capture(f)
    selected = {s['frame_number']: s for s in samples}
    # Sequential grab establishes decoder frame identity without materializing skipped images.
    for number in range(1, max(selected)+1):
        guard()
        before = cap.get(cv2.CAP_PROP_POS_FRAMES)
        assert cap.grab(), (video, number)
        if number not in selected: continue
        ok, native = cap.retrieve()
        assert ok and native is not None and native.dtype == np.uint8
        after = cap.get(cv2.CAP_PROP_POS_FRAMES)
        assert before == number-1 and after == number
        name = f'{video}-{number:04d}.png'
        assert name in ledger
        image = cv2.resize(native, (960,540), interpolation=cv2.INTER_CUBIC)
        assert cv2.imwrite(str(P/'frames_960x540'/name), image)
        overlay = image.copy()
        for a in anns[name]:
            xx, yy, w, h = a['bbox']
            color = (60,255,60) if a['category_id'] in (1,2) else (0,170,255)
            cv2.rectangle(overlay, (round(xx),round(yy)), (round(xx+w),round(yy+h)), color, 1)
        header = f'frame={number} decoder_index={number-1} boxes={len(anns[name])} {ledger[name]["coverage"]}'
        cv2.rectangle(overlay, (0,0),(960,24),(0,0,0),-1)
        cv2.putText(overlay,header,(4,17),cv2.FONT_HERSHEY_SIMPLEX,0.43,(255,255,255),1,cv2.LINE_AA)
        assert cv2.imwrite(str(P/'overlays'/name),overlay)
        record = {
            **selected[number], 'filename':name, 'source_video_SHA256':original['sha256'],
            'native_shape_HWC':list(native.shape), 'native_dtype':str(native.dtype),
            'native_decoded_BGR_SHA256':hashlib.sha256(native.tobytes()).hexdigest(),
            'decode_backend':'OpenCV CAP_FFMPEG CPU, sequential grab/retrieve',
            'decoder_index_before':before,'decoder_index_after':after,
            'resize_for_QA':'OpenCV INTER_CUBIC 960x540; not asserted pixel-identical to official ffmpeg bicubic',
            'image_shape_HWC':list(image.shape),'image_PNG_SHA256':sha(P/'frames_960x540'/name),
            'overlay_PNG_SHA256':sha(P/'overlays'/name),
            'coverage':ledger[name]['coverage'],'original_splits':ledger[name]['published_splits'],
            'annotations_unchanged':anns[name], 'YOLO_rows':int(ledger[name]['YOLO_rows']),
            'fish_target_boxes':int(ledger[name]['fish_target_boxes']),
            'original_bbox_coordinates_not_scaled_or_clipped':True,
            'pixel_variance':float(native.var()),
            'zero_animal_annotations_not_claimed_visual_absence':True
        }
        result.append(record)
        info['pixel_samples_retrieved'] += 1
    cap.release()
assert len(result) == len({x['filename'] for x in result}) == 8
summary = {
    'status':'EIGHT_FROZEN_FRAMES_DECODED_PENDING_VISUAL_INSPECTION',
    'completed_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'python':sys.version,'OpenCV':cv2.__version__,'opencv_threads':cv2.getNumThreads(),
    'hardware_acceleration':'explicit VIDEO_ACCELERATION_NONE; OpenCL disabled',
    'video_files':89,'video_bytes':sum(x['bytes'] for x in video_info.values()),
    'container_codec_counts':dict(collections.Counter(x['codec_FOURCC'] for x in video_info.values())),
    'container_dimensions_counts':dict(collections.Counter(f"{x['width']}x{x['height']}" for x in video_info.values())),
    'container_reported_total_frames':sum(x['container_reported_frames'] for x in video_info.values()),
    'container_vs_metadata_frame_count_differences':[x for x in video_info.values() if x['container_reported_frames'] != x['YOLO_filename_count']],
    'selected_unique_frames_retrieved':8,'samples':result,
    'model_calls':0,'training_steps':0,'installs':0,'new_download_bytes':0,'upload_bytes':0,'new_CU':0
}
(P/'PIXEL_QA.json').write_text(json.dumps(summary,indent=2)+'\n')
(P/'VIDEO_CONTAINER_INVENTORY.json').write_text(json.dumps(list(video_info.values()),indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='samples'},indent=2))
for x in result:
    print(json.dumps({k:x[k] for k in ('filename','native_shape_HWC','decoder_index_before','decoder_index_after','fish_target_boxes','YOLO_rows','coverage')}))
