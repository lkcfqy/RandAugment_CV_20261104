#!/usr/bin/env python3
"""Finish verified CIFAR archive extraction; no network, test payload, or training.

The official archive contains fine_label_names.txt~. The earlier strict member
allowlist stopped before extraction. This version recognizes that harmless regular
backup metadata member, leaves it unextracted, and preserves the failure receipt.
"""
from pathlib import PurePosixPath
import datetime
import hashlib
import json
import shutil
import tarfile
import time
import numpy as np
from prepare_cifar100 import ROOT, URL, MD5, OFFICIAL_PAGE, SEED, digest_file


def main():
    start = time.perf_counter()
    old = ROOT / 'preparation.json'
    previous = json.loads(old.read_text())
    failure_receipt = ROOT / 'preparation-range-transfer-and-member-stop.json'
    with failure_receipt.open('x') as f:
        f.write(old.read_text())
    report = dict(previous)
    report.pop('error', None)
    report['status'] = 'finalizing'
    report['finalization_started_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report['member_stop_reason'] = 'Official archive also contains regular fine_label_names.txt~; earlier allowlist stopped before any extraction. Recognized here, not extracted.'
    times = {}
    try:
        archive = ROOT / 'cifar-100-binary.tar.gz'
        t = time.perf_counter()
        assert archive.stat().st_size == 168513733
        assert digest_file(archive, 'md5') == MD5
        assert digest_file(archive) == previous['archive_sha256']
        report['verified_archive_md5'] = MD5
        report['official_page_sha256'] = digest_file(ROOT / 'official-download-page.html')
        times['reverify_archive_seconds'] = time.perf_counter() - t
        expected = {
            'cifar-100-binary': (0, True),
            'cifar-100-binary/fine_label_names.txt': (725, False),
            'cifar-100-binary/train.bin': (153700000, False),
            'cifar-100-binary/coarse_label_names.txt': (328, False),
            'cifar-100-binary/fine_label_names.txt~': (724, False),
            'cifar-100-binary/test.bin': (30740000, False),
        }
        wanted = {'cifar-100-binary/train.bin', 'cifar-100-binary/fine_label_names.txt', 'cifar-100-binary/coarse_label_names.txt'}
        outdir = ROOT / 'train-only'
        assert outdir.is_dir() and not list(outdir.iterdir()), 'preserve any existing extracted files'
        t = time.perf_counter()
        with tarfile.open(archive, 'r:gz') as tar:
            entries = tar.getmembers()
            names = [e.name.rstrip('/') for e in entries]
            assert len(set(names)) == len(names) and set(names) == set(expected)
            for e, name in zip(entries, names):
                p = PurePosixPath(name)
                assert not p.is_absolute() and '..' not in p.parts
                size, directory = expected[name]
                assert e.size == size and e.isdir() == directory
                assert e.isdir() or e.isfile(), 'reject links and devices'
            report['archive_members'] = [{'name': n, 'size': e.size, 'extracted': n in wanted} for e, n in zip(entries, names)]
            for e, name in zip(entries, names):
                if name not in wanted:
                    continue
                target = outdir / PurePosixPath(name).name
                with tar.extractfile(e) as source, target.open('xb') as dest:
                    shutil.copyfileobj(source, dest, length=1024*1024)
                assert target.stat().st_size == e.size
        assert {p.name for p in outdir.iterdir()} == {'train.bin', 'fine_label_names.txt', 'coarse_label_names.txt'}
        times['safe_train_extraction_seconds'] = time.perf_counter() - t
        t = time.perf_counter()
        train = outdir / 'train.bin'
        records = np.memmap(train, mode='r', dtype=np.uint8, shape=(50000, 3074))
        fine = np.asarray(records[:, 1]); coarse = np.asarray(records[:, 0])
        assert np.array_equal(np.bincount(fine, minlength=100), np.full(100, 500))
        assert int(coarse.min()) == 0 and int(coarse.max()) == 19
        fit, dev = [], []
        for c in range(100):
            ids = np.flatnonzero(fine == c).tolist()
            ids.sort(key=lambda i: hashlib.sha256(f'{SEED}|{c}|{i}'.encode('ascii')).digest())
            dev.extend(ids[:100]); fit.extend(ids[100:])
        fit = np.array(sorted(fit), dtype=np.int64); dev = np.array(sorted(dev), dtype=np.int64)
        assert len(fit) == 40000 and len(dev) == 10000 and np.intersect1d(fit, dev).size == 0
        assert np.array_equal(np.sort(np.concatenate([fit, dev])), np.arange(50000))
        assert np.array_equal(np.bincount(fine[fit], minlength=100), np.full(100, 400))
        assert np.array_equal(np.bincount(fine[dev], minlength=100), np.full(100, 100))
        splits = ROOT / 'splits'; splits.mkdir(exist_ok=False)
        np.save(splits / 'fit_indices.npy', fit, allow_pickle=False)
        np.save(splits / 'development_indices.npy', dev, allow_pickle=False)
        times['train_labels_and_split_seconds'] = time.perf_counter() - t
        report['split'] = {'seed_string': SEED, 'method': 'Per fine class, SHA256(seed|class|official TRAIN row index), first 100 development, remaining 400 fit; final arrays sorted.', 'fit': 40000, 'development': 10000, 'fit_per_class': 400, 'development_per_class': 100, 'official_test_used': False}
        report['train_format'] = 'uint8 [50000,3074]: coarse label, fine label, then RGB planar image reshape(3,32,32)'
        t = time.perf_counter()
        report['files'] = [{'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size, 'sha256': digest_file(p)} for p in sorted(outdir.iterdir()) + sorted(splits.iterdir())]
        times['train_and_split_hash_seconds'] = time.perf_counter() - t
        report['status'] = 'ready_train_only'
    except BaseException as e:
        report['status'] = 'blocked'
        report['error'] = f'{type(e).__name__}: {e}'
        raise
    finally:
        report['finalization_timings_seconds'] = times
        report['finalization_wall_seconds'] = time.perf_counter() - start
        report['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        old.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
