#!/usr/bin/env python3
"""Read saved pixels/tensor independently, using only Python's standard library.

No image generation, model training, package installation, or network access.
"""
import hashlib
import json
import math
import pathlib
import statistics
import struct
import sys
import zlib

ROOT = pathlib.Path(__file__).resolve().parent
CHECK_ONLY = '--check' in sys.argv[1:]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_png(path):
    """Validate CRCs and reverse PNG filters for our saved RGBA8 images."""
    data = path.read_bytes()
    assert data[:8] == b'\x89PNG\r\n\x1a\n'
    pos, compressed, dims = 8, [], None
    while pos < len(data):
        n = struct.unpack('>I', data[pos:pos + 4])[0]
        kind = data[pos + 4:pos + 8]
        block = data[pos + 8:pos + 8 + n]
        crc = struct.unpack('>I', data[pos + 8 + n:pos + 12 + n])[0]
        assert zlib.crc32(kind + block) & 0xffffffff == crc
        if kind == b'IHDR':
            w, h, depth, color, compression, filtering, interlace = struct.unpack('>IIBBBBB', block)
            assert (depth, color, compression, filtering, interlace) == (8, 6, 0, 0, 0)
            dims = w, h
        if kind == b'IDAT':
            compressed.append(block)
        pos += n + 12
        if kind == b'IEND':
            break
    assert pos == len(data) and dims
    w, h = dims
    stride, bpp = 4 * w, 4
    raw = zlib.decompress(b''.join(compressed))
    assert len(raw) == h * (stride + 1)
    prior, pixels, offset = bytearray(stride), bytearray(), 0
    for _ in range(h):
        f = raw[offset]
        row = bytearray(raw[offset + 1:offset + stride + 1])
        offset += stride + 1
        assert 0 <= f <= 4
        for i in range(stride):
            a = row[i - bpp] if i >= bpp else 0
            b = prior[i]
            c = prior[i - bpp] if i >= bpp else 0
            if f == 0:
                predictor = 0
            elif f == 1:
                predictor = a
            elif f == 2:
                predictor = b
            elif f == 3:
                predictor = (a + b) // 2
            else:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                predictor = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
            row[i] = (row[i] + predictor) & 255
        pixels.extend(row)
        prior = row
    return w, h, pixels


def dark_fraction(pixels):
    return sum(.2126 * pixels[i] + .7152 * pixels[i + 1] + .0722 * pixels[i + 2] < 128
               for i in range(0, len(pixels), 4)) / (len(pixels) / 4)


def ols(rows):
    xs = [r['dark_fraction_image_only'] for r in rows]
    ys = [r['target_T'] for r in rows]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - slope * mx, slope


def assess(rows, intercept, slope):
    points = [dict(background=r['background'], target_T=r['target_T'],
                   predicted_T=intercept + slope * r['dark_fraction_image_only'],
                   was_cropped=r['was_cropped']) for r in rows]
    errors = [p['predicted_T'] - p['target_T'] for p in points]
    return dict(n=len(points), mae=statistics.mean(map(abs, errors)),
                max_absolute_error=max(map(abs, errors)),
                rmse=math.sqrt(statistics.mean(e * e for e in errors)), points=points)


def information(joint):
    n, k = len(joint), 14
    pc = [sum(joint[y][c * k + t] for y in range(n) for t in range(k)) for c in range(n)]
    py = [sum(row) for row in joint]
    pct = [[sum(joint[y][c * k + t] for y in range(n)) for t in range(k)] for c in range(n)]
    pyt = [[sum(row[c * k + t] for c in range(n)) for t in range(k)] for row in joint]
    pt = [sum(row[t] for row in pyt) for t in range(k)]
    cmi = c_t = c_t_y = mi_yc = 0.
    max_yc = 0.
    for c in range(n):
        for t in range(k):
            if pct[c][t] > 0:
                c_t += pct[c][t] * math.log2(pct[c][t] / (pc[c] * pt[t]))
    for y, row in enumerate(joint):
        for c in range(n):
            pyc = sum(row[c * k:(c + 1) * k])
            max_yc = max(max_yc, abs(pyc - py[y] * pc[c]))
            if pyc > 0:
                mi_yc += pyc * math.log2(pyc / (py[y] * pc[c]))
            for t in range(k):
                p = row[c * k + t]
                if p > 0:
                    cmi += p * math.log2(p * pt[t] / (pyt[y][t] * pct[c][t]))
                    c_t_y += p * math.log2(p * py[y] / (pyc * pyt[y][t]))
    tv_ct = .5 * sum(abs(pct[c][t] - pc[c] * pt[t]) for c in range(n) for t in range(k))
    return dict(marginal_Y_C_MI_bits=mi_yc, conditional_Y_C_given_T_MI_bits=cmi,
                C_T_MI_bits=c_t, C_T_given_Y_MI_bits=c_t_y,
                chain_rule_identity_error=abs(cmi - (mi_yc + c_t_y - c_t)),
                C_T_total_variation_from_independence=tv_ct,
                max_Y_C_factorization_error=max_yc,
                total_probability=sum(py)), pyt, pc, pt


def save_new(name, value):
    if CHECK_ONLY:
        assert json.loads((ROOT / name).read_text()) == value, 'Saved result differs: ' + name
        return
    with (ROOT / name).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def main():
    raw_path = ROOT / 'pixel-raw-results.json'
    raw = json.loads(raw_path.read_text())
    rows = raw['records']
    assert len(rows) == 28
    assert len({(r['background'], r['target_T']) for r in rows}) == 28
    png_rows = []
    max_rgb_error = 0.
    for r in rows:
        path = ROOT / r['file']
        w, h, pixels = read_png(path)
        assert (w, h) == (256, 256)
        assert all(pixels[i] == 255 for i in range(3, len(pixels), 4))
        delta = abs(dark_fraction(pixels) - r['dark_fraction_image_only'])
        assert delta < 1e-14
        max_rgb_error = max(max_rgb_error, delta)
        png_rows.append(dict(file=r['file'], bytes=path.stat().st_size, sha256=digest(path)))
    w, h, fg = read_png(ROOT / 'images/foreground-manual-outline.png')
    alpha_factor = sum(fg[3::4]) / (255 * w * h)
    assert abs(alpha_factor - raw['manual_alpha_factor']) < 1e-12
    background_only_statistics = []
    for filename in ['background-0.png', 'background-1.png']:
        bw, bh, pixels = read_png(ROOT / 'images' / filename)
        assert (bw, bh) == (256, 256)
        assert all(pixels[i] == 255 for i in range(3, len(pixels), 4))
        background_only_statistics.append(dict(file=filename, dark_fraction=dark_fraction(pixels)))
    fit = [r for r in rows if r['role'] == 'fit']
    held = [r for r in rows if r['role'] == 'heldout-T-and-bg']
    assert [r['background'] for r in fit] == [0] * 4
    assert [r['background'] for r in held] == [1] * 4
    assert not set(r['target_T'] for r in fit) & set(r['target_T'] for r in held)
    intercept, slope = ols(fit)
    nocrop = [r for r in rows if not r['was_cropped']]
    probes = [r for r in rows if r['background'] == 1 and r['target_T'] >= .22]
    summaries = dict(
        status='controlled_one_photo_pixel_carrier_check_completed',
        interpretation='Only a known cat against two bright same-photo crops; not ForNet RGB or source-category decodability, visual-shortcut or OOD evidence.',
        calibration=dict(statistic='fraction of opaque RGB pixels with luminance <128',
                         formula='T_hat = intercept + slope * dark_fraction',
                         intercept=intercept, slope=slope, fit_points=4,
                         fit=assess(fit, intercept, slope),
                         heldout_T_and_background=assess(held, intercept, slope),
                         cropped_extrapolation_probes=assess(probes, intercept, slope)),
        no_crop=dict(n=len(nocrop),
                     max_abs_visible_alpha_minus_T=max(abs(r['visible_alpha_fraction'] - r['target_T']) for r in nocrop)),
        background_only_statistics=background_only_statistics,
        oracle_alpha_note='Visible alpha uses the compositing mask and is not automatically provided to the default classifier.',
        area_examples=[dict(target_T=r['target_T'], visible_alpha_fraction=r['visible_alpha_fraction'],
                            retained_fraction_of_requested_T=r['visible_alpha_fraction'] / r['target_T'],
                            cropped=r['was_cropped']) for r in rows if r['background'] == 0 and r['target_T'] in [.05, .15, .5, 1.2]],
        causal_interpretation='Positive I(Y;C|T) is expected collider dependence. It is not unconditional Y-C dependence or proof of a new causal mechanism.',
        source_class_C_image_only_recovery_tested=False,
        full_class_conditional_pixel_distribution_tested=False,
        preprocessing='manual foreground outline, same-photo background crops, centered resize/crop/paste, CoreGraphics interpolation',
        pixel_exact_ForAug_reproduction=False,
        no_ARIS_no_GPU_no_new_download_no_install=True)
    save_new('pixel-results.json', summaries)
    joint_path = ROOT.parent / 'scale-range-joint.json'
    d = json.loads(joint_path.read_text())
    assert digest(joint_path) == 'f11b3a214c38d8e9c0f2c72f461af78e0f01a2c76a3dbeb79699acda77c79d89'
    original, pyt, pc, pt = information(d['joint_rows'])
    shadow = [[pyt[y][t] * pc[c] for c in range(len(pc)) for t in range(14)] for y in range(len(pyt))]
    control, spy, spc, spt = information(shadow)
    assert original['max_Y_C_factorization_error'] < 1e-15
    assert original['chain_rule_identity_error'] < 1e-12
    assert abs(control['conditional_Y_C_given_T_MI_bits']) < 1e-12
    margin_error = max(abs(a - b) for row, srow in zip(pyt, spy) for a, b in zip(row, srow))
    assert margin_error < 1e-12
    mechanism = dict(status='conditional_dependence_is_collider_consistent', original=original, independent_ratio_control=control,
                     max_P_Y_T_margin_difference=margin_error,
                     control_preserves=['P(F,T) in the source-index SCM', 'P(Y,T)', 'P(B)', 'P(F,B)'],
                     control_changes=['P(B,T)', 'P(F,B,T)', 'generally P(X) and P(X|Y), including background-scale/visibility relationships'],
                     does_not_prove='C is visible, a classifier uses C with T, accuracy/OOD harm, or an unknown real-world causal graph.',
                     input_tensor_sha256=digest(joint_path))
    save_new('mechanism-results.json', mechanism)
    verification = dict(status='passed_saved_PNG_and_tensor_readback',
                        independent_PNG_decoder='Python standard library, PNG CRC/zlib/filter reversal; no imaging dependency',
                        composites_checked=28, composite_alpha_all_opaque=True,
                        max_image_only_statistic_readback_error=max_rgb_error,
                        first_verification_failure={'assertion': 'both bright backgrounds have exactly zero dark pixels',
                                                    'actual': 'background 0 has 1/65536 dark pixels; background 1 has zero',
                                                    'resolution': 'Removed the unjustified zero-background assumption; record measured baselines, keep raw pixels and all fit/test roles unchanged.'},
                        alpha_factor_readback=alpha_factor,
                        fit_test_T_disjoint=True, fit_background_0_test_background_1=True,
                        manual_review={'performed': True, 'files': ['images/foreground-manual-outline.png', 'pixel-carrier-contact-sheet.png'],
                                       'finding': 'Upright real cat retained; rough outline includes some light ledge. Both crops appear cat-free. Large targets visibly crop body/head. This is visual inspection, not segmentation-quality validation.'},
                        saved_PNG_files=png_rows, raw_result_sha256=digest(raw_path),
                        swift_source_sha256=digest(ROOT / 'pixel_check.swift'),
                        protocol_sha256=digest(ROOT / 'protocol.md'),
                        mechanism_tensor_sha256=digest(joint_path),
                        limitations=['One photograph; no unseen-object test.', 'Background crops are from the same bright scene.',
                                     'No nominal-source-category C in this photo experiment.', 'No trained classifier or OOD metric.',
                                     'Oracle alpha is not a default model input.', 'Synthetic compositing mechanism check, not a real CV contribution yet.'])
    save_new('pixel-verification.json', verification)
    print(json.dumps(dict(status='verified', holdout_MAE=summaries['calibration']['heldout_T_and_background']['mae'],
                          cropped_extrapolation_MAE=summaries['calibration']['cropped_extrapolation_probes']['mae'],
                          C_T_MI_bits=original['C_T_MI_bits'], C_T_TV=original['C_T_total_variation_from_independence'],
                          conditional_MI_bits=original['conditional_Y_C_given_T_MI_bits']), indent=2))


if __name__ == '__main__':
    main()
