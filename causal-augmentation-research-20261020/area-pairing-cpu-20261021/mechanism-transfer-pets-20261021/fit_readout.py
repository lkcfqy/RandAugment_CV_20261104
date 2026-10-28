"""Small fixed RGB-only centroid readouts. Standard library, one CPU process.

Generated frames are grouped by original photograph and heldout macroblock.
No claim that the generated records are independent observations.
"""
import hashlib
import json
import math
import os
import pathlib
import signal
import statistics
import time

ROOT = pathlib.Path(__file__).resolve().parent
DIM = 64 + 64 * 65 // 2


def save(name, value):
    with (ROOT / name).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def normalize(features, mean, scale):
    z = [max(-5., min(5., (v - m) / s)) for v, m, s in zip(features, mean, scale)]
    return z + [z[i] * z[j] for i in range(64) for j in range(i, 64)]


def normalizer(rows):
    n = len(rows)
    mean = [sum(r['features'][i] for r in rows) / n for i in range(64)]
    scale = [max(.02, math.sqrt(sum((r['features'][i] - mean[i]) ** 2 for r in rows) / n)) for i in range(64)]
    return mean, scale


def fit(rows, mean, scale, label='Y'):
    sums, counts = [[0.] * DIM for _ in range(2)], [0, 0]
    for row in rows:
        y = row[label]
        assert y in [0, 1] and len(row['features']) == 64
        counts[y] += 1
        p = normalize(row['features'], mean, scale)
        for i, v in enumerate(p):
            sums[y][i] += v
    centers = [[v / counts[y] for v in sums[y]] for y in range(2)]
    difference = [b - a for a, b in zip(centers[0], centers[1])]
    intercept = (sum(v * v for v in centers[0]) - sum(v * v for v in centers[1])) / DIM
    return dict(mean=mean, scale=scale, difference=difference, intercept=intercept,
                class_counts=counts, dimension=DIM, input='RGB-derived 64 features only')


def predict(features, model):
    p = normalize(features, model['mean'], model['scale'])
    score = model['intercept'] + 2 * sum(v * d for v, d in zip(p, model['difference'])) / DIM
    return score


def metrics(rows, model, label='Y'):
    if not rows:
        return None
    losses, correct = [], []
    for r in rows:
        score = predict(r['features'], model)
        y = r[label]
        losses.append(max(0., score) + math.log1p(math.exp(-abs(score))) - y * score)
        correct.append(int((score > 0) == bool(y)))
    return dict(records=len(rows), mean_log_loss_nats=statistics.mean(losses), accuracy=statistics.mean(correct))


def compare(test, models):
    table = {}
    for trained, model in models.items():
        table[trained] = {policy: metrics([r for r in test if r['policy'] == policy], model)
                          for policy in ['coupled', 'independent']}
    effects = {}
    for trained in models:
        a, b = table[trained]['coupled'], table[trained]['independent']
        effects[trained + '_trained_delta_LL_independent_minus_coupled'] = b['mean_log_loss_nats'] - a['mean_log_loss_nats']
        effects[trained + '_trained_accuracy_drop_coupled_minus_independent'] = a['accuracy'] - b['accuracy']
    effects['difference_in_differences_LL'] = (effects['coupled_trained_delta_LL_independent_minus_coupled'] -
                                              effects['independent_trained_delta_LL_independent_minus_coupled'])
    return dict(table=table, effects=effects)


def main():
    try: os.nice(10)
    except OSError: pass
    signal.alarm(300)
    start, cpu = time.perf_counter(), time.process_time()
    rows = [json.loads(line) for line in (ROOT / 'rgb-features.jsonl').open()]
    cases = json.loads((ROOT / 'render-cases.json').read_text())['rows']
    lookup = {r['case']: r for r in cases}
    assert len(rows) == 9344 and len(cases) == 320
    assert len({r['foreground_id'] for r in rows if r['split']=='fit'} &
               {r['foreground_id'] for r in rows if r['split']=='heldout'}) == 0
    result, saved_models = {}, {}
    for layout in ['protected', 'ordinary']:
        subset = [r for r in rows if r['layout'] == layout]
        train = [r for r in subset if r['split'] == 'fit']
        test = [r for r in subset if r['split'] == 'heldout']
        mean, scale = normalizer([r for r in train if r['policy'] == 'coupled'])
        models = {policy: fit([r for r in train if r['policy'] == policy], mean, scale)
                  for policy in ['coupled', 'independent']}
        overall = compare(test, models)
        blocks = {str(b): compare([r for r in test if r['block'] == b], models) for b in range(4)}
        crop_strata = {}
        for cropped in [False, True]:
            ss = [r for r in test if lookup[r['case']]['cropped'] == cropped]
            crop_strata[str(cropped)] = compare(ss, models) if ss else None
        by_class = {str(c): compare([r for r in test if r['Y'] == c], models) for c in [0, 1]}
        deltas = [blocks[str(b)]['effects']['coupled_trained_delta_LL_independent_minus_coupled'] for b in range(4)]
        did = [blocks[str(b)]['effects']['difference_in_differences_LL'] for b in range(4)]
        result[layout] = dict(overall=overall, heldout_source_disjoint_macroblocks=blocks,
                              per_class=by_class, crop_strata=crop_strata,
                              block_delta_LL_min=min(deltas), block_delta_LL_max=max(deltas),
                              positive_delta_LL_blocks=sum(x>0 for x in deltas),
                              positive_DID_blocks=sum(x>0 for x in did),
                              cautious_pilot_screen=all(x>0 for x in deltas) and all(x>0 for x in did))
        saved_models[layout] = models
        print(json.dumps({'phase':layout, 'delta_LL':overall['effects']['coupled_trained_delta_LL_independent_minus_coupled'],
                          'DID_LL':overall['effects']['difference_in_differences_LL'], 'block_deltas':deltas}), flush=True)
    bg_rows = json.loads((ROOT/'background-source-features.json').read_text())['rows']
    bg_train = [r for r in bg_rows if r['split']=='fit']
    mean, scale = normalizer(bg_train)
    bg_model = fit(bg_train,mean,scale,label='label')
    bg_metrics = metrics([r for r in bg_rows if r['split']=='heldout'],bg_model,label='label')
    neutral = [dict(r, features=r['neutral_foreground_features']) for r in cases]
    mean, scale = normalizer([r for r in neutral if r['split']=='fit'])
    fg_model = fit([r for r in neutral if r['split']=='fit'],mean,scale)
    fg_metrics = metrics([r for r in neutral if r['split']=='heldout'],fg_model)
    saved_models['background_C'] = bg_model
    saved_models['neutral_foreground_Y'] = fg_model
    out = dict(status='completed_small_RGB_mechanism_transfer_pilot', dataset='Oxford-IIIT Pet',
               real_source_photographs=64, species_classes=2, independent_heldout_macroblocks=4,
               primary_layout='protected', background_C_readout=bg_metrics, neutral_foreground_Y_readout=fg_metrics,
               layouts=result, model_input='RGB-derived features only, no T/C/mask/path',
               fixed_model='64 standardized/clipped spatial RGB/gradient features with products; centroid distance logits divided by 2144',
               normalized_using='coupled training images only; both policy models share this normalizer within each layout',
               geometry_cases=dict(total=len(cases),cropped=sum(r['cropped'] for r in cases),
                                   no_visible_foreground=sum(r['visible_alpha_fraction']==0 for r in cases)),
               wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,
               CPU_only=True, ARIS=False, GPU=False, new_dependencies=False,
               interpretation_limits=['Mechanism migration, not ForNet reconstruction or evaluation.',
                                      'Protected renderer adds a fixed neutral guard, not part of original ForAug.',
                                      'Four heldout macroblocks; no p-value over generated frames or proof of generalization to unseen animals.',
                                      'Fixed weak readout; a null result does not rule out other models.',
                                      'Positive screen is exploratory candidate evidence, not originality or an established contribution.'])
    save('pilot-results.json',out)
    save('readout-models.json',saved_models)
    save('readout-execution.json',dict(status='completed', input_rows=len(rows), models_fit=6,
                                      feature_rows_sha256=hashlib.sha256((ROOT/'rgb-features.jsonl').read_bytes()).hexdigest(),
                                      code_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
                                      no_actual_T_C_or_path_passed_to_prediction=True,
                                      heldout_results_used_for_hyperparameters=False,
                                      no_neural_network_training=True, source_photograph_unit=True))
    print(json.dumps({'status':out['status'],'background_C':bg_metrics,'foreground_Y':fg_metrics,
                       'wall_seconds':out['wall_seconds'],'cpu_seconds':out['cpu_seconds']},indent=2))


if __name__ == '__main__': main()
