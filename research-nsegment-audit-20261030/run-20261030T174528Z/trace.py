"""Conditional synthetic 16-sample CPU pipeline, explicitly not Table I config."""
import json
import time
import traceback
import numpy as np
from audit_core import *


def synthetic(sample_id):
    x, y = np.meshgrid(np.arange(512), np.arange(512))
    image = np.stack([(x + 3*sample_id) % 256, (y + 7*sample_id) % 256,
                      (x//3 + y//5 + 11*sample_id) % 256], axis=-1).astype(np.uint8)
    mask = ((x//13 + y//17 + sample_id) % 7).astype(np.uint8)
    return image, mask


def signature(a):
    return dict(dtype=str(a.dtype), shape=list(a.shape), c_contiguous=bool(a.flags.c_contiguous),
                owns_data=bool(a.flags.owndata), sha256=array_hash(a))


def first_difference(rows1, rows2, fields):
    for i, (a, b) in enumerate(zip(rows1, rows2)):
        for field in fields:
            if a[field] != b[field]:
                return dict(sample_position=i, sample_id_a=a['sample_id'], sample_id_b=b['sample_id'],
                            field=field, a=a[field], b=b[field])
    return None


def main():
    scan = json.loads((ROOT / 'scan-summary.json').read_text())
    fixture = json.loads((ROOT / 'fixture-results.json').read_text())
    eligible = (scan['status'] == 'complete' and scan['rows'] == 1024
                and fixture['status'] == 'pass' and fixture['positive_control_pass']
                and all(g['identity_seeds'] == 128 for k, g in scan['groups'].items() if 'a1_s10' in k)
                and all(g['identity_seeds'] == 0 for k, g in scan['groups'].items() if 'a100_s3' in k))
    if not eligible:
        save_json('trace-summary.json', dict(status='not_eligible', reason='Stage A conditional gate'))
        return
    started = time.time()
    deadline = guard('pipeline_trace', 600)
    checkpoint('trace_started', samples=16, synthetic=True, training=False, deadline_epoch=deadline)
    summary = dict(status='running', start_utc=utc(), samples=16, original_table_i_configuration=False,
                   scope='Constructed one-epoch synthetic pipeline; order -> operation -> crop -> flip',
                   input_shape=[512,512], crop_shape=[256,256], global_seed=20261030,
                   local_seed_rule='B RandomState(sample_id); C full before-state replay from B; independent of global RNG',
                   shuffle_rule='one np.random.permutation(16) before any sample operation')
    all_rows, states, outputs, b_local_records = {}, {}, {}, {}
    initial_global = np.random.RandomState(20261030).get_state()
    states['initial_global'] = state_record(initial_global)
    try:
        for variant in ['A', 'B', 'C']:
            np.random.set_state(initial_global)
            order = np.random.permutation(16).tolist()
            rows, variant_states, variant_outputs = [], [], []
            for i, sample_id in enumerate(order):
                image, mask = synthetic(sample_id)
                original_mask = mask
                row = dict(sample_position=i, sample_id=sample_id, variant=variant,
                           global_before_operation=state_hash(np.random.get_state()),
                           input_image=signature(image), input_mask=signature(mask))
                state_row = dict(sample_position=i, sample_id=sample_id,
                                 global_before_operation=state_record(np.random.get_state()))
                if variant == 'A':
                    row.update(applied=False, skipped=True, gate_value=None,
                               local_before=None, local_after_choice=None, local_after_field=None)
                else:
                    gate_value, applied = gate(1)
                    assert applied
                    row.update(applied=True, skipped=False, gate_value=gate_value)
                    if variant == 'B':
                        local = np.random.RandomState(sample_id)
                        b_local_records[i] = state_record(local.get_state())
                    else:
                        local = np.random.RandomState(0)
                        local.set_state(state_restore(b_local_records[i]))
                    state_row['local_before'] = state_record(local.get_state())
                    maps = make_maps((512,512), [(1,10)], local)
                    mask = remap(mask, maps, identity=(variant == 'C'))
                    state_row['local_after'] = state_record(local.get_state())
                    for key in ['local_before', 'local_after_choice', 'local_after_field']:
                        row[key] = maps[key]
                    row['unmodified_map_hash_x'] = array_hash(maps['map_x'])
                    row['unmodified_map_hash_y'] = array_hash(maps['map_y'])
                    del maps
                row['global_after_operation'] = state_hash(np.random.get_state())
                state_row['global_after_operation'] = state_record(np.random.get_state())
                row['post_operation_mask'] = signature(mask)
                row['mask_shares_input'] = bool(np.shares_memory(mask, original_mask))
                row['applied_identity'] = bool(row['applied'] and np.array_equal(mask, original_mask))
                assert np.array_equal(mask, original_mask), 'trace target not identity'
                top = int(np.random.randint(0, 512-256+1))
                left = int(np.random.randint(0, 512-256+1))
                do_flip = bool(np.random.rand() < 0.5)
                row['crop_top'], row['crop_left'], row['horizontal_flip'] = top, left, do_flip
                image = image[top:top+256, left:left+256]
                mask = mask[top:top+256, left:left+256]
                if do_flip:
                    image, mask = image[:, ::-1], mask[:, ::-1]
                image, mask = np.ascontiguousarray(image), np.ascontiguousarray(mask)
                row['final_image'], row['final_mask'] = signature(image), signature(mask)
                row['global_after_pipeline'] = state_hash(np.random.get_state())
                state_row['global_after_pipeline'] = state_record(np.random.get_state())
                rows.append(row); variant_states.append(state_row); variant_outputs.append((image,mask))
            all_rows[variant] = rows
            states[variant] = variant_states
            outputs[variant] = variant_outputs
            np.savez_compressed(ROOT / ('trace-outputs-' + variant + '.npz'),
                                image=np.stack([x[0] for x in variant_outputs]),
                                mask=np.stack([x[1] for x in variant_outputs]), sample_order=order)
            checkpoint('trace_variant_complete', variant=variant, samples=len(rows), peak_rss_bytes=peak_rss())

        pairs = {}
        for a,b in [('A','B'), ('A','C'), ('B','C')]:
            pairs[a+b] = dict(
                sample_order_equal=[r['sample_id'] for r in all_rows[a]] == [r['sample_id'] for r in all_rows[b]],
                global_rng_all_stages_equal=all(x[k] == y[k] for x,y in zip(all_rows[a],all_rows[b])
                    for k in ['global_before_operation','global_after_operation','global_after_pipeline']),
                post_operation_masks_equal=all(x['post_operation_mask'] == y['post_operation_mask']
                    for x,y in zip(all_rows[a],all_rows[b])),
                crop_flip_parameters_equal=sum(all(x[k] == y[k] for k in
                    ['crop_top','crop_left','horizontal_flip']) for x,y in zip(all_rows[a],all_rows[b])),
                final_images_equal=sum(np.array_equal(x[0],y[0]) for x,y in zip(outputs[a],outputs[b])),
                final_masks_equal=sum(np.array_equal(x[1],y[1]) for x,y in zip(outputs[a],outputs[b])),
                first_divergence=first_difference(all_rows[a],all_rows[b],
                    ['sample_id','global_before_operation','global_after_operation','post_operation_mask',
                     'crop_top','crop_left','horizontal_flip','final_image','final_mask','global_after_pipeline']))
        bc = pairs['BC']
        bc['local_states_equal'] = all(x['local_before'] == y['local_before'] and x['local_after'] == y['local_after']
                                      for x,y in zip(states['B'],states['C']))
        bc['unmodified_maps_equal'] = all(x['unmodified_map_hash_x'] == y['unmodified_map_hash_x'] and
                                         x['unmodified_map_hash_y'] == y['unmodified_map_hash_y']
                                         for x,y in zip(all_rows['B'],all_rows['C']))
        bc['dtype_contiguity_ownership_equal'] = all(x['post_operation_mask'] == y['post_operation_mask'] and
                                                     not x['mask_shares_input'] and not y['mask_shares_input']
                                                     for x,y in zip(all_rows['B'],all_rows['C']))
        assert bc['first_divergence'] is None and bc['local_states_equal'] and bc['unmodified_maps_equal']
        assert bc['dtype_contiguity_ownership_equal'] and bc['final_images_equal'] == 16 and bc['final_masks_equal'] == 16
        summary.update(status='complete', comparisons=pairs,
                       result='B=C; A differs in this constructed pipeline' if pairs['AB']['first_divergence'] else 'A=B=C')
        save_json('trace-records.json', all_rows)
        save_json('trace-states.json', states)
    except Exception as exc:
        summary.update(status='error', error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        summary.update(elapsed_seconds=time.time()-started, end_utc=utc(), peak_rss_bytes=peak_rss())
        save_json('trace-summary.json', summary)
        checkpoint('trace_finished', status=summary['status'], elapsed_seconds=summary['elapsed_seconds'],
                   peak_rss_bytes=peak_rss())


if __name__ == '__main__':
    main()
