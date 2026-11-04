"""Engineering diagnostics for native outputs; never serialize predictions."""
import torch

ATOL = 1e-4
RTOL = 1e-3


class NumericalFailure(RuntimeError):
    def __init__(self, message, report=None):
        super().__init__(message)
        self.report = report or {}


def validate(raw):
    logits = raw['pred_logits']; coordinates = raw['pred_boxes']
    token_ids = raw['input_ids']; mask = raw['text_mask']
    if any(x.device.type != 'cpu' for x in (logits, coordinates, token_ids, mask)):
        raise NumericalFailure('diagnostics require detached CPU outputs')
    if logits.ndim != 3 or coordinates.shape != (*logits.shape[:2], 4) or mask.shape != (logits.shape[0], logits.shape[2]):
        raise NumericalFailure('unexpected native output shape')
    expected = torch.zeros_like(mask)
    expected[:, :token_ids.shape[1]] = True
    if mask.dtype != torch.bool or not torch.equal(mask, expected):
        raise NumericalFailure('native valid-token mask changed')
    active = mask[:, None, :].expand_as(logits)
    report = {'active_logits_finite': bool(torch.isfinite(logits[active]).all()),
              'full_logits_finite': bool(torch.isfinite(logits).all()),
              'padding_exactly_negative_infinity': bool(torch.isneginf(logits[~active]).all()),
              'coordinates_finite': bool(torch.isfinite(coordinates).all()),
              'sigmoid_scores_finite': bool(torch.isfinite(logits.sigmoid()).all()),
              'logit_shape': list(logits.shape), 'coordinate_shape': list(coordinates.shape),
              'valid_token_length': token_ids.shape[1]}
    if not all(report[k] for k in ('active_logits_finite', 'padding_exactly_negative_infinity', 'coordinates_finite', 'sigmoid_scores_finite')):
        raise NumericalFailure('nonfinite active value or invalid native padding', report)
    return report


def native_mask(raw):
    # Exactly equivalent to native count(): confidence > .23, then positive > negative.
    ids = raw['input_ids'][0].tolist()
    split = ids.index(1012)
    scores = raw['pred_logits'].sigmoid()[0]
    pos = scores[:, :split+1].max(dim=-1).values
    neg = scores[:, split+1:].max(dim=-1).values
    return (pos > .23) & (pos > neg)


def compare(cpu, mps):
    validate(cpu); validate(mps)
    if not torch.equal(cpu['input_ids'], mps['input_ids']) or not torch.equal(cpu['text_mask'], mps['text_mask']):
        raise NumericalFailure('CPU/MPS token identities differ')
    active = cpu['text_mask'][:, None, :].expand_as(cpu['pred_logits'])
    report = {'absolute_tolerance': ATOL, 'relative_tolerance_against_cpu': RTOL, 'native_mask_equal': torch.equal(native_mask(cpu), native_mask(mps))}
    passes = []
    for key, label, selector in [('pred_logits', 'active_logits', active), ('pred_boxes', 'coordinates', None)]:
        reference = cpu[key] if selector is None else cpu[key][selector]
        actual = mps[key] if selector is None else mps[key][selector]
        if reference.shape != actual.shape:
            raise NumericalFailure('CPU/MPS tensor shape differs')
        difference = (actual-reference).abs()
        limit = ATOL+RTOL*reference.abs()
        okay = bool((difference <= limit).all())
        passes.append(okay)
        report[label] = {'max_absolute_error': float(difference.max()), 'max_error_divided_by_tolerance': float((difference/limit).max()), 'within_tolerance': okay}
    report['tolerance_passed'] = all(passes)
    if not report['tolerance_passed'] or not report['native_mask_equal']:
        raise NumericalFailure('CPU/MPS tolerance or native mask mismatch', report)
    return report
