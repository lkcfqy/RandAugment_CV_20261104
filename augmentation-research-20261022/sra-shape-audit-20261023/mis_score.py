"""Local SRA-score adapter with explicit target-shape validation.

Implements the existing cosine score, not a new augmentation method.
No network, model forward, RNG, image processing, or training occurs here.
Set class_exponent=False only for callers whose existing formula is unpowered.
"""
import math
import torch
from torch.nn import functional as F

@torch.no_grad()
def mis_score(logits, targets, *, class_exponent=True):
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
        raise ValueError('logits must have shape (B,C)')
    batch, classes = logits.shape
    if batch < 1 or classes < 2 or not logits.is_floating_point():
        raise ValueError('logits require B>=1, C>=2 and a floating dtype')
    if not torch.isfinite(logits).all():
        raise ValueError('logits must be finite')
    if not isinstance(targets, torch.Tensor) or targets.device != logits.device:
        raise ValueError('targets must be a Tensor on the logits device')
    if targets.ndim == 1:
        if targets.shape[0] != batch or targets.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError('hard labels must be integral with shape (B,)')
        if torch.any(targets < 0) or torch.any(targets >= classes):
            raise ValueError('hard label outside [0,C)')
        target_matrix = F.one_hot(targets.to(torch.int64), num_classes=classes).to(logits.dtype)
    elif targets.ndim == 2 and tuple(targets.shape) == tuple(logits.shape):
        if targets.is_complex():
            raise ValueError('complex targets are unsupported')
        target_matrix = targets.to(logits.dtype)
        if not torch.isfinite(target_matrix).all() or torch.any(target_matrix < 0):
            raise ValueError('soft targets must be finite and nonnegative')
        if not torch.allclose(target_matrix.sum(-1), torch.ones(batch, dtype=logits.dtype, device=logits.device), atol=1e-5, rtol=0):
            raise ValueError('soft target rows must sum to one')
    else:
        raise ValueError('targets must have shape (B,) or exactly (B,C)')
    score = F.cosine_similarity(F.softmax(logits.detach(), dim=-1), target_matrix, dim=-1)
    return score ** (2.0 / math.log(classes)) if class_exponent else score
