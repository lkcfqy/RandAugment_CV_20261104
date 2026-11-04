"""CountGD++ on the Hugging Face Hub.

    from countgdpp import CountGDPlusPlus, count
    model = CountGDPlusPlus.from_pretrained("nikigoli/CountGDPlusPlus").to("cuda").eval()
    result = count(model, "img/336.jpg", "strawberry", negative_texts=["blueberry"])
    print(result["count"])

`CountGDPlusPlus` is the CountGD++ GroundingDINO model with Hugging Face's
PyTorchModelHubMixin added, so `from_pretrained`, `save_pretrained` and
`push_to_hub` work. Its state dict keys are identical to the original
`countgd_plusplus.pth` checkpoint.
"""
import os
import sys
from types import SimpleNamespace
from typing import List, Optional, Sequence, Union

import torch
from PIL import Image
from huggingface_hub import PyTorchModelHubMixin

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from util.slconfig import SLConfig  # noqa: E402
from util.misc import nested_tensor_from_tensor_list  # noqa: E402
import datasets.transforms_app as T  # noqa: E402
from models.GroundingDINO.groundingdino_app import GroundingDINO  # noqa: E402
from models.GroundingDINO.backbone import build_backbone  # noqa: E402
from models.GroundingDINO.transformer import build_transformer  # noqa: E402

# Default score threshold used in the paper's demo.
CONF_THRESH = 0.23
# BERT token id of "." — separates the positive prompt from negative prompts in the caption.
_DOT_TOKEN_ID = 1012
_DROP_KEYS = {"label_list", "val_label_list"}


def default_config() -> dict:
    """Model configuration read from cfg_app.py (the config used by the CountGD++ demo)."""
    cfg = SLConfig.fromfile(os.path.join(_HERE, "cfg_app.py"))._cfg_dict.to_dict()
    cfg = {k: v for k, v in cfg.items() if k not in _DROP_KEYS}
    cfg["text_encoder_type"] = "bert-base-uncased"  # fetched from the Hub by transformers
    cfg["device"] = "cpu"
    return cfg


class CountGDPlusPlus(
    GroundingDINO,
    PyTorchModelHubMixin,
    library_name="pytorch",
    repo_url="https://github.com/niki-amini-naieni/CountGDPlusPlus",
    license="mit",
    tags=["CountGD++", "computer-vision", "counting", "grounding-dino", "multi-modal", "open-vocabulary"],
):
    """CountGD++ model. Build with `CountGDPlusPlus.from_pretrained(repo_id)`."""

    def __init__(self, config: Optional[dict] = None):
        config = dict(config or default_config())
        args = SimpleNamespace(**config)
        backbone = build_backbone(args)
        transformer = build_transformer(args)
        # Same arguments as build_groundingdino() in groundingdino_app.py
        super().__init__(
            backbone,
            transformer,
            num_queries=args.num_queries,
            aux_loss=args.aux_loss,
            iter_update=True,
            query_dim=4,
            num_feature_levels=args.num_feature_levels,
            nheads=args.nheads,
            dec_pred_bbox_embed_share=args.dec_pred_bbox_embed_share,
            two_stage_type=args.two_stage_type,
            two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
            two_stage_class_embed_share=args.two_stage_class_embed_share,
            num_patterns=args.num_patterns,
            dn_number=0,
            dn_box_noise_scale=args.dn_box_noise_scale,
            dn_label_noise_ratio=args.dn_label_noise_ratio,
            dn_labelbook_size=args.dn_labelbook_size,
            text_encoder_type=args.text_encoder_type,
            sub_sentence_present=args.sub_sentence_present,
            max_text_len=args.max_text_len,
        )
        self.countgd_config = config


# ----------------------------------------------------------------------------
# Inference (same pre/post-processing as the CountGD++ demo app)
# ----------------------------------------------------------------------------
def build_transform():
    normalize = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return T.Compose([T.RandomResize([800], max_size=1333), normalize])


def _load(image: Union[str, Image.Image]) -> Image.Image:
    return (Image.open(image) if isinstance(image, str) else image).convert("RGB")


def _prep_exemplars(transform, image, boxes):
    boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.tensor([])
    img, target = transform(image, {"exemplars": boxes})
    return img, target["exemplars"]


@torch.no_grad()
def count(
    model: CountGDPlusPlus,
    image: Union[str, Image.Image],
    text: str,
    exemplar_boxes: Optional[Sequence[Sequence[float]]] = None,
    exemplar_image: Optional[Union[str, Image.Image]] = None,
    negative_texts: Optional[List[str]] = None,
    negative_exemplar_boxes: Optional[List[Sequence[Sequence[float]]]] = None,
    negative_exemplar_images: Optional[List[Union[str, Image.Image]]] = None,
    conf_thresh: float = CONF_THRESH,
):
    """Count objects in `image`.

    text: what to count, e.g. "strawberry". Use "" for exemplars only.
    exemplar_boxes: optional visual exemplars [[x1, y1, x2, y2], ...] in pixels,
        drawn on `exemplar_image` (defaults to `image`).
    negative_texts: optional things NOT to count, e.g. ["blueberry"]. Use "" for
        a negative given by exemplars only. If omitted while negative exemplars
        are given, each negative gets "" as its text.
    negative_exemplar_boxes / negative_exemplar_images: optional, one entry per
        negative, same format as the positive ones.

    Returns dict with "count", "boxes" (normalized cx, cy, w, h) and "scores".
    """
    if not text and not exemplar_boxes:
        raise ValueError("Give a positive text, positive exemplar boxes, or both.")
    device = next(model.parameters()).device
    transform = build_transform()
    image = _load(image)
    pos_img = _load(exemplar_image) if exemplar_image is not None else image
    if negative_texts is None and negative_exemplar_boxes:
        negative_texts = [""] * len(negative_exemplar_boxes)
    negative_texts = list(negative_texts or [])
    n_neg = len(negative_texts)
    neg_boxes = list(negative_exemplar_boxes or [None] * n_neg)
    neg_imgs = list(negative_exemplar_images or [None] * n_neg)
    if len(neg_boxes) != n_neg or len(neg_imgs) != n_neg:
        raise ValueError("Give one negative exemplar entry (or None) per negative text.")

    input_image, _ = transform(image, None)
    pos_exemplar_image, pos_exemplars = _prep_exemplars(transform, pos_img, exemplar_boxes)

    neg_images, neg_exemplars = [], []
    for boxes, img in zip(neg_boxes, neg_imgs):
        img = _load(img) if img is not None else image
        ex_img, ex = _prep_exemplars(transform, img, boxes)
        neg_images.append(nested_tensor_from_tensor_list(ex_img.unsqueeze(0).to(device)))
        neg_exemplars.append([ex.to(device)])

    caption = text + " . " + "".join(t + " . " for t in negative_texts)
    out = model(
        nested_tensor_from_tensor_list(input_image.unsqueeze(0).to(device)),
        nested_tensor_from_tensor_list(pos_exemplar_image.unsqueeze(0).to(device)),
        [pos_exemplars.to(device)],
        neg_images,
        neg_exemplars,
        captions=[caption],
    )

    input_ids = out["input_ids"][0]
    logits = out["pred_logits"].sigmoid()[0]
    boxes = out["pred_boxes"][0]
    split = next(i for i, t in enumerate(input_ids) if int(t) == _DOT_TOKEN_ID)
    pos_logits, neg_logits = logits[:, : split + 1], logits[:, split + 1 :]

    # Stage 1: keep queries confident about the positive prompt.
    keep = pos_logits.max(dim=-1).values > conf_thresh
    boxes, scores = boxes[keep], logits[keep].max(dim=-1).values
    pos_logits, neg_logits = pos_logits[keep], neg_logits[keep]
    # Stage 2: drop queries that match a negative prompt better than the positive one.
    if neg_logits.shape[-1] > 0:
        keep = pos_logits.max(dim=-1).values > neg_logits.max(dim=-1).values
        boxes, scores = boxes[keep], scores[keep]

    return {"count": int(boxes.shape[0]), "boxes": boxes.cpu().numpy(), "scores": scores.cpu().numpy()}
