---
license: mit
language:
- en
library_name: pytorch
pipeline_tag: zero-shot-object-detection
tags:
- CountGD++
- computer-vision
- counting
- grounding-dino
- model_hub_mixin
- multi-modal
- open-vocabulary
- pytorch_model_hub_mixin
---

# CountGD++

**CountGD++: Generalized Prompting for Open-World Counting** (CVPR 2026)

A multi-modal open-world counting model. You say what to count with text,
visual exemplars (boxes on example objects), or both. You can also say what
**not** to count with negative text and negative exemplars. Exemplars can come
from the image being counted or from a different image.

[Paper](https://arxiv.org/abs/2512.23351) · [Code](https://github.com/niki-amini-naieni/CountGDPlusPlus)

![teaser](img/teaser.jpg)

This repository holds both the **weights** (`model.safetensors`, `config.json`)
and the **inference code**, so nothing has to be downloaded from Google Drive.

## Usage

```bash
pip install torch torchvision "transformers<4.50" timm addict "yapf==0.40.1" termcolor \
    scipy matplotlib opencv-python-headless pycocotools pyyaml huggingface_hub safetensors
```

```python
import sys
from huggingface_hub import snapshot_download

repo = snapshot_download("nikigoli/CountGDPlusPlus")   # code + weights
sys.path.insert(0, repo)
from countgdpp import CountGDPlusPlus, count

model = CountGDPlusPlus.from_pretrained("nikigoli/CountGDPlusPlus").to("cuda").eval()

# Text only
print(count(model, f"{repo}/img/336.jpg", "strawberry")["count"])

# Positive text + negative text
print(count(model, f"{repo}/img/336.jpg", "strawberry", negative_texts=["blueberry"])["count"])

# Text + visual exemplars drawn on another image, with a negative exemplar
result = count(
    model, f"{repo}/img/336.jpg", "strawberry",
    exemplar_boxes=[[x1, y1, x2, y2]],                    # pixels, on exemplar_image
    exemplar_image=f"{repo}/img/synthetic_strawberry.jpg",
    negative_texts=["blueberry"],
    negative_exemplar_boxes=[[[x1, y1, x2, y2]]],          # one list per negative text
    negative_exemplar_images=[f"{repo}/img/synthetic_blueberry.jpg"],
)
print(result["count"], result["boxes"])   # boxes: normalized (cx, cy, w, h)
```

`example.py` is a runnable version that also saves a picture of the detections.

Notes:
- The text encoder (`bert-base-uncased`) is fetched from the Hugging Face Hub
  by `transformers` the first time the model loads.
- The compiled multi-scale deformable attention kernels are optional. Without
  them the model uses an equivalent pure-PyTorch implementation on CPU or GPU.
  To compile the kernels, see the GitHub repository.
- Pseudo-exemplars and adaptive cropping used for some benchmark results are not
  part of `count()`. See the test scripts in the GitHub repository.

## Architecture

![architecture](img/inference-architecture.jpg)

## Citation

```bibtex
@InProceedings{AminiNaieni26b,
  title={CountGD++: Generalized Prompting for Open-World Counting},
  author={Amini-Naieni, N. and Zisserman, A.},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```
