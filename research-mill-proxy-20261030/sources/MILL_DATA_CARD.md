---
license: mit
task_categories:
- image-to-image
tags:
- lowlight
pretty_name: M
---

# Evaluating Low-Light Image Enhancement Across Multiple Intensity Levels
# CVPR Findings 2026

[Project Page](https://color.cvc.uab.cat/mill/) - [Arxiv](https://arxiv.org/abs/2511.15496) - [Code](https://github.com/CVC-Color/MILL)

## MILL Dataset
The MILL dataset can be downloaded from this repository or directly from the following links.

| Split | Camera | Resolution | Download Link |
|----------|----------|----------|----------|
| MILLs-DSLR    | Nikon D5200     | 600x400     | [Link](https://datasets.cvc.uab.cat/color/MILLs-DSLR.zip) (196MB)    |
| MILLs-smartphone    | Samsung Galaxy S7     | 600x400     | [Link](https://datasets.cvc.uab.cat/color/MILLs-smartphone.zip) (227MB)   |
| MILLf-DSLR    | Nikon D5200     | 2012x1340     | [Link](https://datasets.cvc.uab.cat/color/MILLf-DSLR.zip) (8.4GB)    |
| MILLf-smartphone    | Samsung Galaxy S7     | 1560x1040     | [Link](https://datasets.cvc.uab.cat/color/MILLf-smartphone.zip) (915MB)   |

The MILLs splits are the low-resolution splits of the MILL dataset, while the MILLf are the high-resolution ones.

### MILL-Small
Both MILLs images are resized to 600x400 and follow the following data structure:

```
MILLs-{split}
|- 📁 train
|  |- 📁 input
|  |  |- 🖼️ Scene1_a-1.png
|  |  |- 🖼️ Scene1_a-2.png
|  |  |         ...
|  |  |- 🖼️ Scene1_a-10.png
|  |  |- 🖼️ Scene1_b-1.png
|  |  |- 🖼️ Scene2_a-7.png
|  |  |- 🖼️ Scene9-4.png
|  |  
|  |- 📁 gt
|     |- 🖼️ Scene1_a.png
|     |- 🖼️ Scene1_b.png
|     |- 🖼️ Scene2_a.png
|     |- 🖼️ Scene9.png
|
|- 📁 validation
|       ...
|- 📁 test
      ...
```
The first part of the filename defines the scene name while the second part of the input filenames defines the intensity level (from 1 to 10).

### MILLf-DSLR
The full resolution DSLR split crops the 6036x4020 original images into 9 2012x1340 images. The filenames have the grid position (row-column id) in between the scene name and intensity level.

```
MILLf-DSLR
|- 📁 train
|  |- 📁 input
|  |  |- 🖼️ Scene1_a-00-1.png
|  |  |- 🖼️ Scene1_a-01-1.png
|  |  |         ...
|  |  |- 🖼️ Scene1_a-22-1.png
|  |  |- 🖼️ Scene1_a-00-2.png
|  |  |         ...
|  |  |- 🖼️ Scene1_a-00-10.png
|  |  |- 🖼️ Scene1_b-22-1.png
|  |  |- 🖼️ Scene2_a-12-7.png
|  |  |- 🖼️ Scene9-02-4.png
|  |  
|  |- 📁 gt
|     |- 🖼️ Scene1_a-00.png
|     |- 🖼️ Scene1_a-01.png
|     |- 🖼️ Scene1_a-22.png
|     |- 🖼️ Scene1_b-22.png
|     |- 🖼️ Scene2_a-12.png
|     |- 🖼️ Scene9-02.png
|
|- 📁 validation
|       ...
|- 📁 test
      ...
```

### MILLf-smartphone
It follows the same structure as the MILLs splits. The images are the original ones captured by the Samsung Galaxy S7 device.

## Results
We benchmark several state-of-the-art low-light enhancement methods on the MILL dataset and reveal significant performance variations across different intensity levels. Our proposed improvements achieve up to 10 dB PSNR improvement for DSLR and 2 dB for smartphone images on Full HD resolution. For detailed results, please refer to our paper.

## Citation
If you use the MILL dataset or our methods in your research, please cite our paper:

```bibtex
@inproceedings{pilligua2026mill,
  title={MILL: Evaluating Low-Light Image Enhancement Across Multiple Intensity Levels},
  author={Pilligua, Maria and Serrano-Lozano, David and Peng, Pai and Baldrich, Ramon and Brown, Michael S. and Vazquez-Corral, Javier},
  booktitle={CVPR Findings},
  year={2026}
}
```