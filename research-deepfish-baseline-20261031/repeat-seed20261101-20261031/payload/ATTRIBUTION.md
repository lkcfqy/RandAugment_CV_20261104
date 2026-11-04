# Sources and limits

DeepFish: Saleh et al., “A realistic fish-habitat dataset to evaluate algorithms for underwater visual analysis,” Scientific Reports 10, 14671 (2020), https://doi.org/10.1038/s41598-020-71639-x . Original author dataset card: https://huggingface.co/datasets/Alzayats/DeepFish , CC BY 4.0, https://creativecommons.org/licenses/by/4.0/ .

Published detection archive: CD-ViTO / CDFSOD-benchmark, https://github.com/lovelyqian/CDFSOD-benchmark/tree/929413d7d61caf72808a8c27cce19bf51c2973dd . Official FISH.tar.gz SHA256: locally measured `714ed9b731efab05ddfb4bd3b73127b67ebcb2a3a5f7efa07f66f314b9ad6094`. The benchmark archive contains no independent detection-label derivation or license explanation. The original source describes classification, points and segmentation; the detector-box creation chain remains unknown. No boxes were derived or changed here. The dev subset selects existing records only.

Official detector: OpenMMLab MMDetection v3.3.0, commit 44ebd17b145c2372c4b700bfb9cb20dbd28ab64a, Apache-2.0. Architecture and geometric configuration derive from its original GroundingDINO Swin-T configuration. https://github.com/open-mmlab/mmdetection/tree/44ebd17b145c2372c4b700bfb9cb20dbd28ab64a . Upstream license is included as LICENSE-MMDETECTION.

No claim of paper reproduction, novelty, augmentation benefit or independent-scene generalization is made by this preparation.
