# Prior-work boundary

[VEST: Enabling Validation for Robust Few-Shot Recognition](https://arxiv.org/html/2506.04713v4), Wang, Liu and Kong, arXiv:2506.04713, version4 dated2026-10-14, was read on2026-11-02. Sections3.2–3.4 explicitly address checkpoint and layer selection. The method combines performance on few-shot ID data and retrieved external OOD data within a multi-objective validation strategy, used in stage-wise CLIP visual-encoder fine-tuning. Thus lack of validation, model selection and ID/OOD trade-offs are established prior topics, not our first discovery.

The current proposed experiment uses existing frozen DINOv2/DeiT features, a fixed ridge head, no external retrieved data, and four-fold CV to choose among three pixel augmentation policies. It neither implements nor fairly compares against VEST: it has different models, supervision/resources and selection objects. No VEST retrieval, code, weights or external dataset is introduced. Cross-dataset replication of this limited opportunity/cost pattern alone would not establish an algorithmic or paper contribution.

The earlier project review of Evaluating the Evaluators, CLAP, FroFA and PBA remains in [the first pilot's related-work note](../augmentation-selection-pilot-20261102/RELATED_WORK.md). Existing findings cannot be relabelled as first discoveries of validation noise, label cost or training-scale mismatch.
