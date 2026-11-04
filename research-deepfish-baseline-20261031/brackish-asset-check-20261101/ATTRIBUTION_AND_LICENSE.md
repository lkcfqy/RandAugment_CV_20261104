# Source attribution and unresolved license metadata

Dataset: Brackish, publisher Aalborg University, creator listed as Malte Pedersen; fixed Kaggle dataset version 4, published 2020-08-25. Retrieved 2026-11-01 through the public official entry points, without credentials.

- [AAU dataset page](https://vap.aau.dk/the-brackish-dataset/)
- [Author dataset publication](https://www.kaggle.com/datasets/aalborguniversity/brackish-dataset)
- [Public metadata](https://www.kaggle.com/api/v1/datasets/view/aalborguniversity/brackish-dataset)
- Publisher license label: [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/).

Requested research citation: Pedersen, Malte; Haurum, Joakim Bruslund; Gade, Rikke; Moeslund, Thomas B.; Madsen, Niels. *Detection of Marine Animals in a New Underwater Dataset with Varying Visibility*. CVPR Workshops, 2019. The v4 publisher notice distinguishes the corrected annotations from the older version used in that paper.

Preserve source attribution, license link, and indications of modifications; the publisher-selected license has a share-alike condition for shared adaptations. This note records the publisher's license and is not a customer deployment clearance. No data were redistributed externally in this task.

Unresolved discrepancy: the three released COCO JSON files contain `licenses: ["MIT"]`. The released `scripts/AnnotationsToCOCO.py` line 136 writes that value as a constant. No standalone LICENSE file was found in the archive. This provides provenance for the field but does not establish that the author intended to override the dataset publication's CC BY-SA 4.0 license. No author clarification was obtained, and the data have not been relabeled as MIT.

Archive SHA256: `95de9551dd3792e4c690b59ea8dfe9c4ab59dae4c7ad32d994bd6820a75630da`.

Original annotation and video bytes are unchanged. Local additions comprise manifests, file-name and annotation statistics, excluded-coverage ledgers, proposed date-pool membership tables, and research notes. No derivative model, frame images, replacement labels, or uploaded dataset was produced.
