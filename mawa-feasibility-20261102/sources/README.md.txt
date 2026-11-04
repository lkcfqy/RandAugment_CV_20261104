# Post-Training Augmentation Invariance

Code for Post-Training Augmentation Invariance, published in *Transactions on Machine Learning Research* (May 2026).

We introduce a framework for post-training augmentation invariance, in which our goal is to add invariance properties to a pretrained network without altering its behavior on the original, non-augmented input distribution. Concretely, we freeze the pretrained network $F$ and train a lightweight, one-hidden-layer MLP adapter $E_\theta$ on its latent space such that (i) $E_\theta \circ F$ is approximately invariant to the chosen augmentation, and (ii) $E_\theta$ acts nearly isometrically on the non-augmented latent distribution, ensuring that downstream performance on clean data is preserved.

We introduce two losses for this:

- **Markov-Wasserstein (MaWa) minimization:** essentially an anchored MSE loss 
  that pulls augmented features toward their clean counterparts. This is the strongest method when the pretrained latent space has low collision rate (see Def. 3.2).

- **Wasserstein correlation (WaCo) maximization:** (sliced) Wasserstein 
  correlation maximization on the joint distribution induced by the 
  augmented encoder. Competitive with MaWa and additionally supports 
  dimensionality reduction. Also more robust on feature spaces with 
  high collision rates.

Example results: on STL10 with $F=\text{DINO}$ features, the composite network $C\circ E_{\theta}\circ F$, where $C$ is a linear classifier and where $E_{\theta}$ is an adapter trained with the MaWa loss, achieves 94\% classification accuracy on arbitrarily rotated images, whereas a network of the form $C\circ F$ without the adapter $E_{\theta}$ drops to 71\% accuracy. Similarly, we can boost noise-invariant classification results from 58\% up to 86\%.

## Installation

```bash
git clone https://github.com/keenan-eikenberry/augmentation_invariance.git
cd augmentation_invariance
pip install -r requirements.txt
```

Tested with Python 3.10.12 on Linux with CUDA 11.8 

## Quick start

`main.py` runs a single training configuration end-to-end. Use `experiments` list to override template config with specific loss, augmentations, pretrained feature network, etc. 

Set `root_dir` at the top of `main.py` to point at where you want 
checkpoints, logs, and evaluation outputs written.

## Citation

```bibtex
@article{eikenberry2026posttraining,
  title   = {Post-Training Augmentation Invariance},
  author  = {Eikenberry, Keenan and Liu, Lizuo and Lee, Yoonsang},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  url     = {https://openreview.net/forum?id=Z4uUwU6zRe}, 
  note    = {Reviewed on OpenReview}
}
```

## License

MIT — see [LICENSE](LICENSE).
