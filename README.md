# Autocolorizer — Basic Automatic Image Colorization

> Hybrid U-Net + Zhang Classification approach to automatic black-and-white photo colorization.

**Authors:** Malwina Belczewska & Oliwier Mróz

> *Note: This notebook and its documentation were originally written in Polish and later translated into English. A few Polish strings may still remain in the code (e.g. console messages).*

------------------------------------------------------------------------

## Overview

This project implements an AI system for automatic, photorealistic colorization of grayscale images. Rather than treating colorization as a regression problem (which tends to produce desaturated, "safe" brownish results), we implement the **Zhang 2016 classification approach** combined with a **hybrid U-Net architecture** and a **PatchGAN discriminator**.

The core idea: instead of predicting exact color values, the network classifies each pixel into one of **313 discrete color bins** in CIELab space — producing vivid, confident colorizations.

------------------------------------------------------------------------

## Architecture

### Generator — Hybrid U-Net

| Component   | Details                                              |
|-------------|------------------------------------------------------|
| **Encoder** | EfficientNet-B4 (pretrained on ImageNet)             |
| **Decoder** | U-Net with skip connections                          |
| **Input**   | Grayscale L channel, triplicated → `(256 × 256 × 3)` |
| **Output**  | Per-pixel distribution over 313 Zhang color bins     |

The L channel is triplicated so the pretrained EfficientNet-B4 encoder can be used without modification (it expects 3-channel input). Skip connections preserve fine spatial detail and prevent color bleeding across edges. The architecture is built on top of `segmentation_models_pytorch` (`smp.Unet`).

### Discriminator — PatchGAN

A PatchGAN discriminator evaluates local `70×70` patches rather than the full image, focusing on whether local textures and color transitions look realistic. It outputs a `30×30` judgment map, uses **spectral normalization** + **InstanceNorm + LeakyReLU(0.2)**, and is trained as an **LSGAN** (MSE loss instead of BCE). Its input is the concatenation of the `L` and `ab` channels.

### Loss Function (Generator)

```         
L_total = λ_zhang · L_zhang + λ_l1 · L_l1 + λ_adv · L_adv
```

-   **Zhang rebalanced CrossEntropy** — main classification loss; class weights boost rare, vivid colors (used in all phases)
-   **L1 loss** — pixel-level consistency against the ground-truth ab channels (GAN phase only)
-   **Adversarial loss** — GAN signal from the PatchGAN discriminator (active only after warmup)

> Note: the pre-training phase uses the Zhang rebalanced CrossEntropy **alone**. The L1 and adversarial terms are introduced only in the GAN phase.

------------------------------------------------------------------------

## Training Pipeline

### Phase 1 — U-Net Pre-training (10 epochs)

Train the generator alone with the **Zhang rebalanced CrossEntropy** loss. This establishes a solid colorization baseline before adversarial training. Uses gradient accumulation (every 4 steps), mixed precision (AMP), early stopping (patience = 7), and a `ReduceLROnPlateau` scheduler (×0.5 after 3 epochs without improvement).

### Phase 2 — GAN Training (15 epochs, warmup 5 epochs)

During the first 5 epochs (warmup), only the generator trains — giving it time to stabilize before the discriminator starts providing feedback. After warmup, both networks train alternately each iteration (LSGAN).

### Phase 3 — Final Fine-Tuning (resume from `gan_best`, no warmup)

The first GAN run did not improve over the pre-trained baseline, so training is resumed from the best checkpoint with a corrected loss balance: a lower `lambda_l1`, a stronger `lambda_adv`, and a lower generator learning rate for more stable convergence. Warmup is skipped because the generator is already trained.

**Key hyperparameters:**

| Parameter               | Phase 2 — GAN run | Phase 3 — Fine-tuning |
|-------------------------|-------------------|-----------------------|
| Epochs                  | 15                | 10                    |
| Warmup epochs           | 5                 | 0                     |
| `lambda_zhang`          | 1.0               | 1.0                   |
| `lambda_l1`             | 10.0              | 2.0                   |
| `lambda_adv`            | 0.1               | 0.2                   |
| `lr_G`                  | 2e-4              | 1e-4                  |
| `lr_D`                  | 5e-5              | 5e-5                  |
| Early stopping patience | 7                 | 7                     |

------------------------------------------------------------------------

## Dataset

[COCO 2017](https://cocodataset.org/) — train/val/test splits (≈30 000 train / 1 000 val images per loader).

Images are converted to CIELab color space. The L channel serves as network input; the ab channels are quantized to the 313 Zhang bins as training targets. Images with very low color variance (`std(ab) < 0.5`) are filtered out so that near-grayscale photos do not dominate training.

------------------------------------------------------------------------

## Results

The pre-trained U-Net produces semantically plausible colorizations for the majority of scenes. **Neither GAN phase yielded a consistent metric improvement over the pre-trained baseline** — in both runs the validation loss (Zhang + L1) failed to improve after a few epochs, triggering early stopping. The best checkpoint originated from the **warmup phase of the first GAN run (epoch 3, Val = 4.4753)**, before the discriminator started training; from epoch 6 onward the validation loss rose steadily (4.52 → 4.56 → 4.66).

The likely cause was an overly strong L1 penalty (`lambda_l1 = 10.0`) that suppressed the adversarial signal. The fine-tuning phase corrected this (`lambda_l1 = 2.0`, `lambda_adv = 0.2`) but still did not produce a measurable improvement.

Overall, balancing the L1 and adversarial signals remains the main open challenge — a known difficulty in conditional image generation. Further gains would likely require longer GAN training, a larger dataset, or a more global encoder architecture (e.g. a Swin Transformer).

------------------------------------------------------------------------

## Project Structure

```         
autocolorizer/
├── Autocolorizer_EN.ipynb     # Main training notebook (pre-training + GAN + fine-tuning), translated from Polish
├── README.md
└── .gitignore
```

> **Model weights** are not included in this repository due to file size (\~80–270 MB each).\
> Pretrained checkpoints available on Google Drive: [link]\
> *(replace with your actual Drive link)*

------------------------------------------------------------------------

## Setup & Usage

This project was developed in Google Colab. Open the notebook directly:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1dR0Qc7AZMdR7VJKRcWFlQZmqNNYJ7otj/view?usp=sharing)

### Dependencies

``` bash
pip install torch torchvision segmentation-models-pytorch
pip install scikit-image Pillow tqdm matplotlib numpy
```

The Zhang quantization data (`pts_in_hull.npy`, `prior_probs.npy`) is downloaded automatically from the original repository when the notebook runs.

------------------------------------------------------------------------

## References

-   Zhang, R., Isola, P., & Efros, A. A. (2016). **Colorful Image Colorization**. *ECCV 2016*. [[paper]](https://arxiv.org/abs/1603.08511) [[code]](https://github.com/richzhang/colorization)
-   Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A. (2017). **Image-to-Image Translation with Conditional Adversarial Networks**. *CVPR 2017*. [[paper]](https://arxiv.org/abs/1611.07004)

Color quantization data (`pts_in_hull.npy`, `prior_probs.npy`) sourced from [richzhang/colorization](https://github.com/richzhang/colorization) under BSD-2-Clause license.

------------------------------------------------------------------------

## License

MIT License — see [LICENSE](LICENSE) for details.
