# IDGuardian

Official implementation of **No Way To Steal My Face: Proactive Defense Against Identity-Preserving Personalized Generation**.

IDGuardian protects a reference image before it is shared with an identity-preserving image generator. It optimizes an imperceptible perturbation with the same two signals described in the paper:

1. **Cross-Encoder Identity Field Confounding:** minimize the cosine similarity between clean and protected image features from CLIP and FaceNet.
2. **Guidance-Flow Identity Deflection:** compare IP-Adapter SDXL Plus Face denoising predictions conditioned on clean versus protected IP-Adapter image tokens, then use the score difference as the second PGD direction.

The repository contains the actual optimization entry point only. Generation and identity-metric evaluation are intentionally left to the downstream personalization/evaluation pipelines.

## Installation

```bash
conda create -n idguardian python=3.10 -y
conda activate idguardian
pip install -r requirements.txt
```

## Download the required weights

The optimizer uses the public IP-Adapter weights and an SDXL base checkpoint. The IP-Adapter repository is not bundled into this repository and the large checkpoints are not committed to Git.

```bash
mkdir -p checkpoints/ip_adapter

hf download h94/IP-Adapter \\
  --include "models/image_encoder/*" \\
  --local-dir checkpoints/ip_adapter

hf download h94/IP-Adapter \\
  --include "sdxl_models/ip-adapter-plus-face_sdxl_vit-h.bin" \\
  --local-dir checkpoints/ip_adapter
```

The default paths are then:

```text
checkpoints/ip_adapter/models/image_encoder
checkpoints/ip_adapter/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.bin
```

`stabilityai/stable-diffusion-xl-base-1.0` is loaded through Diffusers. To run offline, download it first and pass its local directory to `--pretrained-model-name-or-path`.

The FaceNet checkpoint used by `facenet-pytorch` is downloaded automatically on first use (`InceptionResnetV1(pretrained="vggface2")`).

## Run IDGuardian

For the main 224 x 224 setting in the paper:

```bash
python scripts/IDGuardian_train.py \\
  --input path/to/image_or_directory \\
  --output-dir output/idguardian_224 \\
  --image-size 224
```

For the 512 x 512 cross-dataset setting:

```bash
python scripts/IDGuardian_train.py \\
  --input path/to/image_or_directory \\
  --output-dir output/idguardian_512 \\
  --image-size 512
```

The default optimization configuration matches the paper: prompt `A photo of a person`, `epsilon=8/255`, `alpha=0.005`, and 200 PGD steps. All settings are exposed as command-line arguments.

Useful options:

```text
--device cuda:0
--steps 200
--epsilon 0.031372549
--alpha 0.005
--max-images 10
--seed 42
```

Each protected image is saved under `--output-dir`. The accompanying `results.jsonl` records the input, output, optimization settings, and pixel PSNR.

## Implementation notes

- The clean latent is encoded once and reused across iterations.
- The clean and protected IP-Adapter image tokens are concatenated with the real SDXL text embeddings before the UNet call.
- The identity-loss CLIP embedding and the IP-Adapter conditioning tokens are kept as separate quantities; they are not interchangeable.
- The score direction is computed with the same noisy latent and timestep for clean and protected conditions, then bilinearly upsampled to RGB resolution.
- The code supports both 224 and 512 without separate duplicated scripts.

## Citation

```bibtex
@inproceedings{xiong2026idguardian,
  title={No Way To Steal My Face: Proactive Defense Against Identity-Preserving Personalized Generation},
  author={Xiong, Lizhi and Li, Jun and Li, Ziqiang and Jiang, Weiwei and Fu, Zhangjie},
  booktitle={CVPR},
  year={2026}
}
```
