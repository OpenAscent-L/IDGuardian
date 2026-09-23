# IDGuardian

Official implementation of:

**No Way To Steal My Face: Proactive Defense Against Identity-Preserving Personalized Generation**

IDGuardian is a proactive adversarial defense framework designed to protect facial identity against identity-preserving personalized generation models.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
python scripts/train_IDGuardian.py --image_size 224
```

## Pretrained Models

Please check `checkpoints/README.md` for required model weights.

## Citation

```bibtex
@inproceedings{IDGuardian,
  title={No Way To Steal My Face: Proactive Defense Against Identity-Preserving Personalized Generation},
  booktitle={CVPR},
  year={2026}
}
```
