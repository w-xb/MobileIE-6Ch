# Reproducibility Notes

This document records the practical details needed to reproduce the MobileIE-6Ch submission workflow.

## Environment

Recommended environment:

```bash
conda create -n mobileie python=3.10 -y
conda activate mobileie
pip install -r requirements.txt
```

The project was designed for PyTorch with CUDA. If your CUDA runtime requires a specific PyTorch wheel, install the correct PyTorch build before installing the rest of the requirements.

## Checkpoint

The released inference checkpoint is:

```text
result/model_best.pt
```

It contains the slim MobileIE-6Ch model with 101,922 parameters.

## Inference Reproduction

Place test images in:

```text
competition/low/
```

Run:

```bash
python infer_6channel.py
```

Predictions are written to:

```text
competition/enhanced_pt/
```

## Training Reproduction

Prepare paired training images with matching filenames:

```text
lowlight/
|-- low/
`-- normal/
```

Run DDP training:

```bash
bash train_ddp.sh "0,1,2,3" 4
```

The main training configuration is:

```text
config/lle.yaml
```

Key settings:

| Setting | Value |
| --- | --- |
| Model | MobileIE-6Ch |
| Channels | 32 |
| Epochs | 800 |
| Patch size | 768 |
| Batch size | 16 |
| Learning rate | 1.5e-4 |
| Warm-up | 20 epochs |
| EMA | Enabled |
| Cross validation | 5 folds |

## Dataset Notice

The NTIRE challenge data is not redistributed in this repository. Please obtain the data from the official challenge source and follow the challenge license and usage policy.

## Known Practical Notes

- The training script expects a Linux-like shell environment for `torchrun`.
- `experiments/` is intentionally ignored by Git because it stores logs and generated checkpoints.
- `competition/enhanced_pt/` is ignored by Git because it stores generated predictions.
- For exact benchmark comparison, use the official NTIRE evaluation protocol and metric implementation from the challenge organizers.
