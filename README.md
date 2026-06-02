# MobileIE-6Ch: Efficient Low-Light Image Enhancement

Official implementation of **MobileIE-6Ch**, the HIT-LLIE-team submission to the **NTIRE 2026 Efficient Low-Light Image Enhancement Challenge**.

MobileIE-6Ch is an ultra-lightweight, fully convolutional low-light image enhancement model built on the MobileIE family. It predicts a 6-channel Retinex-style representation: three RGB illumination channels and three residual channels for noise/detail compensation. The submitted inference checkpoint has only **101,922 parameters** and is included in this repository.

[[Paper / Challenge Report](https://arxiv.org/html/2605.02212v1)] [[NTIRE 2026](https://cvlai.net/ntire/2026/)]

## Highlights

- **Ultra-lightweight model**: 101,922 parameters in the released checkpoint.
- **6-channel enhancement head**: jointly estimates RGB illumination and residual correction.
- **Retinex-inspired reconstruction**: enhances brightness while suppressing low-light noise.
- **Multi-Branch Reparameterization (MBR)**: uses richer branches during training and compact convolutions for inference.
- **Dual attention modulation**: adaptively refines features for diverse dark scenes.
- **Ready-to-run inference**: pretrained checkpoint is provided at `result/model_best.pt`.

## NTIRE 2026 Results

The challenge report lists HIT-LLIE-team / MobileIE-6Ch in the NTIRE 2026 Efficient Low-Light Image Enhancement results.

| Table | SSIM | LPIPS | DISTS | LIQE | MUSIQ | Q-Align | Params | Final Rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Main technical-report table | 0.5766 | 0.5176 | 0.2319 | 2.2977 | 60.3387 | 3.2007 | 101,922 | 7 |
| Full final-testing table | 0.5766 | 0.5176 | 0.2319 | 2.2977 | 60.3387 | 3.2007 | 101,922 | 9 |

The full table includes teams that participated in final testing but did not submit a technical report; the main paper table reports teams included in the technical-methods section.

## Repository Structure

```text
.
|-- competition/
|   |-- low/                 # Put low-light images here for inference
|   `-- enhanced_pt/         # Inference outputs are written here
|-- config/
|   `-- lle.yaml             # Training and model configuration
|-- data/
|   |-- lledata.py           # Low-light enhancement dataset loader
|   `-- ispdata.py
|-- lowlight/
|   |-- low/                 # Training low-light images
|   `-- normal/              # Training ground-truth images
|-- model/
|   |-- lle.py               # Original MobileIE LLE model
|   |-- lle_6channel.py      # MobileIE-6Ch model
|   `-- utils_IWO.py         # MBR and feature modulation blocks
|-- result/
|   `-- model_best.pt        # Released MobileIE-6Ch checkpoint
|-- infer_6channel.py        # Inference script
|-- main_ddp.py              # DDP training entry point
|-- train_ddp.sh             # Training launcher
|-- requirements.txt
`-- team_info.txt
```

## Installation

The code was developed with PyTorch and CUDA. Python 3.8+ is recommended; Python 3.10 is also suitable.

```bash
conda create -n mobileie python=3.10 -y
conda activate mobileie
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version if the default wheel is not suitable for your machine.

## Inference

1. Put input images into:

```text
competition/low/
```

2. Run:

```bash
python infer_6channel.py
```

3. Enhanced images will be saved to:

```text
competition/enhanced_pt/
```

By default, `infer_6channel.py` loads:

```text
result/model_best.pt
```

The script automatically uses CUDA when available; otherwise it falls back to CPU.

## Training

Prepare paired training data:

```text
lowlight/
|-- low/       # low-light inputs
`-- normal/    # normal-light ground truth with matching filenames
```

Start DDP training:

```bash
bash train_ddp.sh
```

You can also choose a custom GPU list:

```bash
bash train_ddp.sh "0" 1
bash train_ddp.sh "0,1" 2
bash train_ddp.sh "0,1,2,3" 4
```

The main configuration is in `config/lle.yaml`. The released setting uses:

| Setting | Value |
| --- | --- |
| Model | MobileIE-6Ch |
| Channels | 32 |
| Epochs | 800 |
| Batch size | 16 |
| Warm-up | 20 epochs |
| Learning rate | 1.5e-4 |
| Scheduler | Cosine annealing |
| Patch size | 768 |
| Cross validation | 5 folds |
| EMA | Enabled |
| Gradient clipping | 0.5 |

Checkpoints and logs are written under `experiments/`.

## Method Summary

MobileIE-6Ch follows a compact Retinex-style enhancement pipeline:

```text
low-light RGB image
        |
        v
MobileIE-6Ch feature extractor + dual attention
        |
        v
6-channel prediction = RGB illumination + RGB residual
        |
        v
enhanced image = input / illumination + residual
```

During training, multi-branch convolution blocks improve representation capacity. For inference, the model uses the slim reparameterized form, keeping deployment simple and compact.

## Team

**HIT-LLIE-team**

- Xinbai Wang, Harbin Institute of Technology
- Duo Liu, Harbin Institute of Technology

Contact information is available in `team_info.txt`.

## Citation

If this repository helps your research, please cite the NTIRE 2026 challenge report and this solution:

```bibtex
@article{yan2026ntire,
  title={NTIRE 2026 Challenge on Efficient Low Light Image Enhancement: Methods and Results},
  author={Yan, Jiebin and Tu, Chenyu and Lin, Qinghua and others},
  journal={arXiv preprint arXiv:2605.02212},
  year={2026}
}
```

```bibtex
@misc{hitllie2026mobileie6ch,
  title={MobileIE-6Ch: Efficient Low-Light Image Enhancement for NTIRE 2026},
  author={Wang, Xinbai and Liu, Duo},
  year={2026},
  note={HIT-LLIE-team submission to the NTIRE 2026 Efficient Low-Light Image Enhancement Challenge}
}
```

## Acknowledgements

This work is built upon the MobileIE baseline and was developed for the NTIRE 2026 Efficient Low-Light Image Enhancement Challenge.

## License

This project is released under the Apache License 2.0. See `LICENSE` for details.
