# Model Card: MobileIE-6Ch

## Model Details

- **Model name:** MobileIE-6Ch
- **Task:** Low-light image enhancement
- **Team:** HIT-LLIE-team
- **Affiliation:** Harbin Institute of Technology
- **Challenge:** NTIRE 2026 Efficient Low-Light Image Enhancement
- **Framework:** PyTorch
- **Checkpoint:** `result/model_best.pt`
- **Parameter count:** 101,922
- **License:** Apache-2.0

## Intended Use

MobileIE-6Ch is intended for research on efficient low-light image enhancement and image restoration. It is suitable for studying compact enhancement networks, Retinex-inspired reconstruction, and resource-constrained restoration pipelines.

## Out-of-Scope Use

This model should not be treated as a general-purpose image forensics, surveillance, or safety-critical vision component. Enhanced images may change brightness, color, contrast, or local texture appearance, so downstream decisions should account for possible restoration artifacts.

## Architecture Summary

MobileIE-6Ch predicts a six-channel output:

- three RGB illumination channels
- three RGB residual correction channels

The enhanced image is reconstructed as:

```text
enhanced = input / illumination + residual
```

The training-time model uses multi-branch reparameterized convolution blocks, while the released checkpoint uses the slim inference form.

## Evaluation

The model is reported in the NTIRE 2026 Efficient Low-Light Image Enhancement Challenge report.

| Table | SSIM | LPIPS | DISTS | LIQE | MUSIQ | Q-Align | Params | Final Rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Main technical-report table | 0.5766 | 0.5176 | 0.2319 | 2.2977 | 60.3387 | 3.2007 | 101,922 | 7 |
| Full final-testing table | 0.5766 | 0.5176 | 0.2319 | 2.2977 | 60.3387 | 3.2007 | 101,922 | 9 |

## Limitations

- Results may vary across sensors, compression levels, noise patterns, and extremely dark scenes.
- The model can amplify sensor noise when the input contains severe underexposure or color casts outside the training distribution.
- The repository does not redistribute the NTIRE challenge dataset.
- The released code is research-oriented and may require adaptation for production deployment.

## Citation

Please cite the NTIRE 2026 challenge report and this repository if you use MobileIE-6Ch in your work.
