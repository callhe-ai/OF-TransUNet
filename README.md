# OF-TransUNet

Official repository for the manuscript:

**Efficient Transformer Integration in nnU-Net for Liver Tumor Segmentation: An External Validation Study**

## Repository status

This manuscript is currently **under review** at *BMC Medical Imaging*.  
To support transparency during peer review, this repository has been made publicly available in advance of acceptance.

**Current status:** The core trainer implementation (`nnUNetTrainerTransUNet.py`) is provided for peer review transparency. Full documentation, inference scripts, and pretrained weights will be finalized upon manuscript acceptance.  
**Planned release:** Complete reproducibility materials will be released upon acceptance.

---

## Source code

- **`nnUNetTrainerTransUNet.py`** — Core trainer implementation (provided for peer review transparency).  
  This file contains the complete OF-TransUNet architecture and output-focused progressive unfreezing schedule as described in the manuscript.

---

## Correspondence to manuscript

The following tables document the direct correspondence between key implementation details in the code and the descriptions in the manuscript.

### Architecture insertion

| Manuscript description | Code location | Value |
|---|---|---|
| Single Conv-Transformer block inserted at encoder stage 3 (×8 downsampling) | `transformer_config['insert_stage']` in `__init__` | `3` |
| Lightweight convolutional self-attention (QKV projection) | `CustomTransformer.qkv` in `create_custom_transformer` | `Conv2d(hidden_dim, hidden_dim * 3, kernel_size=1)` |
| Convolutional FFN | `CustomTransformer.conv2` / `conv3` | `Conv2d(hidden_dim, hidden_dim * 4, ...)` → `Conv2d(hidden_dim * 4, hidden_dim, ...)` |
| Output-side residual mapping (`final`) | `CustomTransformer.final` | `Conv2d(hidden_dim, in_channels, kernel_size=3, padding=1)` |
| Input residual connection | `forward`: `out = out + x` | Global skip connection to input feature map |

### Output-focused progressive unfreezing schedule

The `layer_groups` dictionary in `nnUNetTrainerTransUNet.__init__` directly implements the output-focused unfreezing schedule described in the manuscript:

| Group ID | Released components | Epoch triggered | Manuscript description |
|---|---|---|---|
| 1 | `["final"]` | Epoch 90 | Output-side mapping component; released **first** |
| 2 | `["final", "conv2", "conv3"]` | Epoch 110 | FFN components added |
| 3 | `["qkv", "proj"]` | Epoch 130 | Attention projection components added |
| 4 | `["all"]` | Epoch 150+ | Full unfreezing |

```python
self.layer_groups = {
    1: ["final"],                       # output-side mapping, released first
    2: ["final", "conv2", "conv3"],     # FFN components
    3: ["qkv", "proj"],                 # attention projections
    4: ["all"]                          # full unfreezing
}
```

### Epoch boundaries and warmup

| Manuscript description | Code parameter | Value used in reported experiments |
|---|---|---|
| Transformer frozen during epochs 0–89 | `base_training_epochs` | `90` |
| Output-side release begins at epoch 90 | `unfreeze_transformer_blocks` triggered at `current_epoch == base_training_epochs` | epoch 90 |
| Warmup period epochs 90–94 | `warm_up_transformer_lr(warmup_epochs=5)` | 5 epochs |

### Optimizer and training hyperparameters

| Manuscript description | Code parameter | Value |
|---|---|---|
| SGD optimizer | `torch.optim.SGD` in `configure_optimizer` | momentum=0.99, Nesterov=True |
| Initial learning rate 0.01 | `self.initial_lr` | `1e-2` |
| Poly LR scheduler | `PolyLRScheduler` | applied to all parameter groups |
| Gradient clipping threshold | `self.grad_clip_value` | `12.0` |
| Transformer LR factor after unfreezing | `self.unfreeze_lr_factor` | `0.1` (i.e., 1e-3) |
| NaN/Inf handling | `handle_nan_in_loss` + gradient check in `train_step` | LR halved after 3 consecutive NaN events |

### Note on TransUNetEncoder

The file imports `TransUNetEncoder` via a conditional `try/except` block. This fallback encoder is only invoked when `use_custom_transformer=False`, which requires sufficiently large feature map spatial dimensions. **In all reported experiments, the custom Conv-Transformer path (`create_custom_transformer`) was used exclusively.** The `TransUNetEncoder` branch was not exercised in the results reported in the manuscript.

---

## Currently available files

| File | Description |
|---|---|
| `nnUNetTrainerTransUNet.py` | Core trainer — OF-TransUNet architecture and unfreezing schedule |
| `Supplementary Note S2. Definition of staged Transformer unfreezing schedules` | Clarifies all four unfreezing schedule variants evaluated in ablation |
| `requirements.txt` | Full dependency list |
| `LICENSE` | MIT license |

---

## Supplementary Note S2

**Definition of staged Transformer unfreezing schedules**

This note clarifies the four schedule variants evaluated in internal ablation:

- No Fine-tuning
- Full Fine-tuning
- Simple Progressive schedule
- Final Output-Focused schedule (OF-TransUNet)
- Epoch boundaries and warmup details used in the archived implementation logs

---

## Planned release upon acceptance

The full release is expected to include:

- Complete model implementation with documentation
- nnU-Net v2-based training pipeline
- Training and inference scripts
- Evaluation scripts for segmentation and lesion-level detection
- Pretrained model weights
- Step-by-step instructions for reproducing the main tables in the manuscript

---

## Environment summary

Primary development environment used in this study:

| Package | Version |
|---|---|
| Python | 3.11 |
| PyTorch | 2.5.1 |
| CUDA | 11.8 |
| torchvision | 0.20.1 |
| torchaudio | 2.5.1 |
| nnU-Net | v2 |
| SimpleITK | 2.4.0 |
| NumPy | 2.2.1 |
| SciPy | 1.14.1 |
| scikit-learn | 1.5.2 |
| matplotlib | 3.10.0 |
| pandas | 2.2.3 |

A more detailed dependency list is provided in `requirements.txt`.

---

## Data availability

The public HCC-TACE dataset used for internal development is publicly available:  
https://doi.org/10.1038/s41597-023-01928-3

The external clinical cohort is not publicly released due to institutional data-sharing restrictions.

De-identified derived results (for example, per-patient segmentation metrics and lesion-level detection summary tables) may be shared upon reasonable request, subject to institutional approval and applicable data-sharing restrictions.

---

## Contact

For questions regarding the repository during peer review, please contact the corresponding author listed in the manuscript.

---

## License

MIT
