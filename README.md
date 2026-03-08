# OF-TransUNet

Official repository for the manuscript:

**Efficient Transformer Integration in nnU-Net for Liver Tumor Segmentation: An External Validation Study**

## Repository status

This manuscript is currently **under review** at *BMC Medical Imaging*.  
To support transparency during peer review, this repository has been created in advance.

**Current status:** the full source code is still being cleaned, documented, and organized.  
**Planned release:** complete code and reproducibility materials will be released upon manuscript acceptance.

## Current public contents

At the current review stage, this repository provides:

- Project-level metadata
- License information
- A supplementary implementation note describing the staged Transformer unfreezing schedules
- Environment summary and dependency information
- Planned reproducibility notes

## Currently available note

- **Supplementary Note Sx. Definition of staged Transformer unfreezing schedules**

This note clarifies:

- No Fine-tuning
- Full Fine-tuning
- Simple Progressive schedule
- Final Output-Focused schedule
- Epoch boundaries and warmup details used in the archived implementation logs

## Planned release upon acceptance

The full release is expected to include:

- Complete model implementation
- nnU-Net v2-based training pipeline
- OF-TransUNet architecture definition
- Training and inference scripts
- Staged unfreezing / adaptation logic
- Evaluation scripts for segmentation and lesion-level detection
- Pretrained model weights
- Documentation for reproducing the main tables in the manuscript

## Environment summary

Primary development environment used in this study:

- Python 3.11
- PyTorch 2.5.1 + CUDA 11.8
- torchvision 0.20.1
- torchaudio 2.5.1
- nnU-Net v2
- SimpleITK 2.4.0
- NumPy 2.2.1
- SciPy 1.14.1
- scikit-learn 1.5.2
- matplotlib 3.10.0
- pandas 2.2.3

A more detailed dependency list will be provided in a dedicated environment file.

## Reproducibility note

After acceptance, the repository will include step-by-step instructions for reproducing the manuscript results, including:

- preprocessing and training setup
- inference commands
- lesion-level detection analysis
- computational profiling
- generation of the main quantitative results reported in Tables 1–3

## Data availability

The public HCC-TACE dataset used for internal development is publicly available:
https://doi.org/10.1038/s41597-023-01928-3

The external clinical cohort is not publicly released due to institutional data-sharing restrictions.

De-identified derived results (for example, per-patient segmentation metrics and lesion-level detection summary tables) may be shared upon reasonable request, subject to institutional approval and applicable data-sharing restrictions.

## Contact

For questions regarding the repository during peer review, please contact the corresponding author listed in the manuscript.

## License

MIT

