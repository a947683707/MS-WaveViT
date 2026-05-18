# MS-WaveViT: Multi-Scale Wavelet Vision Transformer with HDR-Aware Attention for HDR Image Quality Assessment

This repository provides the implementation of **MS-WaveViT**, a no-reference HDR image quality assessment model. The proposed model integrates a WaveViT backbone, HDR-aware feature enhancement, multi-scale wavelet attention, and a HyperNet-TargetNet prediction framework.
## Important Clarifications

This repository has been updated to clarify the relationship between the proposed MS-WaveViT model and the baseline HyperIQA implementation.

### 1. Wavelet basis

The original WaveViT backbone contains Haar-based wavelet attention. In contrast, the proposed Multi-Scale Wavelet Attention (MSWA) module uses **Daubechies-4 (db4)** as the default wavelet basis.

Therefore, the Haar wavelet mentioned in the backbone and the db4 wavelet used in the proposed MSWA module correspond to different parts of the architecture. The wavelet-basis ablation experiments in the manuscript refer to the MSWA module.

### 2. Relation to HyperIQA

MS-WaveViT adopts the HyperNet-TargetNet prediction paradigm for content-adaptive quality regression. The original HyperIQA-related code is retained only as a baseline/reference implementation and is not the default training pipeline of the proposed model.

The proposed model is implemented in:

- `wave_models/wavehypernet.py`

The proposed training solver is implemented in:

- `wave_models/wavehyper_solver.py`

The main training entry is:

- `train_mswavevit.py`

### 3. Main datasets

The main experiments in the manuscript are conducted on the **Narwaria** and **Korshunov** HDR-IQA datasets.

Dataset support is provided in:

- `data_loader.py`
- `folders.py`

Raw HDR images are not redistributed in this repository due to dataset license restrictions. Users should download the datasets from the official source and organize them following the required folder structure.
## Repository Structure

```text
MS-WaveViT/
├── train_mswavevit.py
├── data_loader.py
├── folders.py
├── wave_models/
│   ├── wavehypernet.py
│   ├── wavehyper_solver.py
│   ├── wavevit.py
│   └── torch_wavelets.py
├── models.py
├── CITATION.cff
└── README.md
```

Note: `models.py` contains the TargetNet implementation used by the MS-WaveViT training pipeline. Legacy HyperIQA training scripts have been removed from the repository to avoid confusion.

```bash
pip install torch torchvision
pip install numpy scipy tqdm timm PyWavelets opencv-python openpyxl
```
## Dataset Preparation

Please organize the HDR-IQA datasets as follows:

```text
upiq_dataset/
├── upiq_subjective_scores.csv
└── images/
    ├── narwaria/
    │   ├── 01/
    │   ├── 02/
    │   └── ...
    └── korshunov/
        ├── ...
```

The raw HDR images are not included in this repository. Please download the datasets from the official source and place them following the folder structure above.
## Training

Train MS-WaveViT on Narwaria:

```bash
python train_mswavevit.py --dataset narwaria --batch_size 8 --epochs 12 --train_patch_num 25 --test_patch_num 25 --patch_size 224 --train_test_num 10 --lr 2e-5 --weight_decay 5e-4 --lr_ratio 10
```

Train MS-WaveViT on Korshunov:

```bash
python train_mswavevit.py --dataset korshunov --batch_size 8 --epochs 12 --train_patch_num 25 --test_patch_num 25 --patch_size 224 --train_test_num 10 --lr 2e-5 --weight_decay 5e-4 --lr_ratio 10
```
## Reproducibility

The reported results are obtained from 10 independent train/test rounds. The training script records the PLCC and SRCC values for each round and saves the best model checkpoint.

The main files for reproducing the proposed method are:

- `train_mswavevit.py`
- `wave_models/wavehyper_solver.py`
- `wave_models/wavehypernet.py`
- `wave_models/wavevit.py`
- `wave_models/torch_wavelets.py`
- `data_loader.py`
- `folders.py`

## Acknowledgement

The HyperNet-TargetNet prediction paradigm is inspired by HyperIQA. We thank the HyperIQA authors for releasing their implementation.

## Citation

If you use this code, please cite:

```bibtex
@article{zhang2026mswavevit,
  title   = {Multi-Scale Wavelet Vision Transformer with HDR-Aware Attention for High Dynamic Range Image Quality Assessment},
  author  = {Zhang, Rui and Dong, Wu and Lu, Likun and Zhou, Ziyi and Zhang, Tianqi and Niu, Weipeng},
  journal = {Manuscript under review},
  year    = {2026}
}
```
