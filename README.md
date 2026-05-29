# MindGlide MS Lesion Segmentation — Evaluation
##  MSSEG-2 Dataset

Evaluation of the pretrained MindGlide model on the MSSEG-2 dataset for multiple sclerosis (MS) white matter lesion segmentation. All 20 subjects were evaluated using time01 FLAIR MRI scans compared against manual expert annotations.

---

## Model

- **Model:** MindGlide (MS-PINPOINT/mindGlide)
- **Architecture:** Dynamic U-Net (DynUNet) via MONAI
- **Checkpoint:** `_20240404_conjurer_trained_dice_7733.pt`
- **Source:** https://github.com/MS-PINPOINT/mindGlide
- **Output classes:** 20 brain structures — MS Lesion is label 18
- **Parameters:** 30.78 million
- **FLOPs (MACs):** 479.3 GMACs
- **Input patch size:** 128 × 128 × 64

---

## Dataset

- **Dataset:** MSSEG-2 
- **Subjects:** 20 
- **Modality:** FLAIR MRI only
- **Timepoint used:** time01 for all subjects
- **Image dimensions:** 365 × 256 × 256 voxels
- **Voxel spacing:** 0.500 × 0.977 × 0.977 mm (anisotropic)
- **Gold standard:** manual/ annotations (ground/ folder was found unreliable across all subjects and was not used)

---

## Setup & Installation

### 1. Install Git LFS and clone MindGlide

```bash
sudo apt-get install git-lfs
git lfs install
git clone --recurse-submodules https://github.com/MS-PINPOINT/mindGlide.git
cd mindGlide
git submodule foreach 'git lfs pull'
git lfs pull
cd ..
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env with your actual dataset and model paths
```

### 4. Run evaluation

```bash
python master_evaluation.py
```

---

## Preprocessing

All preprocessing is handled internally by MindGlide:

- Resampling to 1.0 × 1.0 × 1.0 mm isotropic spacing
- Reorientation to RAS coordinate system
- Intensity normalization to [0, 1]
- Foreground cropping
- Sliding window inference — patch size 128 × 128 × 64, overlap 0.5, Gaussian weighting
- Post-inference reorientation back to original image space

---

## Metrics Computed

Dice, NSD, ASSD, HD95, Boundary IoU, Precision, Recall, F1 Score, Inference Time

---

## Per-Subject Results

| Subject | Dice | NSD | ASSD | HD95 | Boundary IoU | Precision | Recall | F1 | Time (s) |
|---------|------|-----|------|------|--------------|-----------|--------|----|----------|
| sub-013 | 0.6328 | 0.7405 | 2.583 | 19.44 | 0.3382 | 0.6836 | 0.5890 | 0.6328 | 8.65 |
| sub-015 | 0.6150 | 0.7515 | 3.065 | 29.16 | 0.3301 | 0.6069 | 0.6234 | 0.6150 | 8.70 |
| sub-016 | 0.6961 | 0.7861 | 3.286 | 28.39 | 0.3853 | 0.6992 | 0.6931 | 0.6961 | 8.86 |
| sub-018 | 0.7057 | 0.7285 | 1.778 | 13.43 | 0.3438 | 0.6367 | 0.7915 | 0.7057 | 9.14 |
| sub-019 | 0.6533 | 0.4581 | 2.464 | 9.00 | 0.1699 | 0.5067 | 0.9191 | 0.6533 | 9.21 |
| sub-020 | 0.4594 | 0.4637 | 4.613 | 19.05 | 0.1834 | 0.3640 | 0.6228 | 0.4594 | 9.23 |
| sub-021 | 0.7911 | 0.8141 | 1.670 | 16.43 | 0.3991 | 0.8015 | 0.7810 | 0.7911 | 9.44 |
| sub-024 | 0.6434 | 0.7031 | 2.627 | 32.31 | 0.3176 | 0.5444 | 0.7864 | 0.6434 | 9.83 |
| sub-026 | 0.7834 | 0.8803 | 0.827 | 4.05 | 0.5114 | 0.7546 | 0.8145 | 0.7834 | 9.96 |
| sub-027 | 0.6916 | 0.7479 | 2.463 | 25.80 | 0.3770 | 0.7252 | 0.6609 | 0.6916 | 9.89 |
| sub-029 | 0.3085 | 0.4043 | 29.617 | 214.33 | 0.1256 | 0.3987 | 0.2517 | 0.3085 | 9.95 |
| sub-030 | 0.7669 | 0.7763 | 2.599 | 26.87 | 0.3612 | 0.7300 | 0.8076 | 0.7669 | 11.89 |
| sub-032 | 0.5420 | 0.5863 | 3.661 | 12.17 | 0.2377 | 0.4027 | 0.8283 | 0.5420 | 10.12 |
| sub-035 | 0.6891 | 0.6921 | 2.834 | 21.21 | 0.2904 | 0.7225 | 0.6586 | 0.6891 | 9.94 |
| sub-037 | 0.7395 | 0.8006 | 1.752 | 3.61 | 0.4234 | 0.6724 | 0.8215 | 0.7395 | 10.17 |
| sub-039 | 0.7440 | 0.7226 | 2.443 | 20.63 | 0.3160 | 0.8148 | 0.6845 | 0.7440 | 9.99 |
| sub-043 | 0.3022 | 0.3534 | 6.986 | 28.71 | 0.1418 | 0.2868 | 0.3194 | 0.3022 | 10.04 |
| sub-047 | 0.6971 | 0.6773 | 6.500 | 17.20 | 0.2753 | 0.5707 | 0.8954 | 0.6971 | 10.13 |
| sub-048 | 0.6814 | 0.7468 | 2.255 | 21.02 | 0.3353 | 0.6625 | 0.7014 | 0.6814 | 10.19 |
| sub-049 | 0.7531 | 0.8046 | 3.462 | 20.83 | 0.3978 | 0.7256 | 0.7827 | 0.7531 | 9.86 |

---

## Aggregate Results (20 Subjects)

| Metric | Mean | Std |
|--------|------|-----|
| Dice | 0.6448 | ±0.1409 |
| NSD | 0.6819 | ±0.1485 |
| ASSD (mm) | 4.3742 | ±6.1287 |
| HD95 (mm) | 29.182 | ±44.316 |
| Boundary IoU | 0.3130 | ±0.0999 |
| Precision | 0.6155 | ±0.1526 |
| Recall | 0.7016 | ±0.1693 |
| F1 Score | 0.6448 | ±0.1409 |
| Inference Time (s) | 9.7595 | ±0.7132 |

---

## Computational Profile

| Property | Value |
|----------|-------|
| Architecture | DynUNet (Dynamic U-Net) |
| Total parameters | 30.78 million |
| FLOPs (MACs) | 479.3 GMACs |
| Input patch size | 128 × 128 × 64 |
| Output classes | 20 |
| Normalization | Instance Normalization |
| Deep supervision heads | 3 |
| Mean inference time | 9.76 seconds per subject |
| Runtime | Google Colab, NVIDIA T4 GPU |

---

## Segmentation Failures

- **sub-029** — Dice: 0.308, HD95: 214.33mm, ASSD: 29.62mm. Severe failure. Very low manual lesion volume (1,657 voxels). Model produced spatially incorrect predictions. Confirmed in 3D Slicer — no visible lesion overlay.
- **sub-043** — Dice: 0.302, HD95: 28.71mm. Poor detection and low precision (0.287). Small lesion burden (2,837 voxels).
- **sub-020** — Dice: 0.459, Precision: 0.364. Significant oversegmentation observed.

---

## 3D Slicer Verification

Cases sub-019, sub-021, and sub-029 were opened in 3D Slicer 5.10.0. FLAIR scans loaded as Volume, predictions loaded as Segmentation. Segment 18 (MS Lesion) was isolated and verified visually across axial, coronal, and sagittal planes.

- **sub-021** — Lesions clearly visible near ventricles in all three planes 
- **sub-019** — Large lesion burden confirmed, prominent periventricular lesions 
- **sub-029** — No visible lesion overlay, consistent with segmentation failure 

---

## Runtime Environment

- Google Colab, NVIDIA T4 GPU (16GB VRAM)
- CUDA 12.8 | PyTorch 2.11.0 | MONAI 1.5.2 | Python 3.12

---
