# Breast density in MRI: an AI-based quantification and relationship to assessment in mammography

**By [Yaqian Chen](https://scholar.google.com/citations?user=iegKFuQAAAAJ&hl=en), [Lin Li]([https://scholar.google.com/citations?user=aGjCpQUAAAAJ&hl=en&oi=ao](https://scholar.google.com/citations?user=uRHrZUkAAAAJ&hl=zh-CN&authuser=1)), [Hanxue Gu](https://scholar.google.com/citations?user=aGjCpQUAAAAJ&hl=en&oi=ao), [Haoyu Dong](https://scholar.google.com/citations?user=eZVEUCIAAAAJ&hl=en&oi=ao), [Derek L. Nguyen](#), [Allan D. Kirk](#), and [Maciej A. Mazurowski](https://scholar.google.com/citations?user=HlxjJPQAAAAJ&hl=en&oi=ao)**

Please email [yaqian.chen@duke.edu](mailto:yaqian.chen@duke.edu) for any problem with this code.
---
[![arXiv](https://img.shields.io/badge/arXiv-2502.09779-b31b1b.svg)](https://arxiv.org/abs/2504.15192)

Download the required model weight, demo_preprocessed, demo_raw, demo_result folder from [this Google Drive folder](https://drive.google.com/drive/folders/1QWPbxk6qKoB1Yc1QHCDjIfgWOIGLri4T?usp=sharing). Make sure to select the appropriate files for your configuration.

# BreastMRI_density

This is a deep learning-based breast MRI automatic segmentation and breast density calculation code. It can perform fibroglandular tissue (FGT), vessel, as well as breast region segmentation, on pre-contrast breast MRI images.

## Segmentation Code Pipeline Overview

This project provides a complete breast MRI image preprocessing and segmentation pipeline:
- **Preprocessing**: Automatically converts DICOM files to .npy format suitable for deep learning models
- **Dense Tissue and Vessel Segmentation**: Identifies and segments dense tissue and vessel structures in the breast
- **Breast Region Segmentation**: Segments the entire breast region

## File Structure

```
BreastMRI_github/
├── model/                          # Model definitions
│   ├── __init__.py
│   └── vnet.py                    # V-Net model implementation
├── utils/                         # Utility functions
│   ├── ops.py
│   └── utils.py
├── weight/                        # Pre-trained model weights
│   ├── vnet_baseline_breast_full_9511.pth    # Breast segmentation model
│   └── vnet_dv_baseline_159_aug_48.pth       # Dense tissue and vessel segmentation model
├── buffer/                        # Cache files
│   ├── HeaderList.json
│   ├── Item_Features.json
│   └── Process_df.csv
├── demo_raw/                      # Example DICOM data
│   └── 500.000000-ISPY2 AX VIBRANT-SERIES-52523/
│       ├── 1-001.dcm
│       ├── 1-002.dcm
│       └── ...                    # More DICOM files
├── demo_preprocessed/             # Preprocessed .npy files
│   └── 100899T0.npy
├── demo_result/                   # Segmentation results
│   ├── breast/                    # Breast segmentation results
│   │   └── 100899T0.seg.nrrd
│   └── dv/                        # Dense tissue and vessel segmentation results
│       └── 100899T0.seg.nrrd
├── dataset.py                     # Dataset loader
├── engine.py                      # Training and inference engine
├── preprocess.py                  # Data preprocessing module
├── predict_mask_lin.py            # Main prediction script
├── demo.csv                       # Example data configuration file
├── result_vis.ipynb              # Result visualization notebook
└── README.md                      # Project documentation
```

## Requirements

- Python 3.7+
- PyTorch 1.8+
- MONAI
- pydicom
- numpy
- pandas
- torchio
- nrrd

## Installation

```bash
pip install torch torchvision
pip install monai
pip install pydicom
pip install torchio
pip install nrrd
pip install tqdm
```

## Usage

### 1. Prepare Data

First, you need to prepare a CSV file with the following columns:
- `Study_Name`: Study name (used for naming result files)
- `Sequence_path`: Path to the DICOM folder

Example CSV file (`demo.csv`):
```csv
Study_Name,Sequence_path
100899T0,demo_raw/500.000000-ISPY2 AX VIBRANT-SERIES-52523
```

### 2. Data Preprocessing

Run the preprocessing command to convert DICOM files to .npy format:

```bash
python predict_mask_lin.py --preprocess True --sequence_df_path demo.csv --preprocessed_dir demo_preprocessed
```

Parameter descriptions:
- `--preprocess True`: Enable preprocessing mode
- `--sequence_df_path`: Path to CSV file
- `--preprocessed_dir`: Directory to save preprocessed results

### 3. Dense Tissue and Vessel Segmentation

```bash
python predict_mask_lin.py \
    --preprocessed_dir demo_preprocessed \
    --save_dir demo_result/dv \
    --load_path weight/vnet_dv_baseline_159_aug_48.pth \
    --sequence_df_path demo.csv \
    --patch_size 96 \
    --out_channel 3 \
    --full_eval_size (256, 256, 128) \
    --eval_task vessel_tissue
```

### 4. Breast Region Segmentation

```bash
python predict_mask_lin.py \
    --preprocessed_dir demo_preprocessed \
    --save_dir demo_result/breast \
    --load_path weight/vnet_baseline_breast_full_9511.pth \
    --sequence_df_path demo.csv \
    --patch_size 96 \
    --out_channel 3 \
    --full_eval_size (256, 256, 128) \
    --eval_task full_breast
```

## Output Results

- **Preprocessed results**: .npy files saved in `demo_preprocessed/` directory
- **Dense tissue and vessel segmentation**: .seg.nrrd files saved in `demo_result/dv/` directory
- **Breast segmentation**: .seg.nrrd files saved in `demo_result/breast/` directory

## Model Description

- **V-Net**: Uses 3D convolutional neural networks for medical image segmentation
- **Dense tissue and vessel segmentation model**: Identifies dense tissue and vessel structures in the breast
- **Breast segmentation model**: Segments the entire breast region

## Preprocessing Pipeline

1. **DICOM Reading**: Reads DICOM files and extracts pixel data
2. **Orientation Correction**: Automatically corrects image orientation to ensure patient body is in the lower part of the image
3. **Normalization**: Performs Z-score normalization on the images
4. **Padding**: Applies appropriate padding to meet model input requirements
5. **Saving**: Saves processed data as .npy format

## Important Notes

- Ensure the DICOM folder contains complete sequence data
- Preprocessing may take a long time depending on the data volume
- GPU is recommended for inference to achieve better performance
- Result files are saved in NRRD format for easy viewing with medical image software

## Example Data

The project provides complete example data:
- `demo_raw/`: Contains example DICOM files
- `demo_preprocessed/`: Contains preprocessed example data
- `demo_result/`: Contains segmentation result examples
