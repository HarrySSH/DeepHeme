<div align="center">
    <img src="assets/image.png" alt="DeepHeme Logo" width="256px">
</div>

<div align="center">
    <h1>DeepHeme: A High-Performance, Generalizable, Deep Ensemble for Bone Marrow Morphometry and Hematologic Diagnosis</h1>
</div>

## Overview
DeepHeme is a state-of-the-art deep learning framework designed for bone marrow morphometry analysis and hematologic diagnosis. This repository contains the implementation of our deep ensemble approach, which combines multiple models to achieve robust and accurate results in hematologic image analysis.

## Features
- High-performance deep ensemble architecture
- Support for both regular training and snapshot ensemble approaches
- Flexible data preparation pipeline
- Compatible with ImageFolder-style datasets
- Configurable training parameters

## Installation

1. Clone the repository:
```bash
git clone https://github.com/HarrySSH/DeepHeme.git
cd DeepHeme
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation

The framework requires a structured dataset with proper train/validation/test splits. The data preparation process involves creating a CSV file containing image paths, labels, and split information.

1. Prepare your dataset in an ImageFolder-compatible structure
2. Configure the data preparation script:
   - Open `Data_preparation.py`
   - Set `image_collect_root_dir` to your dataset root directory
3. Run the data preparation script:
```bash
python Data_preparation.py
```

The script will generate a CSV file with the following columns:
- `fpath`: Path to the image file
- `label`: Class label
- `split`: Dataset split (train/val/test)

Note: For specialized datasets (e.g., whole slide images or patch-level analysis), you may need to modify the splitting logic in the data preparation script.

### Training

The framework supports two training approaches:

#### 1. Regular Training
```bash
python main.py
```

#### 2. Snapshot Ensemble
```bash
python main_se.py
```

Both approaches use the same CNN architecture but differ in their training methodology. The snapshot ensemble approach creates multiple model checkpoints during training and combines their predictions for improved performance.

## Configuration

The training parameters can be configured in the respective main scripts:
- `main.py` for regular training
- `main_se.py` for snapshot ensemble

Key parameters include:
- Learning rate
- Batch size
- Number of epochs
- Model architecture
- Ensemble settings

## Citation

If you use this code in your research, please cite our work: