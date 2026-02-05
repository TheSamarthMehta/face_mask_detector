# Task 2: Data Pre-processing & Augmentation

## Overview

This module handles all data preprocessing and augmentation for the face mask detection dataset.

## Processing Steps

### 1. **Image Preprocessing**

- **Extract Face Regions**: Crop faces based on bounding box annotations
- **Resize**: Standardize all images to 224x224 pixels
- **Normalize**: Scale pixel values to [0, 1] range
- **Padding**: Add 10% padding around face regions

### 2. **Data Augmentation**

Applied to increase dataset size and model robustness:

- Horizontal flipping
- Rotation (±15 degrees)
- Brightness adjustment (0.8x and 1.2x)

### 3. **Dataset Split**

- **Training**: 72% of data
- **Validation**: 8% of data
- **Test**: 20% of data

## Files

- `preprocess_data.py`: Main preprocessing script
- `processed_data/`: Output directory for processed data
  - `X_train.npy`, `y_train.npy`: Training data
  - `X_val.npy`, `y_val.npy`: Validation data
  - `X_test.npy`, `y_test.npy`: Test data
  - `metadata.json`: Dataset metadata

## Usage

```bash
python task2_data_preprocessing/preprocess_data.py
```

## Output

- Processed and augmented images
- Train/validation/test splits
- Sample visualization
- Dataset statistics
