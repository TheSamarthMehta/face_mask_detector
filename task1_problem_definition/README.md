# Task 1: Problem Definition & Dataset Acquisition

## Problem Statement

**Task Type:** Object Detection  
**Domain:** Computer Vision - Health & Safety

### Objective

Develop a deep learning system to detect and classify face masks in images. The system should identify people in images and classify them into three categories:

- **with_mask**: Person wearing a mask correctly
- **without_mask**: Person not wearing a mask
- **mask_weared_incorrect**: Person wearing a mask incorrectly

### Application

This system can be used for:

- COVID-19 safety compliance monitoring
- Automated surveillance in public spaces
- Healthcare facility monitoring
- Public transport safety systems

## Dataset Information

- **Source:** Face Mask Detection Dataset
- **Format:** Images (PNG) with XML annotations (PASCAL VOC format)
- **Location:** `archive/` directory
- **Classes:** 3 (with_mask, without_mask, mask_weared_incorrect)

## Files

- `analyze_dataset.py`: Script to analyze dataset statistics and class distribution
- `README.md`: Documentation (this file)

## Usage

Run the analysis script:

```bash
python task1_problem_definition/analyze_dataset.py
```

## Expected Output

- Class distribution statistics
- Data quality assessment
- Visualization of class balance (bar chart and pie chart)
- Imbalance ratio calculation
