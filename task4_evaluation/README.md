# Task 4: Evaluation & Hyperparameter Tuning

## Evaluation Metrics

### Performance Metrics

- **Accuracy**: Overall classification accuracy
- **Precision**: Per-class precision scores
- **Recall**: Per-class recall scores
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown

### Visualizations

1. **Confusion Matrix**: Shows prediction accuracy for each class
2. **Class Metrics Bar Chart**: Compares precision, recall, and F1-score
3. **Training Analysis**:
   - Accuracy over epochs
   - Loss over epochs
   - Train-validation gap analysis

## Overfitting/Underfitting Detection

The evaluation script automatically detects:

### Overfitting Indicators

- Training accuracy >> Validation accuracy (gap > 10%)
- Validation loss increasing while training loss decreasing

**Solutions Applied:**

- Dropout layers (0.25, 0.5)
- Batch normalization
- Early stopping
- Data augmentation

### Underfitting Indicators

- Both training and validation accuracies are low (< 70%)
- High training loss

**Solutions:**

- Increase model capacity
- Train for more epochs
- Adjust learning rate

## Hyperparameter Tuning

### Parameters Tuned

1. **Learning Rate**: 0.001 (with ReduceLROnPlateau)
   - Automatically reduces by 0.5 when validation loss plateaus
2. **Batch Size**: 32
   - Balanced between training speed and memory

3. **Dropout Rates**: 0.25 (conv layers), 0.5 (dense layers)
   - Prevents overfitting

4. **Number of Filters**: 32 → 64 → 128 → 256
   - Progressive feature learning

5. **Dense Layer Sizes**: 512 → 256
   - Adequate capacity without overfitting

## Files

- `evaluate_model.py`: Main evaluation script
- `confusion_matrix.png`: Confusion matrix visualization
- `class_metrics.png`: Per-class performance metrics
- `training_analysis.png`: Comprehensive training analysis
- `evaluation_results.json`: Numerical results

## Usage

```bash
python task4_evaluation/evaluate_model.py
```

## Expected Output

- Test accuracy and loss
- Classification report
- Confusion matrix
- Overfitting/underfitting analysis
- Recommendations for improvement
