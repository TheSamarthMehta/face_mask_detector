# Task 3: Model Architecture Design & Training

## Custom CNN Architecture

### Model Design

Our custom CNN consists of:

**Convolutional Blocks:**

1. **Block 1** (32 filters)
   - Conv2D (3x3) → BatchNormalization → Conv2D (3x3) → BatchNormalization
   - MaxPooling2D → Dropout (0.25)

2. **Block 2** (64 filters)
   - Conv2D (3x3) → BatchNormalization → Conv2D (3x3) → BatchNormalization
   - MaxPooling2D → Dropout (0.25)

3. **Block 3** (128 filters)
   - Conv2D (3x3) → BatchNormalization → Conv2D (3x3) → BatchNormalization
   - MaxPooling2D → Dropout (0.25)

4. **Block 4** (256 filters)
   - Conv2D (3x3) → BatchNormalization
   - MaxPooling2D → Dropout (0.25)

**Dense Layers:**

- Flatten
- Dense (512 units, ReLU) → BatchNormalization → Dropout (0.5)
- Dense (256 units, ReLU) → BatchNormalization → Dropout (0.5)
- Dense (3 units, Softmax)

### Training Configuration

- **Optimizer**: Adam
- **Learning Rate**: 0.001 (with ReduceLROnPlateau)
- **Loss Function**: Sparse Categorical Crossentropy
- **Batch Size**: 32
- **Epochs**: 50 (with Early Stopping)

### Regularization Techniques

- **Batch Normalization**: Stabilize training
- **Dropout**: Prevent overfitting (0.25 and 0.5)
- **L2 Regularization**: Implicit through architecture
- **Data Augmentation**: Applied in Task 2

### Callbacks

1. **ModelCheckpoint**: Save best model based on validation accuracy
2. **EarlyStopping**: Stop training if no improvement for 10 epochs
3. **ReduceLROnPlateau**: Reduce learning rate by 0.5 if no improvement for 5 epochs

## Files

- `train_model.py`: Main training script
- `training_history.png`: Accuracy and loss plots
- `training_history.json`: Detailed training metrics

## Usage

```bash
python task3_model_training/train_model.py
```

## Output

- Trained model saved in `models/custom_cnn_best.h5`
- Training visualization
- Model summary and architecture details
