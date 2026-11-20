# DIT5411 Traditional Chinese Handwriting Recognition




## Project Description

This project uses a Residual Convolutional Neural Network (Residual CNN) to recognise Traditional Chinese handwritten characters. The input images are converted to grayscale, resized to 64 x 64 pixels, augmented during training, and classified into 4,803 character classes.

## Dataset Structure

```text
data/cleaned_data/
├── 中/
│   ├── 中_1.png
│   └── 中_2.png
├── 人/
└── 永/
```

The first 40 images of each character are used for training and validation. Images after the first 40 samples are reserved for testing.

## Image Preprocessing

Each image is:

1. Loaded as a grayscale image.
2. Resized to 64 x 64 pixels.
3. Converted to floating-point format.
4. Normalised from 0-255 to 0.0-1.0.
5. Assigned a numerical class label.

## Data Augmentation

The training images are augmented using:

- Rotation: approximately +/-10 degrees
- Horizontal and vertical translation: 8%
- Zoom: 10%
- Unequal scaling: 8%
- Horizontal shear: 15%
- Vertical shear: 8%
- Clean-image probability: 30%

Augmentation is applied only to the training set. It generates variations of the original handwriting and helps the model recognise different writing styles.

## AI Models

### 1. Custom CNN

The Custom CNN is a baseline convolutional neural network with multiple convolution, activation, pooling, dropout, and dense layers. The new experiment has 7,056,169 parameters and a 13,065-neuron output layer, with one output for each character class.

### 2. Residual CNN

The Residual CNN uses shortcut connections between convolutional layers. These connections help preserve stroke features and improve gradient flow through deeper layers. The completed earlier experiment has 2,838,115 parameters and 4,803 output classes.

### Model Size

- Total parameters: 2,838,115
- Trainable parameters: 2,836,771
- Non-trainable parameters: 1,344
- Approximate model size: 10.83 MB

## Training Configuration

- Model: Residual CNN
- GPU: NVIDIA GeForce RTX 5070 Ti
- Input shape: 64 x 64 x 1
- Character classes: 4,803
- Learning rate: 0.0005 initially, reduced to 0.00025 from epoch 10
- Completed epochs: 15
- Optimizer: Adam
- Loss function: Sparse categorical cross-entropy
- Random seed: 250276567

## Residual CNN Training Results

| Metric | Result |
|---|---:|
| Initial training accuracy | 11.03% |
| Final training accuracy | 94.08% |
| Initial training loss | 5.1305 |
| Final training loss | 0.1913 |
| Best validation accuracy | 82.90% |
| Epoch with best validation accuracy | Epoch 12 (CSV index 11) |
| Best validation loss | 0.7679 |
| Final validation accuracy | 2.77% |
| Final validation loss | 38.0095 |

### Custom CNN experiment

- Framework: TensorFlow
- GPU: NVIDIA GeForce RTX 5070 Ti
- Image size: 64 x 64
- Classes: 13,065
- Epochs completed: 15
- Parameters: 7,056,169

Training accuracy increased steadily from 11.03% to 94.08%, while training loss decreased from 5.1305 to 0.1913. The best validation result occurred at epoch 12, where validation accuracy reached 82.90% and validation loss fell to 0.7679.

However, validation performance was highly unstable. Validation accuracy dropped to 2.77% by the final epoch even though training accuracy continued to improve. This large difference suggests overfitting, an unstable validation pipeline, or a possible issue with validation labels, class ordering, batch-normalisation statistics, or dataset size. The checkpoint from epoch 12 should be used instead of the final epoch model.

## Test-Set Result

The saved best Residual CNN model was evaluated on the held-out test set. The evaluation processed 58,759 test images and produced the following results:

## Test Results

| Model | Classes | Image size | Parameters | Epochs | Test accuracy | Top-5 accuracy | Device |
|---|---:|---:|---:|---:|---:|---:|---|
| Custom CNN | 13,065 | 64 x 64 | 7,056,169 | 5 | **0.0141%** | **0.0880%** | GPU |
| Residual CNN | 4,803 | 64 x 64 | 2,838,115 | 15 | **82.91%** | **95.92%** | GPU |

### Exact Residual CNN Evaluation Values
- Exact test accuracy: `0.829149603843689`
- Exact top-5 test accuracy: `0.959206223487854`
- Top-k value: `5`
- Evaluation device: `GPU (/physical_device:GPU:0)`
- TensorFlow version: `2.21.0`
- Image size: `64 x 64`
- Batch size: `128`
- Number of classes: `4,803`
- Training images: `172,763`
- Epochs requested/trained: `15/15`
- Training repeats: `5`

- ### Exact Custom CNN Evaluation Values

- Test accuracy: `0.00014149492199067026`
- Top-5 test accuracy: `0.000879729341249913`
- Top-k value: `5`
- Number of classes: `13,065`
- Parameters: `7,056,169`
- Epochs trained: `15`
- Device: `GPU (/physical_device:GPU:0)`
- Run directory: `runs/custom_cnn/rtx5070ti_custom_cnn`

The **82.91% test accuracy** means that the correct Chinese character was the model's first prediction for approximately 82.91% of the held-out test images. The **95.92% top-5 accuracy** means that the correct character appeared among the model's five highest-probability predictions for approximately 95.92% of the test images.

The evaluation metrics are stored in:

```text
runs/residual_cnn/common_residual_cnn_gpu_fresh/metrics.json
```

The evaluated model run is stored in:

```text
runs/residual_cnn/common_residual_cnn_gpu_fresh/
```
## How to Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to Train

```bash
python -m src.train \
  --config configs/default.yaml \
  --run-name residual_cnn
```

## How to Evaluate

Use the run directory containing the best model checkpoint:

```bash
python -m src.evaluate \
  --run-dir runs/residual_cnn/common_residual_cnn_gpu_fresh
```



## Files Produced

The training or evaluation process may produce:

- `best.keras`: model checkpoint with the best validation performance
- `checkpoint.keras`: saved training checkpoint
- `training_log.csv`: accuracy, loss, validation metrics, and learning rate by epoch
- `model_summary.txt`: model architecture and parameter counts
- `training_curves.png`: training and validation curves
- `metrics.json`: test-set evaluation results
- `comparison.csv`: comparison results for multiple models
-comparison.md
-comparison.csv
-comparison.png

## Current Conclusion

The Residual CNN achieved **82.91% test accuracy** and **95.92% top-5 test accuracy** across **58,759 held-out test images** from **4,803 character classes**. Also The Custom CNN now satisfies the required 13,065 output classes, but the current 15-epoch test result is not yet suitable as the final recognition model.The difference between top-1 and top-5 accuracy shows that many incorrect first predictions still placed the correct character among the five most likely choices. These figures are test-set results from `metrics.json`, not training or validation accuracy.
## References

Add the dataset source, TensorFlow documentation, Keras documentation, and the source used to explain residual neural networks.
