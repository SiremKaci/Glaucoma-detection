# Glaucoma Detection from Retinal Fundus Images

> **End-to-end deep learning project for automated glaucoma classification from retinal fundus images, using CLAHE-enhanced preprocessing, data augmentation, progressive transfer learning with ConvNeXt-Tiny, and validation-based decision threshold optimization.**

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/ConvNeXt--Tiny-Model-5C3EE8" alt="ConvNeXt-Tiny">
  <img src="https://img.shields.io/badge/Albumentations-Augmentation-blueviolet" alt="Albumentations">
  <img src="https://img.shields.io/badge/OpenCV-Image%20Processing-5C3EE8?logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/scikit--learn-Evaluation-F7931E?logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Task-Glaucoma%20Classification-00897B" alt="Glaucoma Classification">
</p>

---

## Overview

Glaucoma is a progressive optic neuropathy that can lead to irreversible vision loss. This project investigates the use of deep learning for **automated glaucoma classification from retinal fundus images**.

The proposed pipeline is based on an **ImageNet-pretrained ConvNeXt-Tiny** architecture and combines retinal image enhancement, data augmentation, class-imbalance handling, progressive transfer learning, mixed-precision training, early stopping, and validation-based decision threshold optimization.

The final model achieved a **97.15% ROC AUC**, **95.88% sensitivity**, **91.19% F1-score**, and **91.16% accuracy** on the independent test set.

> **Research disclaimer:** This project was developed for academic and research purposes. The model has not undergone external clinical validation and must not be used as a medical diagnostic system or as a substitute for professional medical assessment.

---

## Results

The best ConvNeXt-Tiny checkpoint was selected according to the **validation F1-score**.

The best model was obtained at **epoch 52**, reaching a validation F1-score of **93.90%**.

Following model selection, the classification threshold was optimized exclusively on the validation set. The selected threshold of **0.25** was then fixed before the final evaluation on the independent test set.

### Final Test Performance

| Metric | Result |
|---|---:|
| **ROC AUC** | **97.15%** |
| **Recall / Sensitivity** | **95.88%** |
| **F1-score** | **91.19%** |
| **Accuracy** | **91.16%** |
| Precision | 86.93% |
| Test Loss | 0.3481 |
| Selected Threshold | 0.25 |
| Best Validation F1 | 93.90% |
| Best Epoch | 52 |

The model achieved a sensitivity of **95.88%**, indicating that it identified most glaucoma-positive samples contained in the test set.

The **97.15% ROC AUC** indicates strong discrimination between the two classes across classification thresholds.

These results represent experimental performance on the dataset used in this project and should not be interpreted as evidence of real-world clinical diagnostic performance.

---

## Deep Learning Pipeline

```text
Retinal Fundus Image
        │
        ▼
Resize to 224 × 224
        │
        ▼
CLAHE Enhancement
        │
        ├──────────────────────────┐
        │                          │
        ▼                          ▼
    Training                Validation / Test
        │                          │
        ▼                          ▼
Data Augmentation           Deterministic
(Albumentations)            Preprocessing
        │                          │
        └────────────┬─────────────┘
                     │
                     ▼
          ImageNet Normalization
                     │
                     ▼
       Pretrained ConvNeXt-Tiny
                     │
                     ▼
        Custom Classification Head
                     │
                     ▼
           Binary Classification
                     │
                     ▼
      Validation Threshold Search
                     │
                     ▼
           Threshold = 0.25
                     │
                     ▼
          Final Test Evaluation
```

---

## Dataset

The final experiment contains **10,040 retinal fundus images** divided into predefined training, validation, and test sets.

| Split | Images | Class 0 | Class 1 |
|---|---:|---:|---:|
| Training | 8,354 | 4,219 | 4,135 |
| Validation | 770 | 385 | 385 |
| Test | 916 | 479 | 437 |
| **Total** | **10,040** | **5,083** | **4,957** |

The dataset follows the `torchvision.datasets.ImageFolder` directory structure:

```text
Fusion/
│
├── train/
│   ├── 0/
│   └── 1/
│
├── val/
│   ├── 0/
│   └── 1/
│
└── test/
    ├── 0/
    └── 1/
```

The dataset is **not included in this repository**.

The exact original datasets, licenses, and required academic citations used to construct the local `Fusion` dataset should be documented according to the corresponding data sources before redistributing any medical images.

---

## Model Architecture

### ConvNeXt-Tiny

The final model is based on **ConvNeXt-Tiny** initialized with ImageNet-pretrained weights provided by `torchvision`.

The original classification layer is replaced with a custom classification head:

```text
ConvNeXt-Tiny
ImageNet-pretrained backbone
        │
        ▼
      Flatten
        │
        ▼
    LayerNorm
        │
        ▼
   Dropout (0.40)
        │
        ▼
 Linear (2 classes)
        │
        ▼
 Binary Prediction
```

The notebook also includes an implementation of **EfficientNetV2-S** as an alternative architecture.

All results reported in this README correspond to **ConvNeXt-Tiny**.

---

## Image Preprocessing

All retinal fundus images are resized to:

```text
224 × 224
```

**CLAHE (Contrast Limited Adaptive Histogram Equalization)** is applied to improve local image contrast before classification.

The images are then normalized using the standard ImageNet mean and standard deviation expected by the pretrained ConvNeXt network.

---

## Data Augmentation

Training augmentation is implemented using **Albumentations**.

The training pipeline includes:

- CLAHE
- Horizontal flipping
- Vertical flipping
- Random rotation
- Brightness and contrast variations
- Hue, saturation, and value variations
- Gaussian noise
- Coarse dropout
- ImageNet normalization
- Tensor conversion

The validation and test pipelines remain deterministic:

```text
Resize
   │
   ▼
CLAHE
   │
   ▼
ImageNet Normalization
   │
   ▼
Tensor
```

Random training augmentation is therefore never applied during validation or final test evaluation.

---

## Transfer Learning

The model uses **transfer learning** rather than training the ConvNeXt network from random initialization.

Training starts from weights learned on ImageNet.

During the initial stage, most of the pretrained backbone is frozen while the final feature block and custom classification head remain trainable.

This allows the model to adapt higher-level representations to retinal fundus images while preserving useful pretrained visual features.

---

## Progressive Fine-Tuning

The training process uses a two-stage fine-tuning strategy.

### Stage 1 — Partial Fine-Tuning

During the initial training stage:

```text
ConvNeXt backbone
├── Earlier feature blocks → Frozen
├── Final feature block    → Trainable
└── Classification head    → Trainable
```

The initial learning rate is:

```text
5e-5
```

### Stage 2 — Full Fine-Tuning

At **epoch 30**, the entire ConvNeXt backbone is unfrozen.

```text
Epoch 30
   │
   ▼
Unfreeze complete backbone
   │
   ▼
Reduce learning rate
   │
   ▼
End-to-end fine-tuning
```

The fine-tuning learning rate is:

```text
1e-5
```

This progressive strategy allows the newly initialized classification layers to adapt before applying smaller updates across the complete pretrained network.

---

## Training Configuration

| Parameter | Value |
|---|---:|
| Architecture | ConvNeXt-Tiny |
| Pretraining | ImageNet |
| Input Size | 224 × 224 |
| Batch Size | 32 |
| Maximum Epochs | 300 |
| Initial Learning Rate | `5e-5` |
| Fine-Tuning Learning Rate | `1e-5` |
| Weight Decay | `1e-5` |
| Label Smoothing | `0.1` |
| Early Stopping Patience | 10 |
| Full Backbone Unfreezing | Epoch 30 |
| Model Selection Metric | Validation F1 |
| Random Seed | 42 |
| Mixed Precision | PyTorch AMP |

---

## Optimization

The model is optimized using **AdamW**.

A `ReduceLROnPlateau` learning-rate scheduler monitors validation performance and reduces the learning rate when improvement stagnates.

The training objective is based on:

```text
CrossEntropyLoss
```

with:

- class weighting;
- label smoothing (`0.1`).

Automatic Mixed Precision (**AMP**) is enabled when CUDA is available to improve GPU training efficiency and reduce memory consumption.

---

## Class Imbalance Handling

Although the final dataset is relatively balanced, the training pipeline includes mechanisms for handling differences in class frequency.

### Weighted Random Sampling

A `WeightedRandomSampler` adjusts the sampling probability of training examples according to their class frequency.

### Weighted Loss

Class weights are independently calculated from the **training set** and incorporated into the cross-entropy loss.

Validation and test class distributions are not used when computing training weights.

---

## Model Selection and Early Stopping

Model selection is performed exclusively using the **validation F1-score**.

Whenever validation performance improves, the corresponding model weights are saved as the best checkpoint.

Early stopping terminates training when validation performance does not improve for the configured patience period.

The final best checkpoint was obtained at:

```text
Epoch: 52
Validation F1: 93.90%
```

---

## Decision Threshold Optimization

Binary classifiers commonly use a probability threshold of `0.50`.

In this project, the operating threshold is instead selected using predictions from the **validation set**.

Candidate thresholds are evaluated according to validation performance:

```text
Validation predictions
        │
        ▼
Candidate thresholds
        │
        ▼
F1-score evaluation
        │
        ▼
Best validation threshold
        │
        ▼
       0.25
        │
        ▼
Threshold fixed
        │
        ▼
Independent test evaluation
```

The final selected threshold was:

```text
0.25
```

The test set was **not used to select this threshold**.

This separation is important because optimizing the threshold directly on test results would introduce test-set leakage.

---

## Evaluation Metrics

The model is evaluated using several complementary classification metrics:

- Accuracy
- Precision
- Recall / Sensitivity
- F1-score
- ROC AUC
- Cross-entropy loss
- Confusion matrix

Sensitivity is particularly relevant to this experimental task because a false negative corresponds to a glaucoma-positive image that the classifier fails to identify.

---

## Repository Structure

The recommended repository structure is:

```text
glaucoma-detection/
│
├── notebook/
│   └── glaucoma_detection.ipynb
│
├── results/
│   └── figures/
│
├── README.md
├── requirements.txt
└── .gitignore
```

The dataset, trained model checkpoints, Python environments, and temporary files are intentionally excluded from version control.

---

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\activate
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

A CUDA-capable GPU is strongly recommended for model training.

---

## Requirements

The main dependencies are:

```text
torch
torchvision
albumentations
numpy
scikit-learn
matplotlib
seaborn
tqdm
opencv-python
jupyter
```

---

## Running the Project

Place the dataset using the expected structure or modify `DATA_ROOT` inside the notebook.

Then open:

```text
notebook/glaucoma_detection.ipynb
```

using Jupyter Notebook or Google Colab and execute the cells sequentially.

---

## Google Colab

The final experiment was trained using **Google Colab** with the dataset stored in Google Drive.

Google Drive can be mounted with:

```python
from google.colab import drive

drive.mount("/content/drive")
```

The dataset path used during the experiment was:

```text
/content/drive/MyDrive/Fusion
```

The best checkpoint was saved to:

```text
/content/drive/MyDrive/Fusion/checkpoints/best_glaucoma_convnext.pth
```

The trained checkpoint is intentionally excluded from this repository.

---

## Reproducibility

The experiment uses a fixed random seed:

```python
SEED = 42
```

Random seeds are configured for:

- Python
- NumPy
- PyTorch
- CUDA, when available

The training, validation, and test datasets remain independent.

Random augmentation is restricted to the training pipeline, while validation and test preprocessing remain deterministic.

Exact numerical reproduction may nevertheless vary depending on:

- PyTorch version;
- CUDA version;
- cuDNN version;
- GPU architecture;
- underlying GPU operations.

---

## Technology Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Deep Learning | PyTorch, Torchvision |
| Architecture | ConvNeXt-Tiny |
| Alternative Architecture | EfficientNetV2-S |
| Image Augmentation | Albumentations |
| Image Processing | OpenCV, CLAHE |
| Machine Learning | scikit-learn |
| Numerical Computing | NumPy |
| Visualization | Matplotlib, Seaborn |
| Development | Jupyter Notebook, Google Colab |
| GPU Acceleration | CUDA, PyTorch AMP |

---

## Limitations

The results reported in this repository are specific to the dataset and experimental split used in this project.

Several limitations should be considered:

- No external clinical dataset was used for final validation.
- No prospective clinical evaluation was performed.
- Performance may vary across different retinal cameras and acquisition protocols.
- Image quality differences may affect predictions.
- Performance may vary across institutions and patient populations.
- The probability threshold of `0.25` is an experimentally optimized operating point, not a clinically established diagnostic threshold.
- High performance on the current test set does not guarantee equivalent performance in real-world clinical environments.

The model should therefore be considered a **research prototype rather than a clinical diagnostic system**.

---

## Future Work

Potential extensions of this project include:

- external validation on independent retinal datasets;
- patient-level evaluation where applicable;
- Grad-CAM and other explainability methods;
- model calibration analysis;
- sensitivity-specificity operating point analysis;
- comparison with additional modern vision architectures;
- cross-dataset generalization experiments;
- lightweight architectures for deployment;
- prospective clinical validation.

---

## Authors

- **Sirem Kaci**


---

## Medical Disclaimer

This software, source code, trained models, and experimental results are provided exclusively for **research and educational purposes**.

The system is **not intended to diagnose glaucoma or any other medical condition**.

It must not replace examination, interpretation, diagnosis, or clinical judgment by qualified healthcare professionals.

---

## License

No software license has currently been specified for this project.

An appropriate open-source license should be selected before granting permission for reuse, modification, or redistribution of the source code.

---

## Acknowledgments

This project was developed using open-source tools and libraries from the Python machine-learning and computer-vision ecosystem, including **PyTorch, Torchvision, Albumentations, OpenCV, NumPy, scikit-learn, Matplotlib, and Seaborn**.

The project was trained using **Google Colab** and GPU acceleration through **CUDA**.