# Heart Reader — Technical Report

## Deep Learning for Automated 12-Lead ECG Analysis

**Author:** Donyes Hsairi  
**Project:** Heart Reader  
**Version:** Final experimental and web-application iteration  
**Date:** September 2026

---

## Abstract

Heart Reader is a research prototype for automated multi-label classification of 12-lead electrocardiograms. The project combines raw ECG signals with structured PTB-XL+ features and evaluates three attention-enhanced one-dimensional convolutional architectures: Inception1D, SE-ResNet1D, and XResNet1D101.

A dedicated ECG robustness strategy is incorporated into training through **Stationary Wavelet Transform (SWT) augmentation, curriculum-based augmentation scheduling, and masked ECG training**. Conventional Gaussian-noise and amplitude perturbations are also used.

The predictions of the three trained models are subsequently stacked into a 15-dimensional representation and passed to a one-vs-rest LogisticRegressionCV meta-learner. The final notebook reports a **test Macro-AUC of 0.9290** on PTB-XL Fold 10.

The project further demonstrates deployment-oriented engineering through ONNX export and a FastAPI web application for ECG upload and analysis.

---

# 1. Introduction

Automated ECG interpretation is an important application of deep learning because ECG recordings contain complex temporal and morphological patterns that can be difficult to model using handcrafted rules alone.

The objective of Heart Reader is to develop a multi-label deep-learning system capable of predicting five diagnostic superclasses:

- **NORM** — Normal ECG
- **MI** — Myocardial Infarction
- **STTC** — ST/T Change
- **CD** — Conduction Disturbance
- **HYP** — Hypertrophy

The project extends a conventional ECG classification pipeline with multimodal feature fusion, attention mechanisms, ECG-specific augmentation, ensemble learning, and deployment through ONNX and FastAPI.

---

# 2. Dataset

## 2.1 PTB-XL

The project uses the PTB-XL ECG dataset and its predefined stratified folds.

The final experimental split is:

| Fold | Role |
|---|---|
| 1–8 | Training |
| 9 | Validation / meta-learning |
| 10 | Test |

The input ECG representation is:

```text
12 leads × 1000 samples
```

corresponding to:

```text
10 seconds × 100 Hz
```

Diagnostic labels are derived from the SCP-ECG annotations using the provided diagnostic superclass mapping. A diagnostic code contributes to a superclass when its likelihood reaches the configured 50% threshold.

## 2.2 PTB-XL+

The multimodal branch incorporates **1,313 structured features** from the PTB-XL+ feature tables.

The two feature sources used by the project are:

- 12SL features
- ECGdeli features

The feature preprocessing pipeline performs training-fold-based median imputation and standardization.

---

# 3. Preprocessing and Data Balancing

## 3.1 ECG Signal

WFDB records are loaded and represented as:

```text
channels × time = 12 × 1000
```

The dataset implementation performs independent lead-wise normalization when training statistics are available.

## 3.2 Intelligent Balancing

The training data is intentionally rebalanced:

- Pure NORM records are capped at 4,000 through downsampling.
- HYP-positive records are oversampled to 4,000 when necessary.
- Dynamic class weights are calculated from the resulting training distribution.

The recorded final training dataframe contains:

```text
16,373 samples
```

---

# 4. ECG Data Augmentation

Data augmentation is one of the main methodological components of the final training pipeline.

## 4.1 Gaussian Noise

With probability:

```text
p = 0.4
```

Gaussian noise with approximately:

```text
σ = 0.03
```

is added to the ECG signal.

## 4.2 Random Amplitude Scaling

With probability:

```text
p = 0.4
```

the signal is multiplied by a random factor:

```text
0.85 – 1.15
```

This introduces amplitude variability during training.

## 4.3 Masked ECG Training

The dataset implements temporal masking through **1D cutout**.

With probability:

```text
p = 0.3
```

individual leads may receive a randomly positioned temporal segment set to zero.

The mask length is randomly selected between:

```text
50 samples
```

and:

```text
25% of the signal length
```

This forces the model to learn from complementary ECG information and improves robustness to partial signal corruption or missing waveform segments.

---

# 5. Stationary Wavelet Transform Augmentation

## 5.1 Motivation

ECG signals contain information at multiple temporal and frequency scales. Wavelet representations provide a way to perturb detailed components while preserving the general structure of the signal.

## 5.2 SWT Implementation

The final notebook applies Stationary Wavelet Transform augmentation using:

```text
Wavelet: db4
Level: 3
```

For selected training batches:

```text
ECG
 ↓
SWT
 ↓
Approximation + detail coefficients
 ↓
Noise added to detail coefficients
 ↓
Inverse SWT
 ↓
Augmented ECG
```

Noise is applied to the detail coefficients with a scale dependent on the decomposition level.

The reconstructed waveform is then passed to the neural network.

---

# 6. Curriculum-Based Augmentation Scheduling

Instead of applying a fixed SWT perturbation throughout training, the final pipeline uses a curriculum scheduler.

The scheduler linearly increases both the SWT noise strength and augmentation probability.

### Initial configuration

```text
Noise standard deviation = 0.01
Augmentation probability = 0.30
```

### Maximum configuration

```text
Noise standard deviation = 0.05
Augmentation probability = 0.60
```

The parameters evolve according to training progress:

```text
progress = epoch / total_epochs
```

with:

```text
noise = initial_noise + (max_noise - initial_noise) × progress

p = initial_p + (max_p - initial_p) × progress
```

The objective is to expose the model to progressively stronger perturbations as training progresses.

---

# 7. Deep Learning Architecture

## 7.1 Multimodal Fusion

The model has two branches.

### Signal branch

Each backbone generates a learned representation of the 12-lead ECG.

### Structured feature branch

The PTB-XL+ feature vector is processed through:

```text
1313
 ↓
256
 ↓
64
```

The two representations are fused:

```text
256-dimensional signal embedding
                +
64-dimensional feature embedding
                =
320-dimensional fused representation
```

The classification head is:

```text
320
 ↓
128
 ↓
5 logits
```

---

# 8. Attention Mechanism

The final model uses PyTorch Multi-Head Attention after the convolutional feature extractor.

Configuration:

```text
Number of attention heads = 4
batch_first = True
```

The sequence is processed as:

```text
CNN features
     ↓
Multi-Head Self-Attention
     ↓
Residual connection
     ↓
LayerNorm
     ↓
Adaptive Average Pooling
```

The attention mechanism allows the model to capture dependencies across the temporal feature sequence before global pooling.

---

# 9. Base Models

Three complementary architectures are trained.

## 9.1 Inception1D

The Inception-style architecture uses parallel temporal convolution branches to capture patterns at different scales.

## 9.2 SE-ResNet1D

The SE-ResNet architecture combines residual connections with Squeeze-and-Excitation channel recalibration.

## 9.3 XResNet1D101

The XResNet-style branch provides a third residual representation of the ECG signal.

---

# 10. Training Configuration

The final advanced training loop uses:

| Parameter | Value |
|---|---|
| Loss | ClassWeightedFocalLoss |
| Focal γ | 2.5 |
| Optimizer | AdamW |
| OneCycleLR | Enabled |
| Maximum epochs | 50 |
| Early stopping patience | 8 |
| Gradient clipping | Maximum norm 1.0 |
| Seed | 42 |
| Training device | CUDA when available |

The training loop calculates Macro-AUC for the training and validation predictions at each epoch.

---

# 11. Training Results

The best validation Macro-AUC values recorded in the final notebook are:

| Backbone | Best Validation Macro-AUC | Best Epoch |
|---|---:|---:|
| Inception1D + Attention | **0.9282** | 9 |
| SE-ResNet1D + Attention | **0.9307** | 19 |
| XResNet1D101 + Attention | **0.9295** | 14 |

The validation curves and epoch histories are stored in the `results/` directory.

---

# 12. Ensemble Meta-Learning

After training the three base models, each model produces five sigmoid probabilities.

Therefore:

```text
3 models × 5 classes = 15 meta-features
```

The meta-feature vector is:

```text
[
 Inception1D probabilities,
 SE-ResNet1D probabilities,
 XResNet1D probabilities
]
```

A one-vs-rest:

```text
LogisticRegressionCV
```

with:

```text
cv = 5
class_weight = "balanced"
```

is trained on the validation predictions.

The resulting meta-learner is then evaluated on Fold 10.

---

# 13. Final Evaluation

## 13.1 Macro-AUC

The final notebook reports:

```text
Test Macro-AUC = 0.9290
```

This is the primary performance metric reported for the final model.

## 13.2 Classification Report

The final thresholded report is:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| NORM | 0.81 | 0.92 | 0.86 | 954 |
| MI | 0.68 | 0.76 | 0.72 | 416 |
| STTC | 0.74 | 0.79 | 0.76 | 506 |
| CD | 0.80 | 0.73 | 0.76 | 497 |
| HYP | 0.34 | 0.86 | 0.48 | 222 |
| **Macro Average** | **0.67** | **0.81** | **0.72** | **2,595** |
| **Weighted Average** | **0.73** | **0.83** | **0.77** | **2,595** |

The results show high recall for NORM and HYP, while HYP has comparatively lower precision.

---

# 14. Threshold Selection Limitation

The final notebook uses:

```text
NORM = 0.5000
MI   = 0.7032
STTC = 0.6726
CD   = 0.7352
HYP  = 0.4289
```

The NORM threshold is fixed at 0.5.

The thresholds for MI, STTC, CD and HYP are selected from precision-recall curves calculated on the test set.

Consequently, the threshold-dependent precision, recall and F1 values are not a completely untouched test evaluation.

The **0.9290 Macro-AUC** remains the preferred headline metric because it is threshold-independent.

### Recommended future protocol

A stricter evaluation should:

1. Train the base models on Folds 1–8.
2. Train the meta-learner using Fold 9.
3. Optimize classification thresholds using Fold 9.
4. Freeze the thresholds.
5. Evaluate Fold 10 exactly once.

---

# 15. ONNX Deployment

The final notebook exports the attention-enabled Inception1D fusion model to:

```text
results/fusion_model.onnx
```

The exported model accepts two inputs:

```text
signal:
(batch, 12, 1000)

features:
(batch, 1313)
```

and produces:

```text
logits:
(batch, 5)
```

The ONNX export uses opset 18 and dynamic batch-size support.

---

# 16. FastAPI Web Application

The web application is implemented using FastAPI.

## 16.1 Architecture

```text
Browser
   │
   ▼
FastAPI
   │
   ▼
CSV ECG Parser
   │
   ▼
12 × 1000 ECG Signal
   +
1313 Feature Vector
   │
   ▼
ONNX Runtime
   │
   ▼
Sigmoid Probabilities
   │
   ▼
Diagnostic Results
```

## 16.2 Endpoints

### `/api/health`

Reports:

- application status
- whether the ONNX model is loaded
- available diagnostic classes
- test-sample availability

### `/api/model-info`

Provides:

- class names
- class descriptions
- model parameter count
- feature count
- ECG input dimensions
- lead names
- reported model performance

### `/api/random-sample`

Loads a random CSV ECG from the local test directory and runs the prediction pipeline.

### `/api/predict`

Accepts an uploaded CSV ECG and returns:

- filename
- signal shape
- waveform
- predicted classes
- class probabilities
- lead names

---

# 17. CSV ECG Processing

The web application supports CSV ECG input.

The parser:

1. Detects whether a header is present.
2. Converts values to numeric data.
3. Extracts the first 12 columns as ECG leads.
4. Detects additional columns as structured features when available.
5. Pads or truncates the ECG to 1,000 samples.
6. Supplies a zero-filled 1,313-feature vector when structured features are absent.

Therefore, an uploaded ECG containing only the 12 waveform leads does not provide the same structured feature information as the full PTB-XL+ training setup.

---

# 18. Application Demonstration

The repository includes:

```text
Demo.gif
Demo.mp4
```

These demonstrate the browser-based ECG analysis interface.

The interface is intended to provide a practical demonstration of how the trained deep-learning pipeline can be exposed through a user-facing application.

---

# 19. Project Organization

```text
Heart-Reader/
│
├── README.md
├── REPORT.md
│
└── Heart_Reader/
    ├── notebook/
    │   └── ecg-final.ipynb
    │
    ├── frontend/
    │   ├── app.py
    │   ├── __init__.py
    │   └── static/
    │       ├── index.html
    │       ├── app.js
    │       └── style.css
    │
    ├── results/
    │   ├── TEST_AUC.png
    │   ├── Confusion_matrix.png
    │   ├── Seuils.png
    │   ├── INCEPTION1D_curves.png
    │   ├── INCEPTION1D_epochs.png
    │   ├── SE_RESNET1D_curves.png
    │   ├── SE_RESNET1D_epochs.png
    │   ├── XRESNET1D_curves.png
    │   └── XRESNET1D_epochs.png
    │
    ├── test_files/
    ├── requirements.txt
    ├── Demo.gif
    └── Demo.mp4
```

---

# 20. Limitations

The current implementation has several limitations that should be considered when interpreting the results.

### 20.1 Test-set threshold optimization

Thresholds for four classes were selected using test-set labels. A future version should move threshold optimization to validation data.

### 20.2 Web multimodal input

The trained multimodal model expects 1,313 structured features. When an uploaded CSV contains only ECG waveform columns, the application fills the feature vector with zeros.

### 20.3 ONNX runtime validation

The notebook successfully contains the ONNX export and tests the FastAPI interface. However, the recorded notebook API test reported that the ONNX runtime session was not loaded and therefore used the application's fallback inference path during that particular test.

A complete deployment validation should confirm:

```text
/api/health
model_loaded = true
```

and compare ONNX predictions against PyTorch predictions on identical inputs.

### 20.4 Attention visualization

The application contains an attention-heatmap field, but the current implementation does not extract a learned attention map from the neural network. It should therefore not be presented as a clinically meaningful explanation.

### 20.5 Clinical validation

The model has not been presented as a clinically validated medical device and has not been externally validated on an independent clinical dataset in this project.

---

# 21. Future Work

The following improvements would strengthen the project:

1. Optimize classification thresholds exclusively on validation data.
2. Verify ONNX Runtime inference end-to-end.
3. Implement PTB-XL+ feature extraction for uploaded ECGs.
4. Add a signal-only deployment model for waveform-only uploads.
5. Replace the placeholder attention visualization with a genuine attribution method.
6. Benchmark ONNX inference latency and memory usage.
7. Investigate model calibration.
8. Perform external validation on an independent ECG dataset.
9. Add automated tests for the parser, model inference, ONNX inference, and API.
10. Investigate model compression and quantization for edge deployment.

---

# 22. Conclusion

Heart Reader demonstrates an end-to-end deep-learning workflow for multi-label 12-lead ECG classification.

The project combines:

```text
PTB-XL / PTB-XL+
        ↓
Preprocessing & intelligent balancing
        ↓
Gaussian Noise + Amplitude Scaling
        ↓
Masked ECG Training
        ↓
SWT Augmentation
        ↓
Curriculum-Based Scheduling
        ↓
Attention-Enhanced CNNs
        ↓
Multimodal Fusion
        ↓
Three-Model Ensemble
        ↓
Logistic Regression Meta-Learner
        ↓
Fold 10 Evaluation
        ↓
ONNX Export
        ↓
FastAPI Web Application
```

The final notebook reports a **0.9290 test Macro-AUC** for the stacked meta-learner.

Beyond the final metric, the project demonstrates the complete path from deep-learning experimentation to a deployable software interface, while explicitly documenting the remaining methodological and deployment limitations.

---

# 23. References

1. Wagner, P. et al. **PTB-XL, a large publicly available electrocardiography dataset.** Scientific Data, 2020.
2. Strodthoff, N. et al. **Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL.** IEEE Journal of Biomedical and Health Informatics, 2021.
3. Strodthoff, N. et al. **PTB-XL+, a comprehensive electrocardiographic feature dataset.** Scientific Data, 2023.
4. Fawaz, H. I. et al. **InceptionTime: Finding AlexNet for Time Series Classification.** Data Mining and Knowledge Discovery, 2020.
5. Hu, J., Shen, L., Sun, G. **Squeeze-and-Excitation Networks.** CVPR, 2018.
6. Lin, T.-Y. et al. **Focal Loss for Dense Object Detection.** IEEE TPAMI, 2020.

---

## Research Disclaimer

Heart Reader is a research and demonstration project. It is not a clinically validated medical device, and its predictions must not be used as a substitute for professional medical interpretation or clinical decision-making.
