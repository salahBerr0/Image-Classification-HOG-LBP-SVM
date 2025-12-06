# 🖼️ Image Classification: HOG/LBP + SVM Comparison

A comparative study of traditional computer vision features (HOG vs LBP) with SVM classifier for image classification.

## 🎯 Key Features

- **Feature Extraction**: Implementation of HOG and multi-channel LBP feature extractors
- **Machine Learning**: SVM classification with RBF and Sigmoid kernels
- **Comprehensive Evaluation**: 5-fold cross-validation, multiple metrics, confusion matrices
- **Visualization**: Feature maps, performance comparisons, training history
- **Modular Design**: Clean, reusable code structure

## 📈 Results Preview

| Method | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| HOG + SVM | 0.85 | 0.86 | 0.85 | 0.85 |
| LBP + SVM | 0.78 | 0.79 | 0.78 | 0.78 |

*HOG features significantly outperform LBP for this image classification task.*

## 🛠️ Installation

# Install dependencies
pip install -r requirements.txt

### Option 1: Run in Google Colab (Recommended)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]( https://colab.research.google.com/drive/1RCfvZOg7B_wpdObyz-yMnKQv60FEtFja?usp=sharing)

### Option 2: Run Locally

```bash
# 1. Clone repository
git clone https://github.com/salahBerr0/Image-Classification-HOG-LBP-SVM.git
cd IMAGE-CLASSIFICATION-HOG-LBP-SVM

# 2. Download dataset
python scripts/download_data.py
# Follow the printed instructions to download from Google Drive

# 3. Run analysis
jupyter notebook notebooks/main_analysis.ipynb