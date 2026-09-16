# Image Classification with CNN (Built from Scratch)

This repo is an ongoing collection of Convolutional Neural Networks **built from scratch in PyTorch** (no pre-trained/transfer-learning backbones), each trained end-to-end on a different image dataset.

Every model follows the same pipeline: load raw images → resize/transform to tensors → build a custom `Dataset`/`DataLoader` → define a CNN class → train with a loss function → evaluate on held-out validation/test splits.

This README is updated with a new entry every time a CNN is trained on a new dataset — see [Results Summary](#-results-summary) for the running scoreboard and [Adding a New Dataset](#-adding-a-new-dataset) for the template to copy when you add one.

---

## 📁 Repository Structure

```
Image-Classification-with-CNN/
├── CNN-with-1k-images_.ipynb   # Cats vs Dogs 1k images
├── Intel_Image_Classification_CNN_.ipynb # 25k images
└── README.md
```

> Update this tree every time a new notebook is added.

---

## ⚙️ Requirements

```
torch
torchvision
scikit-learn
Pillow
matplotlib
numpy
```

```bash
pip install torch torchvision scikit-learn pillow matplotlib numpy
```

Notebooks are written for **Google Colab**, with datasets mounted from Google Drive (`drive.mount('/content/drive')`), and use CUDA automatically when available.

---

## 📊 Results Summary

| # | Dataset | Task | Classes | Val Acc | Test Acc | Notebook |
|---|---|---|---|---|---|---|
| 1 | [Cats and Dogs Mini Dataset](https://www.kaggle.com/datasets/aleemaparakatta/cats-and-dogs-mini-dataset) | Binary classification | 2 | 59.33% | 68.67% | `notebooks/cat_dog_classifier.ipynb` |

> Add a new row here every time a new dataset is trained.

---

## 1. Cats vs Dogs

**Dataset:** [Cats and Dogs Mini Dataset (Kaggle)](https://www.kaggle.com/datasets/aleemaparakatta/cats-and-dogs-mini-dataset) — 500 cat + 500 dog images.

**Pipeline**
1. Images read from `cats_set/` and `dogs_set/`, labeled `0` (cat) / `1` (dog).
2. Resized to `224x224`, converted to tensors via `torchvision.transforms`.
3. Custom `CatDogDataset(Dataset)` wraps paths + labels.
4. Split: 70% train / 15% val / 15% test, stratified.
5. `DataLoader` — batch size `32`, train shuffled.
6. `Making Prediction`
7. `Loss function`
8. `Training Loop`

**Architecture**
```
Input (3, 224, 224)
  → Conv2d(3, 16, k=3, p=1) → ReLU → MaxPool2d(2)   → (16, 112, 112)
  → Conv2d(16, 32, k=3, p=1) → ReLU → MaxPool2d(2)  → (32, 56, 56)
  → Flatten                                          → (32*56*56,)
  → Linear(100352, 128) → ReLU
  → Linear(128, 1)                                   → logit
```

**Training** — `BCEWithLogitsLoss` + `Adam(lr=0.001)`, 20 epochs.

**Results**

| Split | Accuracy | Avg Loss |
|---|---|---|
| Validation | 59.33% | 1.9158 |
| Test | 68.67% | 1.2724 |

**Notes** — Training loss fell to ~0 by epoch 20 while val/test accuracy stayed in the .60s, The overfitting nature of the model is known (might be fixed with pumping more training data) and any possible suggesation for improvemnt shall be appreciated.

---

## ➕ Adding a New Dataset

When you train a CNN on a new dataset, do three things:

1. **Add a row to [Results Summary](#-results-summary)**.
2. **Add a new numbered section below**, copying this template:

    ## <emoji> N. <Dataset Name>

    **Dataset:** [<name>](<link>) — <size / class breakdown>.

    **Pipeline**
    1. ...
    2. ...

    **Architecture**
    ```
    <input shape>
      → ...
      → output
    ```

    **Training** — <loss fn> + <optimizer>(lr=<x>), <N> epochs.

    **Results**

    | Split | Accuracy | Avg Loss |
    |---|---|---|
    | Validation | ...% | ... |
    | Test | ...% | ... |

    **Notes** — <overfitting/underfitting nature are known, ideas for improvement may be helpfull>.

---
