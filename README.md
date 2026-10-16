# Resource-Constrained CNN for Edge Image Classification

**EN3150 Pattern Recognition: Assignment 03**
Department of Electronic and Telecommunication Engineering, University of Moratuwa

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Status](https://img.shields.io/badge/status-in%20progress-yellow)

> Design, train and compare lightweight convolutional neural networks for low-resolution image classification on edge devices (microcontrollers, Raspberry Pi, edge nodes), and benchmark them against fine-tuned lightweight state-of-the-art networks.

---

## Group Name - Pattern Pioneers 


| Name | Index No. |
|------|-----------|
| Pathirana P.T.S. | 230468V |
| Prishmika H.W.N. | 230500N |
| Umayangana M.V.I. | 230657B |
| Tharaka A.K.P.S. | 230635G |

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Models](#models)
4. [Optimizer Study](#optimizer-study)
5. [Fine-Tuned SOTA Networks](#fine-tuned-sota-networks)
6. [Results](#results)
7. [Repository Structure](#repository-structure)
8. [Getting Started](#getting-started)
9. [Reproducibility](#reproducibility)
10. [Assignment Coverage](#assignment-coverage)
11. [Report and Submission](#report-and-submission)
12. [Academic Integrity](#academic-integrity)
13. [References](#references)

---

## Project Overview

Heavy architectures such as VGG or standard ResNet are often impractical on edge hardware because of strict memory and compute limits. This project studies that trade-off directly by building and comparing four networks on the same data splits:

| Network | Type | Purpose |
|---------|------|---------|
| **Model A** | Custom standard CNN (Conv + MaxPool) | Baseline |
| **Model B** | Custom depthwise-separable CNN, **under 100,000 trainable parameters** | Resource-constrained model |
| **MobileNetV2** | Pretrained, fine-tuned | Lightweight SOTA |
| **ShuffleNetV2** | Pretrained, fine-tuned | Lightweight SOTA |

The central question is the balance between **accuracy, memory footprint and computational cost**.

---

## Dataset

| Property | Value |
|----------|-------|
| Dataset | EuroSAT (RGB) `<update if you use a different dataset>` |
| Classes | 10 land-use / land-cover classes |
| Total images | about 27,000 |
| Image size | 64 x 64 x 3 (no downscaling needed) |
| Split | 70% train / 15% validation / 15% test, stratified |
| Split seed | `<SEED>`, saved to `outputs/splits.npz` |

**Preprocessing**

- Normalisation statistics are computed from the **training set only**.
- Random horizontal flip is applied to **training batches only**.
- The same saved split indices are reused by every notebook, so all models are evaluated on an identical test set.

> The group's dataset entry has been added to the course dataset sheet: `<yes/no>`.

---

## Models

### Model A: Standard CNN

Four blocks of `Conv 3x3 -> ReLU -> MaxPool 2x2` with 32, 64, 128 and 256 filters, followed by `FC 128 (dropout)` and `FC 10`.

| Metric | Value (hand-calculated) |
|--------|-------------------------|
| Trainable parameters | 914,122 |
| MACs per image | about 60.7 M |

The first fully connected layer holds roughly 57% of all parameters, which is the main inefficiency Model B removes.

### Model B: Depthwise-Separable CNN

Five depthwise-separable blocks (depthwise 3x3 + pointwise 1x1) with BatchNorm and ReLU6, followed by global average pooling and a single FC layer.

| Metric | Value (hand-calculated) |
|--------|-------------------------|
| Trainable parameters | 67,179 (limit: 100,000, enforced by an `assert`) |
| MACs per image | about 11.8 M (roughly 5x fewer than Model A) |

Model B also differs from Model A in its use of BatchNorm, ReLU6 and a global-average-pooling head, not only in the convolution type. This is discussed in the report's trade-off section.

### Activation choice

ReLU and ReLU6 are used instead of sigmoid, tanh or swish. They need only a comparison and a clamp, with no exponentials, which suits integer and fixed-point hardware. ReLU6 also bounds activations, which helps quantisation.

---

## Optimizer Study

All runs use Model A for 20 epochs from identical initial weights.

| Run | Optimizer | Learning rate |
|-----|-----------|---------------|
| 1 | Adam | 1e-3 |
| 2 | SGD | 1e-2 |
| 3 | SGD + momentum 0.9 | 1e-2 |

An optional momentum sweep (0, 0.5, 0.9, 0.99, and 0.99 with a reduced learning rate) is controlled by the `RUN_SWEEP` flag in the notebook. A toy plot of momentum on a narrow valley illustrates the effect for the report.

---

## Fine-Tuned SOTA Networks

Both pretrained networks use ImageNet weights, 64 x 64 inputs and ImageNet normalisation.

1. **Phase 1:** train only the new classifier head (3 epochs).
2. **Phase 2:** fine-tune the whole network (12 epochs) with a lower backbone learning rate.

Parameter count and model size (MB) are computed automatically.

---

## Results

> Fill these tables from your own runs. All values below are placeholders.

### Custom models (Part 4)

| Metric | Model A | Model B |
|--------|---------|---------|
| Trainable parameters | 914,122 | 67,179 |
| Model size on disk (KB) | `TBD` | `TBD` |
| Training time per epoch (s) | `TBD` | `TBD` |
| CPU latency per image (ms) | `TBD` | `TBD` |
| Test accuracy (%) | `TBD` | `TBD` |
| Macro precision | `TBD` | `TBD` |
| Macro recall | `TBD` | `TBD` |

### Optimizer comparison (Part 3, Model A)

| Optimizer | Final val. accuracy | Test accuracy | Epochs to converge |
|-----------|--------------------|---------------|--------------------|
| Adam | `TBD` | `TBD` | `TBD` |
| SGD | `TBD` | `TBD` | `TBD` |
| SGD + momentum | `TBD` | `TBD` | `TBD` |

### Custom vs. pretrained (Parts 5 and 6)

| Model | Parameters | Size | Test accuracy | Time per epoch |
|-------|-----------|------|---------------|----------------|
| Model B (custom) | 67,179 | `TBD` KB | `TBD` | `TBD` |
| MobileNetV2 | `TBD` | `TBD` MB | `TBD` | `TBD` |
| ShuffleNetV2 | `TBD` | `TBD` MB | `TBD` | `TBD` |

Figures (loss curves, confusion matrices, comparison plots) are saved in `outputs/figures/`.

---

## Repository Structure

```
.
├── a03_common.py                  # Shared data, models, training and evaluation code
├── EN3150_A03_ModelA.ipynb        # Data prep + Model A (standard CNN)
├── EN3150_A03_ModelB.ipynb        # Model B (depthwise-separable CNN)
├── EN3150_A03_Optimizers.ipynb    # Adam vs SGD vs SGD + momentum
├── EN3150_A03_MobileNetV2.ipynb   # SOTA fine-tuning 1
├── EN3150_A03_ShuffleNetV2.ipynb  # SOTA fine-tuning 2
├── EN3150_A03_Comparison.ipynb    # Final tables and figures
├── outputs/
│   ├── splits.npz                 # Saved 70/15/15 split indices
│   ├── *_results.json             # Per-model metrics
│   └── figures/                   # Plots for the report
├── report/                        # LaTeX source and final PDF
├── requirements.txt
└── README.md
```

---

## Getting Started

### Requirements

- Python 3.10 or newer
- PyTorch and torchvision
- NumPy, scikit-learn, matplotlib, seaborn, Jupyter

```bash
git clone <REPO_URL>
cd <REPO_NAME>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch torchvision numpy scikit-learn matplotlib seaborn jupyter
```

### Run order

Keep `a03_common.py` in the same folder as the notebooks, and run them in this order:

| # | Notebook | Assignment part |
|---|----------|-----------------|
| 1 | `EN3150_A03_ModelA.ipynb` | Data prep, Part 2 and 4 (Model A) |
| 2 | `EN3150_A03_ModelB.ipynb` | Part 2 and 4 (Model B) |
| 3 | `EN3150_A03_Optimizers.ipynb` | Part 3 |
| 4 | `EN3150_A03_MobileNetV2.ipynb` | Part 5 |
| 5 | `EN3150_A03_ShuffleNetV2.ipynb` | Part 5 |
| 6 | `EN3150_A03_Comparison.ipynb` | Part 4 table and Part 6 |

Notebook 1 must run first because it caches the dataset and writes the shared split file. Pretrained weights download automatically the first time notebooks 4 and 5 run, so an internet connection is needed once.

### Hardware note

All experiments were designed for a CPU-only laptop. Typical cost: Models A and B take tens of seconds per epoch, while the fine-tuned MobileNetV2 and ShuffleNetV2 take several minutes per epoch. Depthwise convolutions are comparatively slow on CPU, so Model B's time per epoch does not drop in proportion to its MAC reduction.

---

## Reproducibility

- Fixed random seed for the split, weight initialisation and data shuffling.
- One saved split file shared by all models.
- Identical initial weights for the three optimizer runs.
- Best-validation checkpoint is used for test evaluation.

---

## Assignment Coverage

| Part | Marks | Where |
|------|-------|-------|
| 1. Data preparation | n/a | Notebook 1 |
| 2. Custom architecture design | 20 | Notebooks 1, 2 |
| 3. Optimizer selection and tuning | 15 | Notebook 3 |
| 4. Training and evaluation | 25 | Notebooks 1, 2, 6 |
| 5. SOTA fine-tuning | 20 | Notebooks 4, 5 |
| 6. Final comparison and discussion | 20 | Notebook 6 and report |

---

## Report and Submission

- The report (PDF) contains the interpretation and discussion, and includes the names and index numbers of all group members.
- Submission file name: `Yourgroupno_A03_EN3150`. Report and code are uploaded to Moodle separately; one submission per group.
- This repository's commit history documents the work done over time.

---

## Academic Integrity

This work was completed by the group named above for EN3150. Third-party resources (PyTorch, torchvision, pretrained ImageNet weights, the EuroSAT dataset) are used under their respective licences and cited below.

---

## References

1. P. Helber, B. Bischke, A. Dengel and D. Borth, "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification," *IEEE JSTARS*, 2019.
2. M. Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks," *CVPR*, 2018.
3. N. Ma et al., "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design," *ECCV*, 2018.
4. A. Howard et al., "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications," 2017.
5. Y. LeCun, L. Bottou, Y. Bengio and P. Haffner, "Gradient-based learning applied to document recognition," *Proc. IEEE*, 1998.
6. PyTorch documentation: https://pytorch.org/docs/

---

## License

For academic use as part of the EN3150 course at the University of Moratuwa.
