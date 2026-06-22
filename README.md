# NV-center-code: Deep Learning for NV Center Power Prediction

Deep learning-based laser and microwave power prediction for Nitrogen-Vacancy (NV) center quantum sensing. Uses dual-stream convolutional neural networks to decode ODMR (Optically Detected Magnetic Resonance) spectra into precise power estimates, with comprehensive reliability analysis via Monte Carlo uncertainty quantification.

## Overview

NV centers in diamond are promising quantum sensors, but their optical readout is sensitive to both laser power and microwave (MW) power levels. This project builds deep learning models that predict power conditions directly from ODMR spectra, enabling:

- **Power classification** — 9-group classification (3 laser × 3 microwave levels) from spectra alone
- **Power regression** — continuous prediction of laser and MW power levels
- **Reliability analysis** — Monte Carlo sensitivity quantification for real-world deployment
- **Virtual sensor verification** — validated as virtual instruments against physical sensors

## Architecture

```
ODMR Spectrum (400 points)
         │
    ┌────┴────────────┐
    ▼                 ▼
Shape Stream     Amplitude Stream
(1D CNN)         (FC layers)
Conv1D 16→32→64  Linear 400→128→32
    │                 │
    ▼                 ▼          Aux (laser%, mW dBm)
  Fusion ─────────────┼─────────────────┘
    │
    ▼
Prediction (1–9 groups or power values)
```

**DualStreamNet**: Two parallel streams — a 1D CNN learns spectral shape features, while fully-connected layers capture amplitude statistics. Auxiliary inputs (laser/microwave settings) optionally inform the fusion layer.

**Four model variants:**
| Model | Aux Input | Predicts |
|-------|-----------|----------|
| `model_blind.pth` | None | Power from spectrum only |
| `model_laser.pth` | Laser % | MW power |
| `model_mw.pth` | MW dBm | Laser power |
| `model_full.pth` | Laser + MW | Power (full info) |
| `model_power_classifier_9groups.pth` | — | 9-group classification |

## Key Results

- **9-group classification** across 3 laser levels (20%/50%/100%) × 3 microwave levels (−5/0/+5 dBm)
- **Monte Carlo reliability analysis** with REC (Regression Error Characteristic) curves
- **Noise robustness verification** (Fig. 6, IEEE-format publication figure)
- **Virtual sensor validation** against 9 physical sensor groups
- **Temperature coefficient calibration** for real-world environmental correction

## Project Structure

```
NV-center-code/
├── 激光微波功率预测.py          # Main: power classification (9-group)
├── 四个模型的深度学习.py         # DualStreamNet training (4 model variants)
├── 四个模型测试.py              # Model evaluation scripts
├── 四模型稳定性测试.py           # Stability testing across models
├── 四模型测试（1）.py           # Extended testing
├── 蒙特卡洛灵敏度分析.py         # Monte Carlo sensitivity analysis
├── 数据集构件图与物理先验.py     # Dataset visualization + physical priors
├── 虚拟仪器结果代码.py           # Virtual sensor verification
├── 功率标定.py                  # Power calibration
├── 全功率组_温度系数标定组.py    # Temperature coefficient calibration
├── 九组功率数据加强.py           # Data augmentation for 9 groups
├── 预测后的功率组再次预测温度.py # Temperature prediction from power groups
├── 加上双洛伦兹拟合的模型学习率.py  # Dual Lorentzian + learning rate study
├── 评估代码.py                  # Evaluation harness
├── model_*.pth                  # Trained model weights (5 files)
├── *.png/*.pdf                  # Result figures and charts
├── *.npy                        # Fitted calibration coefficients
└── 训练集/测试集 .pth            # Dataset files
```

## Technical Stack

- **PyTorch** — model definition, training, inference
- **NumPy / SciPy** — data processing, curve fitting
- **scikit-learn** — classification metrics
- **Matplotlib** — all visualizations
- **CUDA** — GPU acceleration (auto-detected)

## Datasets

- Training: ~50,000 aligned ODMR spectra (`训练集_50k_aligned.pth`)
- Testing: ~5,000 aligned ODMR spectra (`测试集_5k_aligned.pth`)
- Each sample: 400-point frequency sweep (2858–2878 MHz) + auxiliary labels (laser %, MW dBm)

## Setup & Usage

```bash
# Install dependencies
pip install torch numpy scipy scikit-learn matplotlib pandas

# Train the 9-group power classifier
python 激光微波功率预测.py

# Train the 4 model variants (blind/laser/mw/full)
python 四个模型的深度学习.py

# Run evaluation
python 评估代码.py

# Run Monte Carlo sensitivity analysis
python 蒙特卡洛灵敏度分析.py
```

## Citation

If you use this code or models in your research, please reference this repository.

---

*Part of NV center quantum sensing research — bridging deep learning and quantum metrology.*
