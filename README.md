# NV-Center Thermometry: Physics-Prior-Guided Cascaded Neural Network

Code and dataset for the manuscript *"Physics-Prior-Guided Cascaded Neural
Network for Robust NV Center Thermometry under Laser and Microwave Power
Perturbations"* (submitted to the IEEE Sensors Journal).

**Authors:** Tianyu Chu, Libin Huang (corresponding author), Liye Zhao, Yuhai Wang
School of Instrument Science and Engineering, Southeast University, Nanjing 210096, China

## Repository layout

| Path | Contents |
|---|---|
| `scripts/` | Pipeline: joint 3x3 power-state calibration, ResNet-1D power classifier, dual-stream regressor, evaluation, Monte-Carlo, figure scripts |
| `figure_scripts/` | Graphical-abstract generator |
| `data/raw_spectra_seed104/` | 104 seed CW-ODMR spectra covering the 3x3 laser x microwave power-state matrix and 40-80 degC calibration range |
| `data/raw_spectra_validation110/` | 110-spectrum real-world validation set |
| `results/` | Per-group calibration (`fulldata_groups.json`), 110-spectrum evaluation, bootstrap/Monte-Carlo records, 9x9 confusion matrix, fits (JSON/CSV) |
| `models_weights/` | Trained network weights (`.pth`), calibrated per-state coefficients (`.npy`/CSV/JSON), and the aligned 50k synthetic training set |
| `environment/` | `requirements.txt` (Python >= 3.10; CPU inference is sufficient for all evaluation scripts) |

## Reproducing the headline numbers

1. `pip install -r environment/requirements.txt`
2. Per-state calibration of the apparent temperature coefficient: scripts operating on `data/raw_spectra_seed104/` reproduce `results/fulldata_groups.json` (k_i range -86.9 to -96.0 kHz/K, mean -90.2, R2 0.9896-0.9994).
3. Evaluation on `data/raw_spectra_validation110/` reproduces the 0.31 degC temperature MAE of the proposed cascade (vs 2.18 / 0.64 degC for Single/Double-Lorentzian fitting) and the 9x9 confusion matrix (99.1% power-state classification accuracy).

## Data availability

Consistent with the manuscript: the per-group calibration table, evaluation
records, trained weights, and datasets are available from the corresponding
author (Libin Huang, huanglibin@seu.edu.cn) on reasonable request.
