ZapShield: Smart Grid Fault Detection & Line Localization
ZapShield is an end-to-end, AI-powered system for detecting power grid faults and precisely localizing affected lines in multi-phase smart grids. Developed for Smart India Hackathon 2025, ZapShield combines physics-aligned simulation, engineered feature extraction, and a hybrid CatBoost+XGBoost ML ensemble to robustly classify faults (normal, switch_off, line_break, transient) and identify the responsible line (L1/L2/L3) from raw time-series signals.

Key Features
Automated Data Generation: Realistic three-phase signal simulation with controlled noise, harmonics, phase lag and domain randomization for robust model training.

Advanced Feature Engineering: Time, frequency, wavelet, phase-lag, unbalance, and harmonic features extracted per window to capture fault and line fingerprints.

Hybrid ML Pipeline: CatBoost multiclass model for fault classification; CatBoost + XGBoost ensemble for line localization with soft-voting, probability alignment, and StandardScaler preprocessing.

Robust Evaluation: Early stopping, 5-fold cross-validation, held-out and domain-shifted test sets to demonstrate stability and guard against overfitting.

Secure & Reproducible: Strict feature schema matching, model artifact signing/versioning, and input checks for reliable inference.

Ready-to-Deploy Demo: Real-time inference pipeline, majority-vote smoothing, and actionable UI output for integration into monitoring dashboards.

Project Structure
data_generation.py → Physics-inspired signal simulation and labeled dataset creation.

feature_extraction.py → Window-level feature extraction using numpy, scipy, and pywt.

train_models.py → Fault and line model training, validation, cross-validation, and artifact saving.

demo.py → End-to-end real-time inference aligned with training pipeline.

SIH 2025 Achievement
Presented at Smart India Hackathon 2025, ZapShield ranked 130th out of 600+ teams nationwide, recognizing our system's novelty, robustness, and real-world applicability.
![1759211072787](https://github.com/user-attachments/assets/0972cdc4-50f5-47f3-acf6-cd91c7bd7644)
![1759211072361](https://github.com/user-attachments/assets/cf372b26-0dd0-44c0-8790-c5e0662e228d)
![1759211072360](https://github.com/user-attachments/assets/77a7fbc4-7465-4a27-a469-53c67cfe3f32)



