# Product Specification: ECG-FPGA Accelerator

## 1. INTRODUCTION

Cardiac arrhythmias are among the leading causes of sudden cardiac death globally. Continuous, real-time electrocardiogram (ECG) monitoring is crucial for the early detection and management of these anomalies. Traditionally, high-accuracy classification of ECG signals has relied on complex deep learning models, which are computationally expensive and power-hungry, making them unsuitable for deployment on low-power, resource-constrained embedded devices like wearable monitors.

This product, the **ECG-FPGA Accelerator**, bridges the gap between clinically validated deep-learning classifiers and low-power hardware deployment. It provides a complete, end-to-end pipeline for real-time cardiac arrhythmia detection on an FPGA. The system ingests raw PhysioNet ECG signals, trains a high-precision Multi-Scale 1D Convolutional Neural Network (CNN), and employs Sensitivity-Preserving Quantization-Aware Training (SP-QAT) to compress the model to INT8 precision. Crucially, this quantization process is bound by strict clinical constraints—ensuring a maximum sensitivity drop of 2.0% per class compared to the floating-point baseline. The final output is a set of hardware-ready weight files that can be directly synthesized onto a Vivado-targeted FPGA architecture, enabling low-latency, low-power edge inference without compromising patient safety.

## 2. LITERATURE SURVEY

The development of the ECG-FPGA Accelerator builds upon extensive research in the fields of cardiology, machine learning, model compression, and hardware design. The following 15 papers represent the foundational literature surveyed to design this system:

1. **Chazal, P. D., et al. (2004).** "Automatic classification of heartbeats using ECG morphology and heartbeat interval features." *IEEE Transactions on Biomedical Engineering*, 51(7), 1196-1206. *(Foundation for AAMI inter-patient split standards)*
2. **Hannun, A. Y., et al. (2019).** "Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms using a deep neural network." *Nature Medicine*, 25(1), 65-69. *(Demonstrates CNN efficacy for 1D ECG signals)*
3. **Kiranyaz, S., et al. (2016).** "Real-Time Patient-Specific ECG Classification by 1-D Convolutional Neural Networks." *IEEE Transactions on Biomedical Engineering*, 63(3), 664-675. *(Pioneering work on 1D-CNNs for real-time ECG)*
4. **Jacob, B., et al. (2018).** "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference." *CVPR*. *(Core principles of Quantization-Aware Training)*
5. **Wang, K., et al. (2019).** "HAQ: Hardware-Aware Automated Quantization with Mixed Precision." *CVPR*. *(Hardware-specific constraints in quantization)*
6. **Esser, S. K., et al. (2019).** "Learned Step Size Quantization." *ICLR*. *(Techniques for optimizing INT8 activation and weight scaling)*
7. **Brevitas Contributors (2020).** "Brevitas: A PyTorch library for Quantization-Aware Training." *Xilinx Research*. *(The underlying framework utilized for SP-QAT)*
8. **Goldberger, A. L., et al. (2000).** "PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals." *Circulation*, 101(23). *(Source of the MIT-BIH Arrhythmia Database)*
9. **Gao, J., et al. (2020).** "An Energy-Efficient ECG Processor based on a Lightweight CNN for Real-time Arrhythmia Detection." *IEEE ISCAS*. *(Hardware acceleration considerations for ECG)*
10. **Li, Dan, et al. (2017).** "An FPGA-Based Hardware Accelerator for CNNs Using On-Chip Memories Only." *IEEE Transactions on Circuits and Systems*. *(Guidelines for memory mapping weights on FPGA)*
11. **Wu, M., et al. (2021).** "A Deep Learning Approach for ECG Arrhythmia Classification without QRS Detection." *IEEE Access*. *(Insights into sliding window multi-scale feature extraction)*
12. **Lin, X., et al. (2016).** "Fixed-Point Quantization of Deep Convolutional Networks." *ICML*. *(Analysis of precision loss vs. hardware efficiency)*
13. **Rajpurkar, P., et al. (2017).** "Cardiologist-Level Arrhythmia Detection with Convolutional Neural Networks." *arXiv preprint*. *(Advanced multi-class loss weighting strategies)*
14. **Gholami, A., et al. (2021).** "A Survey of Quantization Methods for Efficient Neural Network Inference." *IEEE CVPR*. *(Comprehensive review of post-training vs. QAT methods)*
15. **Chen, T., et al. (2020).** "Deep Learning for ECG Analysis: Benchmarks and Insights." *Physiological Measurement*. *(Evaluation metrics and clinical sensitivity importance)*

## 3. PROBLEM STATEMENT

Real-time ECG analysis on wearable or embedded devices presents a critical engineering conflict. On one hand, clinical-grade arrhythmia classification requires high-capacity deep learning models (like Multi-Scale CNNs) to handle the complex morphology of cardiac signals. These models inherently rely on 32-bit floating-point (FP32) arithmetic, which demands significant computational bandwidth and power. 

On the other hand, edge devices (such as holter monitors or pacemakers) operate under extreme battery and thermal constraints, lacking the hardware to execute FP32 operations efficiently. When models are compressed (quantized) to fit into integer-only hardware (like FPGAs or ASICs) using standard techniques, they often suffer a catastrophic drop in sensitivity—particularly for rare but lethal arrhythmias like ST-segment ischaemia or QT-interval anomalies. 

There is an urgent need for an end-to-end framework that can successfully shrink an FP32 clinical model down to an INT8 hardware-compatible format without violating strict medical accuracy thresholds.

## 4. OBJECTIVES

The primary objective of this project is to deliver a fully verifiable software-to-hardware pipeline for an ECG inference accelerator. Specifically:

1. **Develop a Clinical-Grade FP32 Baseline:** Implement a Multi-Scale 1D-CNN trained on the MIT-BIH dataset capable of 5-class multi-label arrhythmia classification with high sensitivity.
2. **Implement Sensitivity-Preserving QAT (SP-QAT):** Design a Quantization-Aware Training pipeline that compresses the model to 8-bit integer weights and activations while guaranteeing a maximum sensitivity degradation of ≤ 2.0% per clinical class.
3. **Bridge Software and Hardware:** Automate the extraction and formatting of INT8 weights into raw memory (`.mem`) files that seamlessly map to a Vivado Verilog/SystemVerilog `$readmemh` workflow.
4. **Enforce Medical Rigor:** Ensure all data splitting is done strictly at the inter-patient level (AAMI EC57 standard) to prevent data leakage, and utilize a custom `MorphologyWeightedBCELoss` to penalize false negatives for high-risk conditions.

## 5. REQUIREMENTS

### 5.1 Software Requirements
* **Environment:** Python 3.10+, Virtual environment (venv/conda).
* **Deep Learning Framework:** PyTorch 2.1.2+ (for model definition, FP32 training, and inference).
* **Quantization Engine:** Brevitas 0.10.2 (for simulating integer arithmetic during QAT).
* **Data Processing:** `wfdb` 4.1.2 (for handling PhysioNet records), `numpy`, `scikit-learn`.
* **Configuration Management:** YAML-based centralized hyperparameter configuration.

### 5.2 Hardware Requirements
* **Target Architecture:** Xilinx/AMD series FPGA.
* **Synthesis Toolchain:** Vivado Design Suite.
* **Input Interface:** ADC providing 256-sample INT16 windows normalized to INT8.
* **Compute Constraints:** Pure INT8 MAC (Multiply-Accumulate) operations; no floating-point units.

### 5.3 Functional Requirements
* **Data Ingestion:** The system must automatically download and preprocess the MIT-BIH database (48 records, 360 Hz).
* **Model Training:** Must train an FP32 model and output clinical metrics (Sensitivity, Specificity, F1) across 5 classes: ST_segment, QT_interval, P_wave, Bundle_Branch_Block, Normal.
* **SP-QAT Calibration & Fine-tuning:** The pipeline must initialize scale factors using a minimum of 100 calibration batches before executing fine-tuning epochs.
* **Validation Checkpoint:** The software must abort the hardware export process if the sensitivity of any single class drops by more than 2% compared to the FP32 baseline.
* **Export Format:** Must generate `.mem` files formatted as 2-digit uppercase hexadecimal in two's complement, row-major order, accompanied by a `weights_manifest.json`.

### 5.4 Non-Functional Requirements
* **Reproducibility:** All data shuffling, initializations, and splits must be globally seeded for exact determinism.
* **Immutability:** The 5-class index output must be strictly frozen to prevent misinterpretation by the hardware RTL layer.
* **Latency:** Hardware inference for a single 256-sample window must operate in real-time (latency negligible relative to the 711ms window duration).

## 6. REFERENCES

1. Chazal, P. D., et al. (2004). "Automatic classification of heartbeats using ECG morphology and heartbeat interval features." *IEEE Transactions on Biomedical Engineering*, 51(7), 1196-1206.
2. Hannun, A. Y., et al. (2019). "Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms using a deep neural network." *Nature Medicine*, 25(1), 65-69.
3. Kiranyaz, S., et al. (2016). "Real-Time Patient-Specific ECG Classification by 1-D Convolutional Neural Networks." *IEEE Transactions on Biomedical Engineering*, 63(3), 664-675.
4. Jacob, B., et al. (2018). "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference." *CVPR*.
5. Wang, K., et al. (2019). "HAQ: Hardware-Aware Automated Quantization with Mixed Precision." *CVPR*.
6. Esser, S. K., et al. (2019). "Learned Step Size Quantization." *ICLR*.
7. Brevitas Contributors (2020). "Brevitas: A PyTorch library for Quantization-Aware Training." *Xilinx Research*.
8. Goldberger, A. L., et al. (2000). "PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals." *Circulation*, 101(23).
9. Gao, J., et al. (2020). "An Energy-Efficient ECG Processor based on a Lightweight CNN for Real-time Arrhythmia Detection." *IEEE ISCAS*.
10. Li, Dan, et al. (2017). "An FPGA-Based Hardware Accelerator for CNNs Using On-Chip Memories Only." *IEEE Transactions on Circuits and Systems*.
11. Wu, M., et al. (2021). "A Deep Learning Approach for ECG Arrhythmia Classification without QRS Detection." *IEEE Access*.
12. Lin, X., et al. (2016). "Fixed-Point Quantization of Deep Convolutional Networks." *ICML*.
13. Rajpurkar, P., et al. (2017). "Cardiologist-Level Arrhythmia Detection with Convolutional Neural Networks." *arXiv preprint*.
14. Gholami, A., et al. (2021). "A Survey of Quantization Methods for Efficient Neural Network Inference." *IEEE CVPR*.
15. Chen, T., et al. (2020). "Deep Learning for ECG Analysis: Benchmarks and Insights." *Physiological Measurement*.
