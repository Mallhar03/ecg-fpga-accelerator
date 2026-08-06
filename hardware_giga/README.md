# Arduino Giga R1 Firmware & Diagnostic Acquisition System

> **Real-time ECG Data Acquisition, Signal Quality Inspection, and INT8 CNN Inference on Arduino Giga R1 (STM32H747XI)**

This directory contains the firmware pipeline for the ECG Arrhythmia Classifier deployed on the **Arduino Giga R1 WiFi** board (powered by dual-core ARM Cortex-M7 @ 480 MHz & Cortex-M4 @ 240 MHz).

---

## Hardware Target & Specs

| Aspect | Detail |
|---|---|
| MCU Target | STM32H747XI (ARM Cortex-M7 @ 480 MHz main core) |
| Board | Arduino Giga R1 WiFi |
| ADC Peripheral | 16-bit SAR ADC via `Arduino_AdvancedAnalog` |
| ADC Resolution | **16-bit** (`0` to `65535` range) |
| Sample Rate | **360 Hz** (2.777 ms sampling interval) |
| Input Pin | Analog Pin `A0` (configurable) |
| Dataset Match | PhysioNet MIT-BIH Arrhythmia Database (native 360 Hz) |
| Output Protocol | USB Serial stream @ 115200 baud (CSV format: `timestamp_ms,raw_adc_val`) |

---

## Directory Structure

```
hardware_giga/
├── README.md                      # Hardware setup & firmware documentation
├── firmware/
│   ├── ecg_giga_main.ino          # Top-level Arduino sketch (setup & main loop)
│   ├── adc_acquisition.h          # ADC configuration & sampling header
│   ├── adc_acquisition.cpp        # 16-bit ADC init, 360 Hz timer, register comments
│   ├── signal_diagnostics.h       # Raw sample streaming & quality analysis header
│   ├── signal_diagnostics.cpp     # CSV output, clipping detection, min/max/mean stats
│   ├── inference.h                # INT8 CNN model inference header (stub)
│   └── inference.cpp              # INT8 model execution placeholder (stub)
└── tests/
    └── golden_vectors/            # Python vs C++ output comparison vectors (.gitkeep)
```

---

## Critical Design Decisions

### 1. Mandatory 360 Hz Sample Rate
The machine learning model was trained on 360 Hz ECG data from the MIT-BIH Arrhythmia Database. The `SAMPLE_RATE_HZ` parameter in `adc_acquisition.h` is set to **360**. Deviating from 360 Hz alters the temporal scale of P-waves, QRS complexes, and T-waves, rendering model inferences invalid.

### 2. 16-Bit Resolution & Prescaler Selection
The STM32H747XI ADC is operated in 16-bit mode using the `Arduino_AdvancedAnalog` library. Inline comments in `adc_acquisition.cpp` document clock prescaling, sampling clock cycles, and internal reference settings to allow hardware debugging of front-end signal distortion.

### 3. Diagnostic Stream Output
Before enabling model inference, `adc_acquisition.cpp` and `signal_diagnostics.cpp` stream raw ADC values over USB Serial in `timestamp_ms,raw_adc_val` format for real-time visualization and diagnostic plotting.

---

## Build & Upload Instructions

1. Open `hardware_giga/firmware/ecg_giga_main.ino` in the **Arduino IDE** (v2.x recommended).
2. Install the **Arduino_AdvancedAnalog** library via Board Manager / Library Manager.
3. Select Board: **Arduino Giga R1 WiFi**.
4. Compile and Upload.
5. Open Serial Monitor or Serial Plotter at **115200 baud**.
