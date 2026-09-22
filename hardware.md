# Jetson Orin Nano

Carrier board: Yahboom.

## SoC

| Component   | Detail                                              |
|-------------|-----------------------------------------------------|
| CPU         | 6-core Arm Cortex-A78AE v8.2 64-bit                 |
| GPU         | 1024-core NVIDIA Ampere, 32 Tensor Cores            |
| AI performance | up to 40 TOPS                                   |
| RAM         | 8 GB LPDDR5 (shared CPU+GPU)                        |
| Storage     | microSD, eMMC (variant dependent), NVMe via M.2      |

## Connectivity

- Wi-Fi + Bluetooth (M.2 module or integrated)
- Gigabit Ethernet
- USB 3.2 Gen 2 + USB 2.0
- MIPI CSI Camera
- GPIO, I2C, SPI, UART
- HDMI / DisplayPort

## Power

- 7–20V DC, typically 12V/2A or USB-C
- TDP 7–15W, configurable via `nvpmodel`

## Software

- JetPack SDK (Ubuntu 20.04 or 22.04)
- CUDA, cuDNN, TensorRT
- ONNX Runtime with acceleration provider on Ampere GPU
