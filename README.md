# Deep Learning-based Camera Object Detection using YOLOv5 and IBM AIHWKIT 

## Description
This repository implements an object detection framework based on YOLOv5 
to train/validate/test analog neural network implementations using 
IBM Analog Hardware Acceleration Kit (AIHWKIT). To investigate the 
capabilities of analog in-memory devices in the domain of autonomous driving,
the Berkeley DeepDrive Dataset (BDD100K) is used for the creation of
training, validation, and test datasets.

## Requirements 
- C++11 Compatible Compiler
- cmake 3.18+
- pybind11 2.6.2+
- scikit-build 0.11.0+
- Python 3.7+
- BLAS Implementation
- PyTorch 1.7+
- CUDA 9.0+ (e.g.:)
  - CUDA Toolkit 11.4 (https://developer.nvidia.com/cuda-downloads)
  - cuDNN v8.2.2 (https://developer.nvidia.com/rdp/cudnn-archive)
- Nvidia CUB 1.8.0

## Installation (Ubuntu)
Install all packages and dependencies with:

`bash setup.sh`

For further instructions on the installation of AIHWKIT, refer to following pages:
- https://aihwkit.readthedocs.io/en/latest/install.html
- https://aihwkit.readthedocs.io/en/latest/advanced_install.html

## Structure
- `config/...`: Contains all parameters and paths regarding the network and the dataset.
- `models/yolov5/...`: Contains YOLOv5 modules and network architectures.
- `utils/...`: Contains auxiliary functions and methods.
- `train.py`: Main script for training the network model on custom training and validation datasets
- `val.py`: Script for numerically validating and testing the trained model on a custom test dataset
- `global_settings.py`: Used for assigning parameters to desired data types. 
- `rpu_settings.py`: Used for modifying RPU device configurations

## Settings
### Dataset
`Berkeley DeepDrive Dataset (BDD100K)`: The dataset configurations can be modified in `config/config_bdd.yaml`.

### Network Architectures
There are four network models adopted from YOLOv5: `yolov5s`, `yolov5m`, `yolov5l`, and `yolov5x`, 
which can be specified in `config/config_net.ini`.

### Network Hyperparameters
The most essential network hyperparameters are defined in `config/config_hyperparameters.ini`.

### RPU Configurations
All parameters regarding analog in-memory device configurations are defined in `config/config_rpu.ini`. 
Available device configurations can be found in `rpu_settings.py` and include:
- Floating-Point Device: `FloatingPointDevice()`
- Single Resistive Devices: `IdealDevice()`, `ConstantStepDevice()`, `LinearStepDevice()`, `ExpStepDevice()`
`SoftBoundsDevice()`, `SoftBoundsPmaxDevice()`, and `PowStepDevice()`.
- Compound Devices: `TransferCompound()` and `MixedPrecisionCompound()`.
- Tiki-Taka Presets: `TikiTakaCapacitorPreset()`, `TikiTakaEcRamPreset()`, `TikiTakaEcRamMOPreset()`, 
`TikiTakaIdealizedPreset()`, `TikiTakaReRamESPreset()`, and `TikiTakaReRamSBPreset()`.
- Mixed-Precision Presets: `MixedPrecisionCapacitorPreset()`, `MixedPrecisionEcRamPreset()`, 
`MixedPrecisionEcRamMOPreset()`, `MixedPrecisionGokmenVlasovPreset()`, `MixedPrecisionIdealizedPreset()`, 
`MixedPrecisionPCMPreset()`, `MixedPrecisionReRamESPreset()`, and `MixedPrecisionReRamSBPreset()`.

## Training
Execute `train.py`, example usage: 

`$ train.py --save-weights True --use-analog True --img-size 480 --batch-size 8 --epochs 400`

## Inference
Execute `val.py`, example usage:

`$ python val.py --weights path/to/best.pt --img-size 480 --batch-size 8 --conf-thres 0.25 --iou-thres 0.45`



