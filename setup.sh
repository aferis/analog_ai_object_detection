#!/bin/sh
# git clone https://gitlab.lrz.de/braindrive/analog_ai_object_detection.git
python3.8 -m venv ./venv
. venv/bin/activate
sudo apt-get install python3-dev libopenblas-dev
pip install cmake scikit-build torch pybind11
sudo apt-get install nvidia-cuda-toolkit
pip install -r requirements.txt
pip install -v aihwkit==0.4.0 --install-option="-DUSE_CUDA=ON"