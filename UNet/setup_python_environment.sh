#!/bin/bash


python3 -m venv tf
source ./tf/bin/activate

pip install --upgrade pip
pip install tensorflow-gpu==2.0.0-beta1
pip install lmdb
pip install scikit-image