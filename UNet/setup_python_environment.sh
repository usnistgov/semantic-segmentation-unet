#!/bin/bash


python3 -m venv tf
source ./tf/bin/activate

pip install --upgrade pip
pip install tensorflow-gpu==1.12.2
pip install lmdb
pip install scikit-image