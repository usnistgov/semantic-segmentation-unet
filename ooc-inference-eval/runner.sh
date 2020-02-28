#!/bin/bash

cd unet

python compute_runtime_overhead.py
python run_inference_sweep.py

cd ../densenet
python compute_runtime_overhead.py
python run_inference_sweep.py