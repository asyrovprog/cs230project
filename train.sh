#!/bin/bash

# python -m experiments.urban3d_training --config "optimal_d"
# python -m experiments.urban3d_training --config "optimal_d2g"
# python -m experiments.urban3d_training --config "optimal_rgb"

# Mask R-CNN training
python -m experiments.urban3d_training --config "easy_rgb"

# Conditional GAN training
python -m experiments.urban3d_training_cgan --itype "rgbd" --epochs 2