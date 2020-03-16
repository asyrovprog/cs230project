#!/bin/bash

# Mask R-CNN evaluation
python -m experiments.urban3d_validation --config "easy_rgb" --dataset "test"

# Conditional GAN evaluation
python -m experiments.urban3d_validation_cgan --itype "rgbd" --dataset "test"