#!/bin/bash

echo "Hostname: $(hostname -s)"
echo "Node Rank ${SLURM_PROCID}"

# prepare environment
source ${VIRTUAL_ENV}/bin/activate

# Define these env variables to run ML models on cuda and gpu workers properly.
# without these, tensorflow or jax will not detect any GPU cards.
# we point to the specific cuda and cudnn versions available on the cluster.

echo "Using Python from: $(which python)"
