#!/bin/bash
set -e
source /mnt/ssd/project/venvs/deepfake/bin/activate
cd /mnt/ssd/project/deepfake_detector/aifactory-deepfake-detector-main

python -m src.train \
  --config config_cluster.yaml \
  --model efficientnet \
  --image-size 300 \
  --batch-size 32 \
  --device cuda
