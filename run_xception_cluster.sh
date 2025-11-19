#!/bin/bash
set -e

# 가상환경 활성화
source /mnt/ssd/project/venvs/deepfake/bin/activate

# 프로젝트 경로로 이동
cd /mnt/ssd/project/deepfake_detector/aifactory-deepfake-detector-main

# 학습 실행
python -m src.train \
  --config config_cluster.yaml \
  --device cuda
