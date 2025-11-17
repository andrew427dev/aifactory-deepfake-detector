# Deepfake Detector

통합 PyTorch 파이프라인으로 여러 모델(Xception, Swin Transformer, EfficientNet, Cross-Attention, CNN-Transformer)을 학습하고 추론하는 딥페이크 탐지 레포지토리입니다. `config.yaml`과 공통 유틸리티를 통해 실험 설정, 데이터 경로, 추론 제출 포맷을 일관되게 관리할 수 있습니다.

## 프로젝트 구조
```
aifactory-deepfake-detector/
├── config.yaml                 # 공통 하이퍼파라미터/경로 설정
├── data/
│   ├── raw/                    # 원본 데이터 (git 미추적)
│   └── processed/              # 전처리된 real/fake 폴더
├── models/                     # 체크포인트 저장 (git 미추적)
├── src/
│   ├── models/                 # 각 모델별 학습 루프
│   ├── preprocess/             # Dataset & DataLoader 유틸
│   ├── train.py                # 단일 학습 엔트리포인트
│   └── infer.py                # 추론 및 제출 파일 생성 스크립트
├── submission/                 # JSON/CSV 제출 결과
├── tests/                      # pytest 기반 단위 테스트
└── requirements.txt
```

## 환경 설정
```bash
python -m venv .venv
source .venv/bin/activate  # Windows는 .venv\\Scripts\\activate
pip install -r requirements.txt
```

## 설정 파일 (`config.yaml`)
```yaml
model: xception
input_size: 299
batch_size: 32
epochs: 10
learning_rate: 0.0001
experiment_name: default_experiment
tracking_uri: file:./mlruns
seed: 42
train_data_dir: data/processed/train
val_data_dir: data/processed/val
checkpoint_dir: models/
```
필요 시 `experiment_name`, `tracking_uri`, `train/val 데이터 경로`, `checkpoint_dir` 등을 수정하거나 `--config` 플래그로 다른 파일을 지정할 수 있습니다.

## 학습 실행 예시
전처리된 `data/processed/real`, `data/processed/fake` 폴더를 준비한 뒤 다음과 같이 실행합니다.
```bash
python -m src.train \
  --config config.yaml \
  --model xception \
  --train-data-dir data/processed/train \
  --val-data-dir data/processed/val \
  --checkpoint-dir models
```
- MLflow 실험명/트래킹 URI는 `config.yaml` 혹은 `--experiment-name`, `--tracking-uri`로 제어합니다.
- `src/utils.py`의 `get_device`가 자동으로 MPS(Apple), CUDA, CPU 순으로 디바이스를 선택합니다.

## 추론 및 제출 파일 저장
```bash
python -m src.infer \
  --config config.yaml \
  --model xception \
  --checkpoint models/best_xception.pt \
  --test-data-dir data/processed/test \
  --output submission/predictions.json \
  --output-format json
```
- `--output-format csv`를 지정하면 동일한 결과를 CSV로 저장합니다.
- 추론 시 `DeepfakeDataset(return_paths=True)`를 사용하여 이미지 경로와 확률을 `{image, probability, label}` 포맷으로 저장하므로 대회 제출 템플릿으로 확장하기 쉽습니다.

## 테스트
새로운 변경 사항을 적용한 후 다음과 같이 기본 동작을 검증하세요.
```bash
pytest
```
- `tests/test_model_init.py`: Xception/Swin 모델이 사전학습 가중치 없이 초기화 및 forward pass가 가능한지 확인
- `tests/test_dataloader.py`: `get_dataloaders`가 PyTorch `DataLoader`를 정상 반환하는지 확인

## 팁
- `models/`와 `submission/`은 `.gitignore`에 포함되어 있어 로컬 아티팩트만 저장합니다.
- `src/infer.py`는 추론 결과를 JSON/CSV로 자동 직렬화하고 경로를 출력하므로 대회 제출 파일을 빠르게 생성할 수 있습니다.
