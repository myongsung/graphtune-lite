# GraphTune-lite (v2)
**GraphTune: 데이터 탄력적 어댑터와 예산 스케줄링을 갖춘 메모리 최적화 멀티시티·멀티그래프 튜닝 벤치마크**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](#installation)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange.svg)](#installation)
[![License](https://img.shields.io/badge/License-Open%20Source-green.svg)](#license)

GraphTune-lite는 멀티시티/멀티그래프 시계열 예측(교통 등)에서  
**전이 성능(Zero-shot → Few-shot → Fine-tune)**과 **자원 효율(메모리/연산/시간)**을 함께 평가하는 경량 벤치마크 파이프라인입니다.

> ✅ **v2 순수 구조(legacy 제거)** 기준입니다.  
> 외부 API는 `from graphtune import ...` 로 단순합니다.

---

## Table of Contents
- 핵심 아이디어
- 설치
- 데이터 준비
- 빠른 시작
- 실험 흐름(파이프라인)
- CLI 옵션
- 출력 결과 이해하기
- 새 도시(데이터셋) 추가
- 새 모델 추가
- Troubleshooting
- Roadmap
- License
- Citation

---

## 핵심 아이디어
기존 전이 학습/벤치마크는 “정확도만” 비교하는 경우가 많습니다.  
GraphTune-lite는 정확도뿐 아니라 **자원 제약 하에서의 적응 효율**을 평가합니다.

GraphTune-lite가 동시에 측정하는 것들:

1. **Stage 기반 멀티시티 전이**
   - **Stage 0**: source 도시에서 풀 프리트레인(pretrain)
   - **Stage 1+**: target 도시에서  
     **Zero-shot 평가 → Few-shot/Budget curve Fine-tune → 전이 곡선 기록**

2. **예산 곡선(Budget Curve)**
   - 예산 비율(e.g., 0.1 → 0.3 → 1.0)에 따라 성능이 어떻게 개선되는지 자동 기록

3. **Loss-Gradient Budget Scheduler**
   - “더 학습해도 개선이 작아질 때” 자동 조기 종료 → 시간/메모리 절약

4. **효율 측정(Efficiency Profiling)**
   - 파라미터 수, FLOPs, 학습 시간, GPU 피크 메모리 기록

5. **S_uep (Unified Efficiency–Elasticity Score)**
   - 성능 + 전이곡선 AUC + 효율 + 예산 패널티를 통합한 단일 점수  
   - **자원 제약 하에서 가장 좋은 튜닝 전략을 비교 가능**

---

## 설치
### Requirements
- Python ≥ 3.9
- PyTorch ≥ 1.12 (CUDA 있으면 자동 사용)

### Install dependencies
아래를 레포 루트에서 실행하세요.

    pip install -r requirements.txt

### (Optional) FLOPs 측정
FLOPs/MACs 측정을 원하면 `thop`이 필요합니다.

    pip install thop

---

## 데이터 준비
기본 데이터 폴더는 `DATA/` 입니다.

예시 구조:

    DATA/
      metr-la.h5
      pems-bay.h5
      adj_mx.pkl
      graph_sensor_locations.csv

데이터 소스 옵션:
- `--data_source auto` : 로컬에 없으면 자동 다운로드/캐시
- `--data_source local` : 로컬만 사용(없으면 에러)
- `--data_source hf` : HuggingFace에서 다운로드
- `--data_source url` : 지정한 URL에서 다운로드

---

## 빠른 시작
### 1) 멀티시티 전이 + 예산 곡선 실험

    python run_experiment.py \
      --model bigst \
      --datasets metr-la,pems-bay \
      --epochs 50,30 \
      --lrs 0.001,0.0005 \
      --fractions 0.1,0.3,1.0 \
      --fewshot_mode subset \
      --data_source auto \
      --data_dir DATA

실행 흐름:
- **Stage 0 (metr-la)**: 풀 프리트레인
- **Stage 1 (pems-bay)**: 제로샷 평가 → fractions별 파인튜닝 → 전이 곡선 기록 → S_uep 출력

### 2) 1분 스모크 테스트(짧게 돌려보기)

    python run_experiment.py \
      --model bigst \
      --datasets metr-la,pems-bay \
      --epochs 1,1 \
      --fractions 0.1 \
      --fewshot_mode subset \
      --data_source local \
      --data_dir DATA

---

## 실험 흐름(파이프라인)
GraphTune-lite의 기본 파이프라인은 아래 순서로 실행됩니다.

1. **Stage Loop**
   - `--datasets`에 지정한 도시 순서대로 반복 실행

2. **각 Stage에서**
   - `prepare_dataset()`  
     → 도시 데이터 로딩/전처리/sequence 생성/train·val·test loader 구성
   - `build_model()`  
     → 그래프/노드수/시계열 길이에 맞는 모델 생성
   - `load_partial_state()`  
     → 이전 도시에서 학습된 가중치 중 **shape이 맞는 부분만 안전하게 전이**
   - 효율/난이도 측정  
     → StaticEff(파라미터/크기), FLOPs, Difficulty(그래프 난이도)

3. **Stage 0**
   - 풀 프리트레인 → 테스트 평가 → 전이 곡선 시작점 기록

4. **Stage 1+**
   - Zero-shot(val/test) 평가
   - fractions 기반 budget curve fine-tune
   - Scheduler가 개선이 작으면 조기 종료

5. **S_uep 및 Leaderboard**
   - stage 결과를 누적해 S_uep 계산
   - 실시간 리더보드 출력
   - 마지막에 JSON/CSV 저장

---

## CLI 옵션
### 필수
- `--model` : 사용할 모델  
  bigst | baseline | hypernet | dcrnn | dgcrn
- `--datasets` : Stage 순서대로 사용할 도시(데이터셋)  
  예: metr-la,pems-bay

### 학습
- `--epochs` : stage별 epoch (예: 50,30)
- `--lrs` : stage별 learning rate (예: 1e-3,5e-4)
- `--batch_size` (default=128)
- `--stride` (default=1)
- `--data_dir` (default=DATA)

### 데이터 소스
- `--data_source` : auto | hf | url | local
- `--cache_dir`
- `--h5_urls`, `--adj_urls`, `--loc_urls` : stage별 URL override

### Few-shot / Budget curve
- `--fractions` (default=0.1,0.3,1.0)
- `--fewshot_mode` : subset | steps | both

### Loss-Gradient Scheduler
- `--min_gain_rate` (default=0.02)
- `--min_rel_improve` (default=0.005)
- `--patience` (default=1)

### S_uep 가중치
- `--w_perf` (default=0.45)
- `--w_transfer_auc` (default=0.35)
- `--w_eff` (default=0.20)
- `--w_budget` (default=0.25)
- `--difficulty_alpha` (default=1.0)

### 리소스 Budget(선택)
예산 초과 penalty를 S_uep에 반영하고 싶을 때:
- `--budget_mem_mb`
- `--budget_time_sec`
- `--budget_flops_g`
- `--budget_trainable_m`

### 출력
- `--out_json` (default=results.json)
- `--out_csv` (default=results.csv)

---

## 출력 결과 이해하기
### 실시간 로그 의미
- **StaticEff** : 모델 파라미터 수/학습가능 파라미터/크기(MB)
- **FLOPs** : 한 배치 기준 연산량(없으면 thop 설치 필요)
- **Difficulty** : 도시 그래프 난이도 proxy
- **Zero-shot** : 전이 전(튜닝 전) 성능
- **Budget x.xx** : 예산 비율별 fine-tune 성능
- **Leaderboard** : stage별 성능/효율/AUC/점수
- **S_uep** : 통합 전이 효율 점수

### 저장 파일
- `results.json` : stage별 전체 결과 기록(재현/분석용)
- `results.csv` : 논문/테이블용 핵심 지표 요약

json stage_result 주요 필드:
- `zero_rmse`, `test_rmse`, `curve_rmse`
- `static_eff` (params/size)
- `dynamic_eff` (time/mem)
- `flops`
- `difficulty`, `difficulty_weight`
- `transfer_auc`, `eff_norm`, `budget_penalty`
- `stage_score`, `S_uep`

---

## 새 도시(데이터셋) 추가
1) 파일 준비
- `*.h5` (시계열)
- `adj_mx.pkl` (그래프)
- `graph_sensor_locations*.csv` (좌표)

2) 데이터 스펙 등록  
`graphtune/data/sources.py`의 `DATA_SOURCES`에 새 항목을 추가하세요.

3) 포맷이 다르면 다음을 확장하세요:
- `data/graph.py` : 그래프/좌표 로딩
- `data/datasets.py` : Dataset 클래스
- `data/time_features.py` : 시간 피처 생성

> 외부 API는 `prepare_dataset()` 형태를 유지하는 것이 원칙입니다.

---

## 새 모델 추가
1) `graphtune/models/`에 모델 코드 추가  
2) `models/factories.py`에 registry 등록

예시(스케치):

    from .registry import register_model

    @register_model("my_model")
    def build_my_model(bundle, **kwargs):
        return MyModel(
            A=bundle["A"],
            num_nodes=bundle["num_nodes"],
            T_in=bundle["T_in"],
            T_out=bundle["T_out"],
            **kwargs
        )

3) (선택) `graphtune/config.py`에 기본 kwargs 추가

---

## Troubleshooting
### ModuleNotFoundError: graphtune
- 레포 루트에서 실행 중인지 확인
- 추천: editable install

    pip install -e .

### FLOPs가 thop not installed
- FLOPs 측정이 필요하면 thop 설치

    pip install thop

### 센서 좌표 누락 경고
다음과 같은 경고는 좌표 CSV에 없는 센서 id가 있을 때 발생합니다.

    [warn] sensor_ids missing ... Filling missing coords with (0,0)

- `DATA/graph_sensor_locations*.csv`를 확인하세요.

---

## Roadmap
GraphTune-lite(v2)를 기반으로 다음을 확장할 예정입니다.
- **Topology-Aware Elastic Adapter**
- **Memory-aware fine-tuning** (AMP / grad accumulation / activation checkpointing)
- **Diagnostics-driven budget allocation**
- **Multi-domain extension** (RAG, 교육, 단백질 구조 분석 등)

---

## License
오픈소스 라이선스를 따릅니다. 자세한 내용은 `LICENSE` 참고.

---

## Citation
논문/프로젝트를 사용하신다면 아래를 인용해 주세요(추후 업데이트 예정).

    @misc{graphtune2025,
      title        = {GraphTune: Elastic Adapters and Budget Scheduling for Memory-Efficient Multi-City Graph Tuning Benchmark},
      author       = {Anonymous Authors},
      year         = {2025},
      howpublished = {arXiv preprint}
    }
