# GraphTune-lite (v2) + GraphTune-RAG
**GraphTune: 데이터 탄력적 어댑터와 예산 스케줄링을 갖춘 멀티시티·멀티그래프 튜닝 & 교통 RAG 벤치마크**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](#설치)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange.svg)](#설치)
[![LLM](https://img.shields.io/badge/LLM-Phi--1.5-lightgrey.svg)](#2-traffic-rag--bigst--phi-15-graphtune-rag)
[![License](https://img.shields.io/badge/License-Open%20Source-green.svg)](#license)

GraphTune-lite는 멀티시티/멀티그래프 시계열 예측(교통 등)에서  
**전이 성능(Zero-shot → Few-shot → Fine-tune)** 과  
**자원 효율(메모리/연산/시간)** 을 함께 평가하는 경량 튜닝 벤치마크입니다.

또한 BigST 기반 그래프 예측 모델과 소형 LLM(Phi-1.5)을 결합한  
**GraphTune-RAG 교통 에이전트 데모**를 포함하여,  
멀티시티 교통 데이터 + RAG + 온디바이스 친화 추론 흐름까지 한 번에 실험할 수 있습니다.

> ✅ **v2 순수 구조(legacy 제거)** 기준입니다.  
> 외부 API는 `from graphtune import ...` 로 단순합니다.

---

## Table of Contents

- [핵심 요약](#핵심-요약)
- [설치](#설치)
- [데이터 준비](#데이터-준비)
- [빠른 시작](#빠른-시작)
  - [1) 멀티시티 전이 + 예산 곡선 (GraphTune-lite)](#1-멀티시티-전이--예산-곡선-graphtune-lite)
  - [2) Traffic RAG + BigST + Phi-1.5 (GraphTune-RAG)](#2-traffic-rag--bigst--phi-15-graphtune-rag)
- [실험 흐름(파이프라인)](#실험-흐름파이프라인)
- [CLI 옵션](#cli-옵션)
- [출력 결과 이해하기](#출력-결과-이해하기)
- [새 도시(데이터셋) 추가](#새-도시데이터셋-추가)
- [새 모델 추가](#새-모델-추가)
- [GraphTune-RAG & TrafficDoc 구조](#graphtune-rag--trafficdoc-구조)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [License](#license)
- [Citation](#citation)

---

## 핵심 요약

기존 전이 학습/벤치마크는 **정확도 하나**만 비교하는 경우가 많습니다.  
GraphTune-lite는 다음을 동시에 측정합니다.

1. **Stage 기반 멀티시티 전이**
   - **Stage 0**: source 도시에서 풀 프리트레인(pretrain)
   - **Stage 1+**: target 도시에서  
     Zero-shot 평가 → Few-shot/Budget curve Fine-tune → 전이 곡선 기록

2. **예산 곡선(Budget Curve)**
   - 예산 비율(e.g., 0.1 → 0.3 → 1.0)에 따라 성능이 어떻게 개선되는지 자동 기록

3. **Loss-Gradient Budget Scheduler**
   - 손실/그래디언트 개선이 포화되는 지점을 감지하여  
     학습 예산(에폭·미니배치 수)을 단계별로 조절 & 조기 종료 → 시간/메모리 절약

4. **효율 측정(Efficiency Profiling)**
   - 파라미터 수, FLOPs, 학습 시간, GPU 피크 메모리 기록

5. **S_uep (Unified Efficiency–Elasticity Score)**
   - 성능 + 전이 곡선 AUC + 효율 + 예산 패널티를 통합한 단일 점수  
   - 자원 제약 하에서 어떤 튜닝 전략이 가장 좋은지 비교 가능

6. **GraphTune-RAG / Traffic Agent (데모)**
   - 멀티시티 교통 그래프(BigST) + 소형 LLM(Phi-1.5) + TF-IDF RAG
   - 각 센서 노드별 텍스트 요약(TrafficDoc)을 생성하고,  
     질의와 관련된 노드의 **실제 미래 예측값**을 요약해 LLM 컨텍스트로 제공
   - “어디가 언제 막히는지, 전이 튜닝으로 얼마나 개선됐는지”를  
     자연어로 질의·응답할 수 있는 교통 에이전트

---

## 설치

### Requirements

- Python ≥ 3.9  
- PyTorch ≥ 1.12 (CUDA 있으면 자동 사용 권장)

### Install dependencies

레포 루트에서:

    pip install -r requirements.txt

### (Optional) FLOPs 측정

FLOPs/MACs 측정을 원하면 `thop` 설치:

    pip install thop

---

## 데이터 준비

기본 데이터 폴더는 `DATA/` 입니다.

예시 구조(교통 예측 + Songdo + RAG 데모 기준):

    DATA/
      metr-la.h5
      pems-bay.h5
      songdo_full.h5

      adj_mx.pkl
      adj_mx_bay.pkl
      adj_songdo_rulebased.pkl

      graph_sensor_locations.csv
      graph_sensor_locations_bay.csv
      songdo_dummy_loc.csv

데이터 소스 옵션:

- `--data_source auto` : 로컬에 없으면 자동 다운로드/캐시  
- `--data_source local` : 로컬만 사용(없으면 에러)  
- `--data_source hf` : HuggingFace에서 다운로드  
- `--data_source url` : 지정한 URL에서 다운로드  

> RAG 데모(`rag_run_experiment.py`)도 동일한 `DATA/`와 그래프/좌표 설정을 그대로 재사용합니다.

---

## 빠른 시작

### 1) 멀티시티 전이 + 예산 곡선 (GraphTune-lite)

METR-LA → PEMS-BAY 순서로 전이 + 예산 곡선 + S_uep 평가:

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

- Stage 0 (metr-la): 풀 프리트레인 → test RMSE 기록  
- Stage 1 (pems-bay):
  - 이전 stage 가중치 일부 전이(`load_partial_state`)
  - Zero-shot 평가
  - 예산 비율별 Fine-tune (0.1 → 0.3 → 1.0)
  - Loss-Gradient Scheduler가 개선 포화 시 조기 종료
  - S_uep와 함께 리더보드 출력 + checkpoint 저장

#### 1분 스모크 테스트(짧게 돌려보기)

    python run_experiment.py \
      --model bigst \
      --datasets metr-la,pems-bay \
      --epochs 1,1 \
      --fractions 0.1 \
      --fewshot_mode subset \
      --data_source local \
      --data_dir DATA

---

### 2) Traffic RAG + BigST + Phi-1.5 (GraphTune-RAG)

BigST 체크포인트 + RAG + LLM을 한 번에 실행하는 end-to-end 데모입니다.

먼저, 멀티시티 전이 실험을 수행해 BigST checkpoint를 생성합니다:

    python run_experiment.py \
      --model bigst \
      --datasets metr-la,pems-bay,songdo \
      --epochs 50,30,20 \
      --lrs 0.001,0.0005,0.0005 \
      --fractions 0.1,0.3,1.0 \
      --fewshot_mode subset \
      --data_source local \
      --data_dir DATA

이 스크립트는 예를 들어 다음 파일을 생성합니다.

- checkpoints/bigst_metr-la_stage0.pt  
- checkpoints/bigst_pems-bay_stage1.pt  
- checkpoints/bigst_songdo_stage2.pt  
- results.json, results.csv  

이제 RAG + LLM 데모를 실행합니다:

    python rag_run_experiment.py \
      --cities metr-la,pems-bay,songdo \
      --city metr-la \
      --data_dir DATA \
      --ckpt_dir checkpoints \
      --results_path results.json \
      --phi_model microsoft/phi-1_5 \
      --query "Which areas in LA tend to be most congested during the evening rush hour?" \
      --horizon 12

이 스크립트는 다음을 수행합니다.

1. 각 도시별 `prepare_dataset()` 호출  
2. 각 센서 노드별 TrafficDoc 텍스트 요약 생성  
3. 모든 도시의 TrafficDoc에 대해 TF-IDF retriever 구축  
4. 각 도시별 BigST checkpoint 로드 & for_bigst 데이터 번들 준비  
5. 지정된 도시(`--city`)에 대해:
   - RAG로 질의와 관련된 센서 노드 k개 retrieval
   - 가장 관련 높은 노드에 대해 BigST로 **실제 미래 교통량 시계열 예측**
   - 스케일 역변환 후 평균/피크/변동성 요약 텍스트 생성
   - `results.json`에서 zero-shot / fine-tuned RMSE를 읽어 성능 메타데이터 포함  
6. 위 모든 컨텍스트를 `microsoft/phi-1_5`에 입력하고  
   **영어로 교통·혼잡 패턴을 설명하는 답변 생성**

---

## 실험 흐름(파이프라인)

### GraphTune-lite (run_experiment.py)

1. Stage Loop  
   - `--datasets`에 지정한 도시 순서대로 반복 실행

2. 각 Stage에서

   - `prepare_dataset()`  
     → 도시 데이터 로딩/전처리/sequence 생성/train·val·test loader 구성  
   - `build_model()`  
     → 그래프/노드 수/시계열 길이에 맞는 모델 생성  
   - `load_partial_state()`  
     → 이전 도시에서 학습된 가중치 중 **shape이 맞는 부분만 안전 전이**  
   - 효율/난이도 측정  
     → StaticEff(파라미터/크기), FLOPs, Difficulty(그래프 난이도 proxy)

3. Stage 0  
   - 풀 프리트레인 → 테스트 평가 → 전이 곡선 시작점 기록

4. Stage 1+  
   - Zero-shot(val/test) 평가  
   - `--fractions` 기반 budget curve fine-tune  
   - Loss-Gradient Scheduler가 개선이 작으면 조기 종료  

5. S_uep 및 Leaderboard  
   - stage 결과를 누적해 S_uep 계산  
   - CLI에서 실시간 리더보드 출력  
   - 마지막에 `results.json` / `results.csv` 저장  

---

## CLI 옵션

### 필수

- `--model`  
  사용할 모델 이름  
  bigst | baseline | hypernet | dcrnn | dgcrn

- `--datasets`  
  Stage 순서대로 사용할 도시(데이터셋)  
  예: metr-la,pems-bay,songdo

### 학습 관련

- `--epochs` : stage별 epoch 수 (예: 50,30,20)  
- `--lrs` : stage별 learning rate (예: 1e-3,5e-4,5e-4)  
- `--batch_size` (default=128)  
- `--stride` (default=1)  
- `--data_dir` (default=DATA)  

### 데이터 소스

- `--data_source` : auto | hf | url | local  
- `--cache_dir` : 캐시 경로  
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

### 실시간 로그

- StaticEff : 모델 파라미터 수 / 학습 가능 파라미터 / 크기(MB)  
- FLOPs : 한 배치 기준 연산량 (없으면 `thop` 설치 필요)  
- Difficulty : 도시 그래프 난이도 proxy  
- Zero-shot : 전이 전(튜닝 전) 성능  
- Budget x.xx : 예산 비율별 fine-tune 성능  
- Leaderboard : stage별 성능/효율/AUC/점수  
- S_uep : 통합 전이 효율 점수  

### 저장 파일

- `results.json` : stage별 전체 결과 기록(재현/분석용)  
- `results.csv` : 논문/테이블용 핵심 지표 요약  

`results.json`의 각 stage_result 주요 필드:

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
- `adj_mx*.pkl` (그래프)  
- `graph_sensor_locations*.csv` (좌표)  

2) 데이터 스펙 등록  

- `graphtune/data/sources.py` 의 `DATA_SOURCES`에 새 항목을 추가

3) 포맷이 다르면 확장할 부분

- `data/graph.py` : 그래프/좌표 로딩  
- `data/datasets.py` : Dataset 클래스  
- `data/time_features.py` : 시간 피처 생성  

> 외부 API는 가능한 한 `prepare_dataset()` 시그니처를 유지하는 게 원칙입니다.

---

## 새 모델 추가

1) `graphtune/models/`에 모델 코드 추가  
2) `models/factories.py`에 registry 등록  

예시 (스케치):

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

3) (선택) `graphtune/config.py`에 DEFAULT_MODEL_KWARGS 추가  

---

## GraphTune-RAG & TrafficDoc 구조

RAG 데모(`rag_run_experiment.py`)에서는 다음 모듈을 사용합니다.

- `graphtune.rag.traffic_docs.TrafficDoc`  
  - `city`, `sensor_id`, `node_index`, `summary_text` 등을 포함  
  - 각 센서 노드별 전체 히스토리/혼잡도 순위를 자연어로 요약한 텍스트

- `build_node_level_docs_from_bundle(bundle, city=...)`  
  - `prepare_dataset()`로 얻은 bundle에서 train/val/test 분포를 보고  
    각 노드의 평균/표준편차/최댓값/혼잡 순위를 계산  
  - 이를 토대로 TrafficDoc를 생성하고, RAG에 사용될 문서를 만듭니다.

- `rag_run_experiment.py` 파이프라인 요약:
  1. 도시별 bundle 준비 (`prepare_dataset`)  
  2. 도시별 TrafficDoc 생성  
  3. 전체 도시 문서에 대해 TF-IDF retriever 구축  
  4. 각 도시별 BigST checkpoint 로드 → test loader에서 **미래 예측 시계열** 추출  
  5. 가장 관련 높은 노드에 대해 **모델 기반 forecast summary** 생성  
  6. `results.json`에서 zero-shot/fine-tuned RMSE를 불러와 성능 메타데이터 생성  
  7. 위 모든 데이터를 LLM(Phi-1.5)에 컨텍스트로 제공해 영어 답변 생성  

---

## Troubleshooting

### ModuleNotFoundError: graphtune

- 레포 루트에서 실행 중인지 확인  
- 권장: editable install  

    pip install -e .

### FLOPs가 `thop not installed` 라고 나올 때

- FLOPs 측정이 필요하면 `thop` 설치:

    pip install thop

### 센서 좌표 누락 경고

다음과 같은 경고는 좌표 CSV에 없는 sensor_id가 있을 때 발생합니다.

    [warn] sensor_ids missing ... Filling missing coords with (0,0)

- `DATA/graph_sensor_locations*.csv`, `songdo_dummy_loc.csv`를 확인하세요.  
- Songdo 등의 커스텀 도시에서는 일부 센서에 (0,0) 더미 좌표를 사용합니다.

---

## Roadmap

GraphTune-lite(v2) 및 GraphTune-RAG를 기반으로 다음을 확장 예정입니다.

- Topology-Aware Elastic Adapter (코드 공개 버전)  
- Memory-aware fine-tuning (AMP / grad accumulation / activation checkpointing)  
- Diagnostics-driven budget allocation (Loss/Gradient 기반 adaptive budget)  
- Multi-domain extension:
  - 지식 그래프 기반 RAG  
  - 교육 지식 맵  
  - 단백질 상호작용 네트워크 등  

---

## License

오픈소스 라이선스를 따릅니다.  
자세한 내용은 `LICENSE` 파일을 참고해주세요.

---

## Citation

논문/프로젝트를 사용하신다면 아래를 인용해 주세요 (초안):

    @misc{graphtune2025,
      title        = {GraphTune: Elastic Adapters and Budget Scheduling for Memory-Efficient Multi-City Graph Tuning and RAG Benchmark},
      author       = {Anonymous Authors},
      year         = {2025},
      howpublished = {arXiv preprint}
    }
