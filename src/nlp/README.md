# HybridOCR — NLP 담당 작업 및 LLM/VLM 실행 가이드

복합 필기 문서의 CV/OCR 결과를 받아 텍스트를 교정하고, 읽기 순서와 요소 간 관계를 분석하여 웹 편집 파트에 전달하는 구조화 JSON을 생성합니다.
이 README는 OCR 이후 담당 작업을 기준으로 작성했으며, 주차 대신 구현 순서와 완료 조건으로 정리합니다.
전체 프로젝트 소개는 [루트 README](../../README.md)를 참고합니다. 아래 실행 명령과 파일 경로는 모두 저장소 루트 기준입니다.
현재 구현 단계와 실제 입력·정답·최종 출력 계약은 [데이터 전달 문서](HANDOFF.md)에 정리했습니다.
순서도 그래프·오류 후보·수정 제안의 웹 전달 규칙은 [출력 계약 v1](OUTPUT_CONTRACT.md)을 참고합니다.

- 저장소: https://github.com/ComputerVision-Detector/HybridOCR
- 작업 브랜치: `feat/llm-vlm-pipeline`
- 현재 시작점: `src/common/cv_nlp_interface.json` mock 데이터로 입출력과 단계 연결을 확인
- 사용할 모델: [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)

## 1. 담당 구간

```text
[CV/OCR 파트]
이미지 전처리 → Text/Arrow/Box 검출 → OCR → 공간 관계 분석
                         ↓ CV 결과 JSON
[LLM/VLM 파트: 이 브랜치]
입력 확인 → OCR 텍스트 교정 → 읽기 순서 추론
→ 관계 복원·보완 → 순서도 의미 분석·오류 탐지 → 결과 JSON
                         ↓ 구조화 결과
[웹/출력 파트]
자동 복원 표시 → 사용자 편집 → 최종 파일 생성
```

LLM은 OCR 텍스트·좌표·관계 데이터를 활용하고, VLM은 이미지와 해당 데이터를 함께 활용합니다.
`src/common/`의 mock JSON은 텍스트 기반 LLM 입력으로 사용합니다.
`--image`를 함께 지정하면 같은 Qwen3.5-4B에 이미지와 JSON을 함께 전달하여 OCR 교정과 기존 관계 판단에 시각적 근거를 사용합니다.

## 2. 현재 구현 상태

| 항목 | 상태 |
|---|---|
| JSON 파일 읽기·쓰기 | 구현됨 |
| 필수 필드·기본 자료형·ID 중복 검사 | 구현됨 |
| 텍스트 교정 → 의미 분석 단계 연결 | 구현됨 |
| 입력 객체·OCR 원문·좌표·CV 관계 보존 | 구현됨 |
| 입력과 같은 파일로 출력하는 작업 차단 | 구현됨 |
| 원문 보존·CLI 입출력 검증 | 테스트 포함 |
| Qwen3.5-4B 로컬 텍스트 추론 | `--use-model`로 실행 |
| Qwen3.5-4B 이미지 + JSON 추론 | `--use-model --image`로 실행 |
| 텍스트 교정·읽기 순서·기존 관계 판단 | 모델 연결 및 응답 검증 구현. 성능 평가는 미완료 |
| 의미 분석의 입력 ID 고정 출력 틀 | 구현됨. 누락·중복·확인된 관계의 유형 누락은 거부 |
| 없는 요소를 참조하는 관계 | 모델 모드에서 판단 제외 및 `validation_error` 표시 |
| 이미지 크기·EXIF 방향 확인, 합성 이미지/JSON mock | 구현됨 |
| CER·읽기 순서·관계·오류 유형 평가 및 여러 결과 비교 | 구현됨. 별도 정답 JSON 사용 |
| 순서도 그래프·노드 역할·규칙 후보·모델 오류 검토 | 구현됨. 순서도 문서에만 적용 |
| 연결 수정 제안 | 확인된 연결 누락·분기 부족·도달 불가 후보의 수정 제안. 자동 적용하지 않음 |
| 이미지 크기·좌표 값·양수 영역·이미지 범위 검사 | 제공된 필드 검사 구현. 원본/전처리 좌표 변환은 미구현 |
| 실제 문서 폴더 처리 | 구현됨. 입력 선검사, 모델 한 번 로드, 별도 빈 출력 폴더 |

기본 실행은 `nlp_pipeline.mode = "scaffold"`이며 교정·의미 분석 단계 상태는 `not_implemented`입니다.
이 모드에서 교정문, 읽기 순서, 관계 확인 값은 기존 값이 없으면 `null`로 남습니다.
순서도 문서는 모델 없이도 그래프·규칙 후보를 생성해 `flowchart_analysis = "rules_only"`로 기록합니다. 모델을 사용하면 `model_reviewed`, 일반 문서는 `skipped`입니다. 신규 출력의 `output_schema_version`은 `1`, 프롬프트 버전은 `3`입니다.
`--use-model` 실행은 `mode = "model"`로 표시하고 모델명·프롬프트 버전·추론 시간을 기록합니다.
`nlp_pipeline.input_type`은 텍스트 입력 시 `json`, 이미지 포함 시 `image_and_json`입니다.
단계의 `completed`는 처리 완료를 뜻하며 예측 정확도를 보장하지 않습니다.

## 3. 입력 데이터와 전달 규칙

현재 mock 입력: [src/common/cv_nlp_interface.json](../common/cv_nlp_interface.json)

`src/common/`에 제공된 파일을 mock 데이터 기준으로 사용합니다. 현재 실제 JSON 데이터는 `cv_nlp_interface.json`이며, 함께 있는 `schema.py`는 빈 Python 파일입니다.
추가 mock 샘플이 생기면 `data/mock_json/`에서도 같은 형식으로 입력할 수 있습니다.

| 필드 | 의미 |
|---|---|
| `document_id` | 문서를 식별하는 문자열 |
| `metadata` | 이미지 크기·처리 시각 등의 정보 |
| `elements[]` | 문서 구성요소 목록 |
| `elements[].element_id` | 요소 ID |
| `elements[].class_type` | 현재 CV 코드의 요소 유형: Text, Arrow, Box |
| `elements[].bounding_box` | 좌상단 x·y, width·height |
| `elements[].cv_result.raw_text` | OCR 원문. 비텍스트 요소는 null일 수 있음 |
| `elements[].cv_result.confidence_score` | 검출 모델 신뢰도 |
| `elements[].cv_result.additional_features.ocr_confidence` | OCR 신뢰도. OCR 결과가 처리된 Text 요소에서 제공 |
| `elements[].nlp_result` | 교정문·읽기 순서 등 NLP 출력 영역 |
| `relations[]` | 요소 간 관계. 관계가 없으면 빈 배열 |
| `relations[].source_id`, `target_id` | 출발·도착 또는 포함·피포함 요소 ID |
| `relations[].cv_relation` | 연결·포함 등 공간 관계와 근거 |
| `relations[].nlp_relation` | 의미 관계 유형 및 확인 결과 |

현재 실행에 필요한 최상위 필드는 `document_id`, `elements`, `relations`입니다.
세부 필드는 현재 mock JSON을 기준으로 개발하고, 실제 CV/OCR 출력과 대조하여 확정합니다. OCR 결과만 받고 공간 관계가 없다면 `relations: []`로 전달하는 규칙을 협의합니다.

- 현재 `data/mock_json/1.txt`는 빈 파일이며 실행용 JSON이 아닙니다.
- VLM 실행용 합성 샘플은 `data/mock_json/vlm_flow.json`과 `data/mock_images/vlm_flow.png`입니다. 두 박스와 화살표, 이미지의 Start/End 문구를 구성했습니다. OCR 입력의 `Strat`은 이미지의 `Start`와 대조할 수 있도록 의도적으로 다르게 기록했습니다.
- 현재 mock JSON에는 요소 목록에 없는 `elem_004`, `elem_005`를 참조하는 관계가 있습니다. 모델 모드에서는 이 관계를 판단에서 제외하고 `nlp_relation.is_confirmed = null`, `validation_error = "missing_element_reference"`로 표시합니다. 관계 평가 전에 샘플 보완이 필요합니다.
- `src/common/schema.py`는 현재 빈 파일입니다. 실행 시 기본 검사는 `src/nlp_pipeline.py`에서 수행합니다.
- 원문·기존 요소 ID·좌표·CV 관계를 보존하고, 모델 판단은 NLP 출력 영역에 기록합니다.
- 판독이나 관계 판단이 어려우면 불확실성을 남기며, 없는 텍스트나 관계를 확정하지 않습니다.

## 4. 구현할 작업

### A. mock JSON 확인 및 입출력 계약 확정

- [x] `src/common/cv_nlp_interface.json`의 요소 유형, 좌표 형식, OCR 텍스트, 관계 필드를 확인한다.
- [x] 요소 ID의 중복을 검사하고, 모델 모드에서 관계의 source/target/via 참조 유효성을 확인한다.
- [ ] 일반 문서, 순서도, 빈 OCR 결과, 관계 없는 문서 샘플을 확보한다.
- [x] 위 상황을 포함한 합성 JSON mock 11개와 별도 정답을 준비한다. 실제 데이터 확보는 남아 있다.
- [ ] 웹 담당과 교정문·읽기 순서·관계·오류 후보의 출력 형식을 합의한다.

완료 조건: 현재 mock 및 추가 샘플을 입력해 원문과 요소 ID를 유지한 결과 JSON을 생성하고, 관계 참조 유효성과 웹 출력 계약을 확정할 수 있다.

### B. OCR 텍스트 교정 — LLM

연결 위치: `src/nlp/correction/text_corrector.py`

- [x] OCR 원문과 주변 요소의 문맥을 모델에 전달한다.
- [x] 교정문을 `nlp_result.corrected_text`에 기록하고 `raw_text`를 보존한다.
- [ ] 고유명사·숫자·수식 등 근거가 부족한 교정은 보류한다.
- [ ] 교정 근거와 불확실성의 기록 형식을 정한다.
- [x] 모델 응답 형식·요소 ID를 검사하고 실패하면 결과 저장을 중단한다.

완료 조건: 동일 데이터에서 교정 전후 CER과 교정으로 새로 발생한 오류를 비교할 수 있다.

### C. 읽기 순서·의미 관계 분석 — LLM, 추후 VLM 보완

연결 위치: `src/nlp/relation/semantic_analyzer.py`

- [x] 좌표와 문맥을 입력해 모델에서 읽기 순서를 추론한다.
- [x] 모델이 반환한 선형 순서를 `nlp_result.sequence_order`에 기록한다.
- [ ] 비선형 문서는 선후 관계·영역 묶음의 별도 표현을 협의한다.
- [x] CV 연결·포함 관계를 모델에 전달하여 `nlp_relation`에 의미 판단을 기록한다.
- [x] 모든 입력 ID가 들어간 출력 틀을 제공하고 관계 누락·중복을 회귀 테스트로 검증한다.
- [x] 기존 관계는 `relations`, 사용자 확인 전 수정 제안은 `proposed_relations`로 구분한다. 팀 API 합의는 별도다.

완료 조건: 요소 ID를 유지하며 읽기 순서와 의미 관계를 출력하고 정답과 비교할 수 있다.

### D. 순서도 의미 분석·오류 탐지 — 그래프 규칙 + LLM/VLM

연결 위치: `src/nlp/relation/flowchart.py`

- [x] 요소와 연결을 그래프로 구성하고 확인·부정·보류 관계를 보존한다.
- [x] 시작·종료·처리·조건 분기·unknown 역할을 추론한다.
- [ ] Yes/No 등의 분기 라벨과 조건 의미를 해석한다.
- [x] 참조 오류·연결 누락·분기 부족·시작/종료의 방향·도달 불가 등의 후보 규칙 v1을 정의한다. 실제 문서의 판정 기준은 팀에서 확정한다.
- [x] 규칙 후보를 LLM/VLM에 전달해 true/false/null로 검토한다.
- [x] 오류 유형·관련 요소 ID·근거·수정 제안을 출력한다.
- [x] 보류 후보를 남기고 수정 제안은 자동으로 CV 관계에 반영하지 않는다.

순환이나 연결 없음 자체를 무조건 오류로 취급하지 않습니다. 정답과 사전에 합의한 문서 규칙을 기준으로 판정합니다.
완료 조건: 정상·오류 순서도에서 유형별 탐지 성능과 정상 문서 오탐을 측정할 수 있다.

### E. 이미지 기반 보완 — VLM

- [x] `--image`로 한 문서의 이미지와 JSON을 함께 전달한다. 기존 문서·요소 ID를 유지한다.
- [x] 이미지와 JSON의 크기를 검사하고 동일 좌표계의 전처리 이미지를 사용한다. 좌표 변환은 아직 구현하지 않았다.
- [x] OCR 영역과 기존 연결·포함 관계를 이미지와 함께 모델에 전달한다.
- [x] 동일한 합성 mock에서 JSON만 사용한 결과와 이미지까지 사용한 결과를 비교한다. 두 실행 모두 프롬프트 버전 3 사용.
- [ ] 실제 필기 문서의 고정된 평가 데이터에서도 두 결과를 비교한다.

현재 CV 좌표는 전처리 이미지 기준입니다. 원본 이미지와 바로 대응한다고 가정하면 안 됩니다.
크기가 같더라도 회전·크롭 등으로 좌표계가 다를 수 있으므로 CV에서 요소를 검출한 이미지 자체를 입력해야 합니다.

### F. 결과 전달 및 평가

- [ ] 결과 JSON을 웹 편집 파트에 전달하고 요소별 수정이 반영되는지 확인한다.
- [x] 모델 연결 후 `nlp_pipeline`의 단계 상태를 처리 모드에 따라 갱신한다.
- [x] 모델 없이 실행하는 평가 코드와 독립된 mock 정답을 만들고 평가 계산을 검증한다.
- [ ] 고정된 테스트 데이터에서 CV/OCR 원본과 보완 결과를 비교한다.
- [ ] 모델·프롬프트 버전, 처리 시간·비용, 대표 실패 사례를 기록한다.

| 평가 대상 | 확인할 지표 |
|---|---|
| 텍스트 교정 | 교정 전후 CER, 새로 발생한 오류 |
| 읽기 순서 | 정답 요소 쌍의 선후 일치율 |
| 연결·포함 관계 | Precision·Recall·F1, CV 단독 대비 변화 |
| 순서도 오류 탐지 | 오류 유형별 Precision·Recall·F1, 정상 사례 오탐 |
| 사용자 수정 부담 | 수정 횟수, 수정 후 남은 오류 |
| 운영 성능 | 문서당 처리 시간·모델 사용 비용 |

모델·프롬프트 개선에는 학습·검증 데이터를 사용하고, 최종 테스트 데이터와 정답은 모델 입력에서 분리합니다.

## 5. 확장 작업 및 협업 범위

확장 작업은 핵심 교정·관계 분석 이후 진행합니다.

- 문서 유형 분류와 표·차트·다이어그램의 구조화: VLM, 추가 요소 유형과 출력 스키마 협의 필요
- 자연어 편집 요청을 텍스트·요소·연결 변경 명령으로 변환: LLM, 웹 편집 API 확정 후 구현

CV/OCR 파트와는 입력 데이터·좌표·이미지 전달을, 웹 파트와는 결과 JSON·편집 반영을 협의합니다.
검출 모델 학습, 웹 편집 UI, 이미지·문서 내보내기는 해당 담당 파트와 연결합니다.

## 6. 실행 및 확인

### 모델 없이 구조 확인

Python 3.9 이상이 필요합니다. 모델 없는 JSON 파이프라인은 표준 라이브러리만 사용하므로 CV용 `requirements.txt` 설치 없이 실행할 수 있습니다.
아래 명령은 `HybridOCR` 저장소 루트에서 실행합니다. 저장소에 있는 mock JSON을 바로 사용합니다.

```powershell
py src/nlp_pipeline.py --input src/common/cv_nlp_interface.json --output data/output/doc_001.nlp.json
```

입력과 출력 파일은 서로 다른 경로여야 합니다.

### Qwen3.5-4B 로컬 실행

개발 PC는 Ryzen 5 7500F, RAM 32GB, RTX 5060 Ti 16GB입니다.
Python 3.11과 기존 CUDA PyTorch `2.7.1+cu128`을 재사용하는 프로젝트 전용 `.venv`를 구성했습니다.
Qwen3.5 지원 Transformers는 `requirements.nlp.txt`에서 버전을 고정합니다.

동일한 환경을 새로 구성할 때 (CUDA PyTorch/torchvision을 먼저 설치):

```powershell
py -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.nlp.txt
```

현재 모델은 BF16으로 GPU에 로드합니다. 텍스트 교정·의미 분석에 각각 한 번 호출하고, 순서도에는 역할 추론 한 번과 오류 후보가 있을 때 검토 한 번을 추가합니다.
생각 과정은 비활성화하고 입력은 호출당 4,096 토큰, 출력은 최대 2,048 토큰으로 제한합니다.
VLM 입력 한도에는 이미지 토큰도 포함됩니다. 모델의 이미지 전처리는 최대 약 100만 픽셀로 제한되며, 전달 JSON에는 좌표 기준이 되는 이미지 크기를 함께 넣습니다.
입력 초과, 올바르지 않은 JSON, 누락·중복·알 수 없는 ID, 잘못된 출력 자료형이 발견되면 결과를 저장하지 않고 종료합니다.
출력 한도로 응답이 잘려 JSON이 완성되지 않은 경우에도 종료합니다.
의미 분석은 입력의 Text ID와 유효한 관계 ID를 모두 넣은 `output_template`을 전달합니다. 모델은 ID를 유지하고 판단 필드만 채우며, 반환된 모든 ID와 자료형을 다시 검사합니다. 순서도 역할·오류 후보 검토도 실제 ID 출력 틀을 사용합니다. 현재 비교 결과는 프롬프트 버전 `3`입니다. 틀에는 판단값 대신 자료형 안내 문자열을 넣어 null을 그대로 복사하는 일을 줄입니다. 잘못된 응답을 자동으로 고치거나 누락 관계를 임의로 확정하지 않습니다.

```powershell
.\.venv\Scripts\python.exe src/nlp_pipeline.py --input src/common/cv_nlp_interface.json --output data/output/doc_001.qwen.json --use-model
```

### 이미지와 JSON으로 VLM 실행

합성 샘플은 아래 스크립트로 생성할 수 있습니다. 실제 데이터의 대체 평가 자료가 아니라 이미지 입력 연결을 확인하는 샘플입니다.

```powershell
.\.venv\Scripts\python.exe scripts/create_mock_vlm.py
.\.venv\Scripts\python.exe src/nlp_pipeline.py --input data/mock_json/vlm_flow.json --image data/mock_images/vlm_flow.png --output data/output/vlm_flow.qwen.json --use-model
```

이미지는 JSON의 `metadata.image_width`, `metadata.image_height`와 크기가 일치해야 합니다.
EXIF 회전이 남아 있는 사진은 먼저 정방향으로 변환하고 해당 이미지 기준으로 좌표를 준비해야 합니다.
`--image`는 `--use-model`과 함께 사용하며, 이미지 경로와 출력 경로도 서로 달라야 합니다.
잘못된 이미지·크기 불일치는 모델 로딩 전에 중단합니다.

최초 실행은 Hugging Face에서 모델 파일을 내려받아 저장소의 `.cache/huggingface/`에 보관합니다.
모델 입력 JSON은 로컬 GPU에서 처리합니다. `.venv/`와 `.cache/`는 Git에서 제외됩니다.
VRAM이 부족하면 다른 GPU 작업을 종료하거나 입력을 나누어 실행합니다.

JSON 전용 mock으로 GPU 실행을 확인했고 결과는 `data/output/doc_001.qwen.json`에 저장했습니다.
텍스트 두 요소의 읽기 순서는 1·2로 기록됐으며, 존재하지 않는 요소를 참조하는 관계는 확인 불가로 표시됐습니다.
이 실행의 두 단계 추론 시간은 약 7.4초로, 다운로드와 모델 로딩 시간은 제외한 값입니다. 한 샘플의 실행 확인이며 품질 평가 결과는 아닙니다.
현재 Windows 환경에서는 선택적 가속 커널 없이 기본 PyTorch 경로를 사용합니다.

이미지 + JSON 합성 mock의 GPU 실행 결과는 `data/output/vlm_flow.qwen.json`에 저장했습니다.
이 실행에서 `Strat → Start` 교정, 두 Text 요소의 읽기 순서 1·2, 박스 간 연결 및 텍스트 포함 관계 판단을 확인했습니다.
프롬프트 버전 3 실행은 순서도 역할 분석까지 포함해 약 18.8초였으며 모델 로딩은 제외한 값입니다. JSON 전용 실행은 `data/output/vlm_flow.llm.json`에 저장했고 약 17.9초였습니다. LLM은 OCR 원문 `Strat`을 유지하고 연결 관계 1개를 확인했으며 포함 관계 2개는 보류했습니다. 역할은 process/end로 예측했습니다. VLM은 Start/End와 start/end 역할을 확인했습니다. 두 실행 모두 규칙 오류 후보는 없었습니다. 한국어 손필기·실제 촬영 문서 성능은 별도 평가가 필요합니다.

공식 참고: [Qwen3.5-4B 모델](https://huggingface.co/Qwen/Qwen3.5-4B), [Transformers Qwen3.5 문서](https://huggingface.co/docs/transformers/v5.17.0/en/model_doc/qwen3_5), [PyTorch 설치](https://pytorch.org/get-started/locally/)

### 테스트

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

테스트 21개는 기존 교정·의미 분석·이미지·평가 검증에 더해 그래프 투영, 포함 관계의 모호성, 반복·평행 분기·역할 보류, 규칙 후보 정답 일치, 오류 검토·수정 제안의 후보 근거, 좌표 검사와 폴더 CLI의 출력 보호를 확인합니다. 여러 관계가 같은 요소를 참조해도 각 ID를 보존하고, 응답의 누락·중복·읽기 순서 중복·유형 없는 확인 관계를 거부하는 회귀 검증을 포함합니다.
모델 응답 병합 테스트는 가짜 응답을 사용하므로 모델 다운로드 없이 실행 가능합니다.
실제 모델 성능을 평가하는 테스트는 아직 포함하지 않습니다.

## 7. 정답 데이터와 평가

평가기는 [evaluation.py](evaluation.py)이며 Python 표준 라이브러리만 사용합니다. GPU나 모델 설치 없이 저장된 결과를 평가할 수 있습니다.
`scripts/create_mock_vlm.py`는 입력 JSON·이미지와 별도로 `data/ground_truth/vlm_flow.json`을 생성합니다. 정답은 그림에 적은 내용에서 작성하며 모델 입력에 넣지 않습니다. 모델 출력을 복사하여 정답으로 사용하면 안 됩니다.

### 같은 문서를 OCR·LLM·VLM으로 비교

현재 생성된 보고서는 동일 입력에 대한 OCR 원본·LLM·VLM 결과 비교입니다.

```powershell
py src/nlp/evaluation.py --reference data/ground_truth/vlm_flow.json --prediction ocr=data/mock_json/vlm_flow.json --prediction llm=data/output/vlm_flow.llm.json --prediction vlm=data/output/vlm_flow.qwen.json --output data/evaluation/vlm_flow.metrics.json
```

LLM 결과는 아래 명령으로, VLM 결과는 앞의 이미지 실행 명령으로 생성합니다. 프롬프트 버전 3은 실제 ID가 들어간 출력 틀과 순서도 분석을 사용합니다. 출력 틀의 안내 문자열은 실제 판단값으로 바꾸도록 지시하고, 자료형·ID 검증을 유지합니다. 한 샘플의 성공이 모든 실제 문서의 응답 안정성을 보장하지는 않습니다.

```powershell
.\.venv\Scripts\python.exe src/nlp_pipeline.py --input data/mock_json/vlm_flow.json --output data/output/vlm_flow.llm.json --use-model
py src/nlp/evaluation.py --reference data/ground_truth/vlm_flow.json --prediction ocr=data/mock_json/vlm_flow.json --prediction llm=data/output/vlm_flow.llm.json --prediction vlm=data/output/vlm_flow.qwen.json --output data/evaluation/vlm_flow.metrics.json
```

보고서는 `runs` 아래 비교 이름별 `summary`와 문서별 `documents`를 담습니다. `summary.cer.corrected_cer`, `summary.relations.nlp_confirmed.f1`, `summary.reading_order.pair_accuracy`로 결과를 비교할 수 있습니다.
OCR 원본에는 NLP 판단이 없으므로 관계 기준 성능은 `relations.cv`를 봅니다. 원본의 `nlp_confirmed` 점수는 NLP 관계 출력이 없는 상태를 나타냅니다.

저장된 [mock 평가 보고서](../../data/evaluation/vlm_flow.metrics.json)의 결과는 다음과 같습니다. 합성 영어 문서 1개·정답 8글자에 대한 실행 확인이며 실제 손필기 성능을 뜻하지 않습니다.

| 항목 | OCR 원본 | LLM 결과 | VLM 결과 |
|---|---|---|---|
| 최종 텍스트 CER | 0.25 | 0.25 | 0.00 |
| CV 관계 F1 | 1.00 | 1.00 | 1.00 |
| NLP 확인 관계 F1 | 0.00 (NLP 출력 없음) | 0.50 | 1.00 |
| 읽기 순서 쌍 일치율 | 0.00 (순서 출력 없음) | 1.00 | 1.00 |
| 순서도 분석 | NLP 출력 없음 | 수행, 후보 0개 | 수행, 후보 0개 |
| 노드 역할 | NLP 출력 없음 | process / end | start / end |

LLM의 관계 F1 0.50은 확인 1개·보류 2개를 반영한 결과입니다. 보류를 자동 확인으로 바꾸지 않습니다. 이번 비교는 프롬프트 버전 3의 합성 샘플 1회 실행이며, 두 모델 입력 방식의 일반적인 우열을 입증하지 않습니다. 정상 샘플 1개이므로 오류 탐지 F1은 정의되지 않으며 정상 문서 오탐은 두 모델 모두 0/1입니다. 역할 예측은 정성적으로 확인했으며 평가기에 역할 지표는 아직 없습니다.

이미 맞는 CV 관계를 확인한 샘플이므로 관계 개선 성능은 검증하지 못했습니다. 정상·오류 관계와 다양한 필기 샘플의 정답을 추가해야 합니다.

### 실제 정답을 받을 때 사용할 형식

정답 파일 하나가 문서 하나에 대응합니다. 아래 ID는 해당 문서의 입력 요소 ID와 같아야 합니다. `texts`에는 모든 Text 요소의 정답을 기록합니다.

```json
{
  "document_id": "mock_vlm_flow",
  "element_ids": ["box_1", "text_1", "box_2", "text_2", "arrow_1"],
  "texts": {"text_1": "Start", "text_2": "End"},
  "reading_order_pairs": [["text_1", "text_2"]],
  "relations": [
    {"source_id": "box_1", "target_id": "box_2", "relation_type": "connects"},
    {"source_id": "box_1", "target_id": "text_1", "relation_type": "contains"},
    {"source_id": "box_2", "target_id": "text_2", "relation_type": "contains"}
  ],
  "errors": []
}
```

`reading_order_pairs`는 첫 번째 요소가 두 번째 요소보다 앞서야 한다는 정답입니다. 비선형 문서는 순서가 정의되는 쌍만 라벨링합니다. `relations`에는 후보 중 정답인 관계를 모두 기록하며, 잘못된 연결은 제외합니다.
오류가 있는 문서는 `errors`에 `{"error_type": "missing_connection", "element_ids": ["box_1", "box_2"]}`처럼 유형과 관련 요소를 기록합니다. 이후 오류 탐지기도 결과 JSON 최상위 `errors`에 같은 형식으로 출력하도록 연결합니다. 오류 유형 명칭은 평가 전에 팀에서 확정해야 합니다.
정답이 아직 없는 선택 항목은 키를 생략합니다. `errors: []`는 오류가 없는 정상 문서라는 정답이고, `errors` 키 생략은 오류 라벨이 없다는 뜻입니다.

여러 문서는 정답과 각 실행 결과를 각각 전용 폴더에 모아 평가합니다.

```powershell
py src/nlp/evaluation.py --reference data/ground_truth --prediction llm=data/predictions/llm --prediction vlm=data/predictions/vlm --output data/evaluation/comparison.json
```

파일명 대신 `document_id`로 대응하며 하위 폴더의 JSON도 읽습니다. 정답과 예측의 문서 집합이 다르거나 ID가 중복되면 평가를 중단합니다. 예제 경로 `data/predictions/llm`, `vlm`은 실제 결과를 모은 뒤 사용합니다. 여러 실행이 섞인 현재 `data/output/` 전체를 한 실행 폴더로 지정하면 중복 문서 ID 때문에 평가할 수 없습니다. 보고서는 입력 파일 또는 입력 폴더 안에 저장할 수 없습니다.

### 지표 계산 규칙과 현재 범위

| 항목 | 계산 규칙 |
|---|---|
| 교정 전후 CER | 요소별 문자 편집거리 합 / 정답 문자 수 합. 삽입·삭제·치환 비용은 각각 1이며 CER은 1을 넘을 수 있음 |
| 교정 보류 | `corrected_text = null`이면 OCR 원문 사용. 빈 문자열은 삭제한 결과로 평가 |
| 추가·누락 텍스트 | 추가 텍스트는 삽입, 누락 텍스트는 삭제로 계산. 해당 ID 목록도 기록 |
| 교정 악화 | 교정 후 편집거리가 늘어난 요소 수와 교정문 제공 비율을 문서별 기록 |
| 읽기 순서 | 정답 쌍 중 선후가 맞는 비율. null·동순위·누락은 미해결이며 전체 정답 쌍 기준 점수에 반영 |
| 연결·포함 관계 | 방향을 포함한 `(유형, source_id, target_id)` 일치로 Precision·Recall·F1. CV 원본과 NLP 확인 결과를 별도 평가 |
| 관계 보류 | NLP는 `is_confirmed = true`인 관계만 예측으로 인정. 보류한 정답 관계는 FN |
| 오류 탐지 | 오류 유형과 관련 요소 ID 집합의 일치로 유형별 Precision·Recall·F1 |
| 처리 시간 | 저장된 `inference_seconds`의 평균. 모델 로딩 시간 제외, 측정 문서 수도 기록 |

텍스트는 Unicode NFC로 정규화하고 연속 공백·줄바꿈을 한 칸으로 통일합니다. 대소문자·문장부호·숫자는 보존합니다. 띄어쓰기를 평가에서 제외하려면 `--ignore-whitespace`를 추가합니다. 서로 다른 정규화 설정의 점수를 직접 비교하지 않습니다.
문서별 CER의 단순 평균 대신 전체 문자 수를 기준으로 집계하고, 관계·오류는 TP·FP·FN 합으로 micro 지표를 계산합니다. 분모가 0인 값은 `null`입니다.
순서도 출력의 오류 평가는 `is_confirmed = true`인 항목만 확정 예측으로 계산합니다. null/false는 확정 오류 점수에 포함하지 않고, 전체 규칙 후보의 지표는 `candidate_generation`에 별도로 기록합니다. 정상 문서의 확정 오류/후보 오탐률도 구분합니다. 이전 결과처럼 errors가 아예 없으면 문서별 prediction_missing, 집계 incomplete로 표시합니다. 미적용을 정상 판정으로 취급하지 않습니다. 외부의 기존 오류 JSON에서 is_confirmed가 생략된 항목은 확정 예측으로 처리합니다.

관계 유형은 현재 출력 계약에 맞춰 `spatial_connects`·`logical_flow`를 `connects`로, `spatial_contains`·`semantic_grouping`을 `contains`로 대응시킵니다. 향후 포함과 의미 묶음을 별개로 출력하면 이 대응 규칙도 수정해야 합니다.
현재 평가기는 같은 CV 요소 ID를 유지하는 후처리 비교용입니다. 검출 좌표의 IoU 매칭, AP/mAP, 사용자 수정 횟수, 비용 평가는 포함하지 않습니다. 제안 관계는 관측 연결이나 확정 관계 평가에 포함하지 않습니다.

계산 참고: [Hugging Face CER 구현](https://github.com/huggingface/evaluate/blob/main/metrics/cer/cer.py), [scikit-learn F1 정의](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html). 평가기 실행에는 이 라이브러리들을 설치할 필요가 없습니다.

## 8. 순서도 전체 파이프라인과 실제 데이터 준비

순서도 입력에는 `metadata.document_type = "flowchart"`를 지정합니다. 기본 규칙은 시작·종료 강제 false, 분기 최소 2, 시작 노드 기준 도달 가능성 검사 true입니다. `metadata.flowchart_rules`로 문서별 기준을 지정하며 상세 유형·필드는 [출력 계약](OUTPUT_CONTRACT.md)에 정의했습니다.

```text
CV/OCR JSON (+ 좌표 기준 이미지)
 → 입력 검증 → 교정 → 읽기 순서·관계 판단
 → 순서도 그래프 → 역할 추론 → 규칙 후보
 → 모델 오류 검토 → 사용자 확인 전 연결 수정 제안 → 결과 JSON
```

모델 없이 규칙 후보를 확인하는 예시입니다. 출력 폴더가 이미 차 있으면 새 실행 폴더를 지정합니다.

```powershell
py scripts/create_flowchart_mock.py
py scripts/run_nlp_batch.py --input-dir data/mock_json/flowchart_cases --output-dir data/output/flowchart_rules_run_02
py src/nlp/evaluation.py --reference data/ground_truth/flowchart_cases --prediction rules=data/output/flowchart_rules_run_02 --output data/evaluation/flowchart_rules_run_02.json
```

`create_flowchart_mock.py`는 정상·분기·반복·빈 OCR·고립 노드·연결 없음·연결 누락·역방향·종료 누락과 일반 메모를 포함한 11문서의 JSON/정답을 생성합니다. 이미지는 없으며 VLM 데이터로 사용하지 않습니다. 합성 CV 역할 힌트와 수작업 정의 오류를 사용한 규칙 회귀 데이터이므로 실제 손필기 성능의 증거가 아닙니다.

### 저장된 규칙·Qwen 비교 결과

[평가 보고서](../../data/evaluation/flowchart_comparison_v3.metrics.json)는 같은 11문서를 모델 없이 처리한 결과와 Qwen3.5-4B로 처리한 결과를 비교합니다. 일반 메모 1개에는 오류 라벨을 붙이지 않아 순서도 오류 평가는 10문서만 사용합니다. 전체 후보 14개 중 Qwen이 확인한 오류는 11개이고 부정한 오류는 3개입니다.

| 항목 | 규칙만 실행 | Qwen 오류 검토 |
|---|---|---|
| 전체 후보 생성 F1 | 1.00 | 1.00 |
| 확인 오류 Precision / Recall / F1 | 미검토 / 0.00 / 0.00 | 1.00 / 0.786 / 0.88 |
| 정상 문서 오탐 | 후보 0/4 | 확인 오류 0/4 |
| 문서당 평균 추론 시간 | 모델 미사용 | 약 30.4초, 로딩 제외 |

규칙 후보는 `is_confirmed: null`이므로 확정 예측으로 세지 않습니다. 후보 F1 1.00은 미리 정한 합성 규칙 사례에 대한 회귀 결과입니다. 관계 정답도 이 세트에서는 관측된 연결을 평가하므로 수정 제안의 정확도를 측정하는 지표가 아닙니다. 연결 수정 제안은 별도로 사람이 확인해야 합니다.

대표 실패 사례는 다음과 같습니다. 원본 결과와 모델 근거를 보관했습니다.

- [연결 누락](../../data/output/flowchart_llm_v3/missing_connection.json): process의 들어오는 연결을 근거로 나가는 연결 누락을 부정했습니다.
- [종료 누락](../../data/output/flowchart_llm_v3/missing_end.json): 새 종료 노드를 제안할 수 없다는 이유로 종료 필수 규칙 위반을 부정했습니다.
- [역방향 연결](../../data/output/flowchart_llm_v3/reversed_connection.json): 종료의 나가는 연결과 도달 불가는 확인했지만 process의 연결 누락은 부정했습니다.

후속 개선에서는 오류 판정과 수정 가능 여부를 분리하고, 입력의 규칙을 일관되게 따르는지 검증해야 합니다. 이 세트에는 읽기 순서 정답 쌍이 없으며 OCR 원문도 이미 정답이므로 해당 품질 개선을 평가할 수 없습니다.

같은 조건으로 재실행하려면 비어 있는 새 폴더를 사용합니다.

```powershell
.\.venv\Scripts\python.exe scripts/run_nlp_batch.py --input-dir data/mock_json/flowchart_cases --output-dir data/output/flowchart_llm_run_02 --use-model
py src/nlp/evaluation.py --reference data/ground_truth/flowchart_cases --prediction rules=data/output/flowchart_rules_v3 --prediction llm=data/output/flowchart_llm_run_02 --output data/evaluation/flowchart_comparison_run_02.json
```

실제 데이터는 별도 CV/OCR JSON 폴더·동일 좌표계 이미지 폴더·별도 정답 폴더로 받습니다. 배치 실행은 모든 입력 ID/기본 자료형/제공 좌표/이미지를 먼저 검사하고 모델을 한 번 로드합니다. 이미지가 있으면 한 페이지씩 입력하며, 모든 추론이 성공한 뒤 결과를 저장합니다. 출력 폴더는 비어 있어야 하고 JSON 파일과 이미지의 상대 경로·파일명이 같아야 합니다. 실제 실행 명령과 웹 처리 기준은 [출력 계약](OUTPUT_CONTRACT.md)에 있습니다.

실제 문서 정답, CV의 좌표 기준 이미지 저장, 팀의 오류 기준·웹 API 합의는 외부 입력이 필요합니다. 이 자료 없이 실제 성능이나 웹 연동 완료를 주장하지 않습니다.

## 9. 작업 파일

```text
HybridOCR/
├─ src/
│  ├─ nlp_pipeline.py                  # JSON 입출력 및 단계 실행
│  ├─ nlp/
│  │  ├─ README.md                    # NLP 담당 작업 및 실행 가이드 (이 문서)
│  │  ├─ OUTPUT_CONTRACT.md           # 웹 전달 필드·오류·제안·실제 입력 규칙
│  │  ├─ qwen.py                      # Qwen3.5-4B LLM/VLM 로딩·추론·응답 검사
│  │  ├─ evaluation.py                # CER·순서·관계·오류 평가 및 실행 결과 비교
│  │  ├─ correction/text_corrector.py  # 텍스트 교정 연결 지점
│  │  ├─ relation/semantic_analyzer.py # 읽기 순서·기존 관계 판단
│  │  └─ relation/flowchart.py         # 그래프·역할·오류 후보·모델 검토·수정 제안
│  ├─ common/
│  │  ├─ cv_nlp_interface.json         # 현재 사용할 CV/OCR mock 데이터
│  │  └─ schema.py                     # 현재 빈 파일
│  ├─ cv/                             # 기존 CV/OCR 구현
│  └─ main.py                         # 기존 이미지 입력 CV 파이프라인
├─ data/
│  ├─ mock_json/                      # 추가 mock JSON 및 VLM 입력 샘플
│  ├─ mock_images/                    # JSON 좌표와 일치하는 합성 이미지
│  ├─ ground_truth/                   # 모델 입력과 분리된 정답 JSON
│  ├─ evaluation/                     # 비교 평가 보고서 JSON
│  └─ output/                         # NLP 결과 JSON
├─ scripts/create_mock_vlm.py          # VLM 합성 입력·이미지·정답 생성
├─ scripts/create_flowchart_mock.py    # 순서도 규칙용 JSON·정답 11개
├─ scripts/run_nlp_batch.py            # 실제/mock JSON 폴더 처리
├─ tests/                             # JSON 입출력·모델 응답·평가 계산 검증
├─ requirements.nlp.txt               # 로컬 모델 실행용 의존성
└─ README.md
```

현재 작업은 로컬 `feat/llm-vlm-pipeline` 브랜치에서 진행합니다. GitHub 푸시는 연결된 `1604jw40` 계정에 저장소 쓰기 권한이 없어 HTTP 403으로 차단되어 있습니다. 권한을 받은 뒤 같은 브랜치를 푸시합니다.
