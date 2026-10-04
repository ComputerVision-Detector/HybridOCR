# NLP 구현 상태 및 실제 데이터 전달 규칙

이 문서는 현재 코드 기준의 구현 상태, CV/OCR 파트에서 받을 데이터, 웹 파트에 전달할 결과를 정리합니다. 실행 방법은 [README](README.md)를 참고합니다. 예제는 형식 설명용이며 실제 추론 결과가 아닙니다.

## 현재 구현 단계

현재는 CV/OCR JSON을 받아 OCR 텍스트를 교정하고 읽기 순서와 기존 관계를 판단한 뒤, 원본에 NLP 결과를 추가한 JSON을 저장하는 단계입니다.

| 단계 | 현재 상태 | 남은 작업 |
|---|---|---|
| JSON 입출력·기본 검증 | 구현 | 좌표 범위 등 상세 검사, 실제 CV 출력 연결 |
| Qwen3.5-4B LLM/VLM 로컬 실행 | 구현, mock GPU 실행 확인 | 실제 필기 데이터 검증 |
| OCR 텍스트 교정 | 구현 | 고유명사·숫자·수식 과교정 평가, 근거·불확실성 출력 |
| Text 읽기 순서 | 구현, 양의 정수 또는 null | 비선형 문서의 선후 관계·영역 묶음 표현 |
| 기존 CV 연결·포함 관계 검토 | 구현, true/false/null | LLM 관계 ID 중복 응답 안정화 |
| 신규·누락 관계 제안 | 미구현 | 기존 관계와 제안 관계를 구분하는 출력 계약 |
| 순서도 그래프·역할 분석·오류 탐지 | 미구현 | 역할·오류 유형·규칙 합의 및 탐지 구현 |
| CER·읽기 순서·관계·오류 평가기 | 구현, 테스트 검증 | 실제 정답 데이터 평가; 오류 탐지 결과 연결 |
| 웹 편집 연동 | 미구현 | 결과 수신·수정 반영 규칙 협의 |
| 표·차트 등 구조화 | 미구현 | 추가 클래스·출력 형식과 처리 범위 협의 |

테스트 11개 통과 기록이 있습니다. 합성 영어 문서 1개에서 VLM 교정 CER 0.25 → 0.00을 확인했지만 실제 필기 성능은 아직 측정하지 않았습니다. 동일 문서의 JSON 전용 LLM 실행은 관계 ID 중복 응답으로 중단됐습니다. 잘못된 결과는 저장하지 않았습니다.

## CV/OCR 파트에서 받을 데이터

문서 한 페이지마다 다음 자료를 받습니다. 현재 CLI는 한 번에 한 문서를 처리합니다.

| 자료 | 용도 | 필요 조건 |
|---|---|---|
| 실제 CV/OCR 결과 JSON | LLM/VLM 공통 입력 | 검출 요소·좌표·OCR 원문·공간 관계 후보 |
| 좌표 기준 이미지 PNG/JPG | VLM 입력 | CV가 요소를 검출한 전처리 이미지와 동일한 좌표계 |
| 사람이 작성한 정답 JSON | 성능 평가 | 모델 입력과 별도 보관, 문서·요소 ID 대응 |
| 문서 유형·데이터 분할 목록 | 문서 유형별 분석·평가 관리 | 권장 자료. 현재 평가기에 자동 분류·분할 기능은 없음 |

LLM만 실행하면 이미지 없이 JSON을 사용할 수 있습니다. VLM까지 비교하려면 이미지를 함께 받아야 합니다. 정답이 없어도 추론은 가능하지만 성능 수치를 검증할 수는 없습니다.

현재 `src/main.py`는 전처리 이미지로 검출·OCR을 수행하고 결과 JSON을 반환하지만, 좌표 기준 이미지를 별도 저장하는 코드는 없습니다. CV 담당에게 그 이미지를 함께 저장·전달하도록 요청해야 합니다. 흐린 이미지의 `RETAKE_REQUIRED` 응답은 정상 입력 JSON이 아니므로 NLP로 보내기 전에 재촬영을 처리합니다.

### 입력 JSON의 전달 계약

| 필드 | 전달 규칙 |
|---|---|
| `document_id` | 문서·페이지 식별자. 평가 데이터 전체에서 중복 없이 관리 |
| `metadata.image_width`, `image_height` | 좌표 기준 이미지의 실제 가로·세로 픽셀 수. VLM 입력 시 필수 |
| `elements[].element_id` | 문서 내 고유 ID. CV → NLP → 웹에서 유지 |
| `elements[].class_type` | 현재 처리 대상은 `Text`, `Arrow`, `Box`. 클래스명 대소문자 유지 |
| `elements[].bounding_box` | `{x, y, width, height}`. 좌상단 원점, 오른쪽 x 증가·아래쪽 y 증가, 픽셀 단위 |
| `elements[].cv_result.raw_text` | Text는 OCR 원문 문자열. 인식 실패는 빈 문자열 또는 null, 도형은 null |
| `cv_result.confidence_score` | 검출 신뢰도. OCR 신뢰도와 구분 |
| `cv_result.additional_features.ocr_confidence` | 제공 가능한 경우 OCR 신뢰도 |
| Arrow의 `additional_features.direction` | 제공 가능한 경우 `right`, `left`, `up`, `down` 또는 `unknown` |
| `relations[].relation_id` | 문서 내 고유 관계 ID |
| `relations[].source_id`, `target_id` | 연결은 출발 → 도착, 포함은 포함 요소 → 피포함 요소 |
| `relations[].cv_relation.relation_type` | `spatial_connects` 또는 `spatial_contains` |
| `relations[].cv_relation.via_id` | 연결에 사용한 Arrow ID, 없으면 null |

좌표는 이미지 범위 안에 있고 width·height는 양수여야 합니다. 이는 전달 계약이며 현재 NLP 코드가 모든 좌표 조건을 검사하지는 않습니다. 현재 코드는 이미지 크기와 EXIF 방향을 검사하지만 회전·크롭 좌표 변환은 수행하지 않습니다. 이미지 크기가 같다는 사실만으로 좌표계가 같은 것은 아닙니다.

관계의 source/target/via는 실제 `elements`의 ID를 참조해야 합니다. 관계를 생성하지 못한 문서는 `relations: []`로 전달할 수 있지만 현재 NLP는 새 관계를 생성하지 않습니다. 따라서 이 입력에서는 관계 복원 결과가 나오지 않습니다.

`nlp_result`·`nlp_relation`은 입력에서 생략할 수 있습니다. 현재 파이프라인은 기존 NLP 필드를 보존할 수 있으므로 처음 추론할 때는 이전 모델 판단 없이 CV/OCR 원본을 전달합니다.

### 입력 예시

```json
{
  "document_id": "real_doc_001_page_01",
  "metadata": {"image_width": 800, "image_height": 400},
  "elements": [
    {
      "element_id": "text_1",
      "class_type": "Text",
      "bounding_box": {"x": 80, "y": 180, "width": 120, "height": 40},
      "cv_result": {"raw_text": "시작", "confidence_score": 0.95}
    },
    {
      "element_id": "text_2",
      "class_type": "Text",
      "bounding_box": {"x": 600, "y": 180, "width": 120, "height": 40},
      "cv_result": {"raw_text": "끗", "confidence_score": 0.90}
    },
    {
      "element_id": "arrow_1",
      "class_type": "Arrow",
      "bounding_box": {"x": 220, "y": 190, "width": 350, "height": 20},
      "cv_result": {"raw_text": null, "additional_features": {"direction": "right"}}
    }
  ],
  "relations": [
    {
      "relation_id": "rel_1",
      "source_id": "text_1",
      "target_id": "text_2",
      "cv_relation": {"relation_type": "spatial_connects", "via_id": "arrow_1"}
    }
  ]
}
```

처음에는 한국어·영어 혼합 필기, OCR 오독·빈 결과, 복잡한 배치, 정상 순서도, 연결 누락·잘못된 연결 사례를 포함해서 받습니다. 문서 유형은 현재 처리 가능한 Text/Arrow/Box 중심으로 시작하고 표·차트 등은 별도 구조 계약을 정합니다.

### 평가용 정답

CV/OCR 결과와 별도로 사람이 이미지와 문서 의도를 확인해 작성합니다. Text 요소가 여러 줄을 하나로 합쳐 인식한 영역이면 그 영역 전체에 대한 정답 문자열을 기록합니다.

```json
{
  "document_id": "real_doc_001_page_01",
  "element_ids": ["text_1", "text_2", "arrow_1"],
  "texts": {"text_1": "시작", "text_2": "끝"},
  "reading_order_pairs": [["text_1", "text_2"]],
  "relations": [
    {"source_id": "text_1", "target_id": "text_2", "relation_type": "connects"}
  ],
  "errors": []
}
```

`texts`는 모든 Text 정답, `reading_order_pairs`는 순서가 정의되는 쌍, `relations`는 올바른 연결·포함 관계입니다. 관계 후보가 잘못 연결된 경우 그 잘못된 연결을 정답에 복사하지 않습니다. 연결 자체가 누락된 오류 문서는 의도한 올바른 연결과 문서에 실제 존재하는 연결을 구분하고, 관계 복원 정답의 기준을 평가 전에 확정해야 합니다.

`errors: []`는 정상 문서라는 라벨입니다. 오류 문서는 `{"error_type": "missing_connection", "element_ids": ["text_1", "text_2"]}`처럼 오류 유형과 관련 요소를 기록합니다. 라벨이 없는 선택 항목은 키를 생략합니다. 오류 유형 명칭과 판정 기준은 팀에서 합의합니다.

현재 평가는 고정된 요소 ID를 유지하는 NLP 후처리 평가입니다. 검출 모델이 놓친 요소까지 평가하려면 독립된 검출 정답·좌표 매칭이 추가로 필요합니다. 학습·검증·테스트는 동일 원본 문서의 재촬영·변형본이 서로 다른 분할에 섞이지 않게 관리합니다.

## NLP에서 최종적으로 내보낼 형식

### 현재 출력: 웹 편집용 결과 JSON

현재 결과는 입력의 `document_id`, `metadata`, `elements`, `relations`, OCR 원문, 좌표, CV 관계를 보존하고 NLP 필드와 실행 정보를 추가한 **문서별 JSON 파일**입니다. 실제 생성 예시는 [VLM 결과](../../data/output/vlm_flow.qwen.json)를 참고합니다.

| 위치 | NLP가 추가하는 결과 |
|---|---|
| Text의 `nlp_result.corrected_text` | 교정 문자열 또는 null |
| Text의 `nlp_result.sequence_order` | 읽기 순서 양의 정수 또는 null |
| 관계의 `nlp_relation.relation_type` | `logical_flow`, `semantic_grouping` 또는 null |
| 관계의 `nlp_relation.is_confirmed` | true: 확인, false: 부정, null: 판단 보류 |
| 관계의 `nlp_relation.validation_error` | 참조 오류가 있는 경우 `missing_element_reference` |
| `nlp_pipeline` | 모드·단계 상태·모델명·프롬프트 버전·입력 종류·추론 시간 |

위 입력에 대한 출력 형식 예시는 아래와 같습니다. 교정문·판단·시간은 형식을 설명하기 위한 예시 값이며, 실제 결과 JSON에는 입력의 모든 요소·좌표·CV 필드가 유지됩니다.

```json
{
  "document_id": "real_doc_001_page_01",
  "metadata": {"image_width": 800, "image_height": 400},
  "elements": [
    {
      "element_id": "text_1",
      "class_type": "Text",
      "bounding_box": {"x": 80, "y": 180, "width": 120, "height": 40},
      "cv_result": {"raw_text": "시작", "confidence_score": 0.95},
      "nlp_result": {"corrected_text": "시작", "sequence_order": 1}
    },
    {
      "element_id": "text_2",
      "class_type": "Text",
      "bounding_box": {"x": 600, "y": 180, "width": 120, "height": 40},
      "cv_result": {"raw_text": "끗", "confidence_score": 0.90},
      "nlp_result": {"corrected_text": "끝", "sequence_order": 2}
    },
    {
      "element_id": "arrow_1",
      "class_type": "Arrow",
      "bounding_box": {"x": 220, "y": 190, "width": 350, "height": 20},
      "cv_result": {"raw_text": null, "additional_features": {"direction": "right"}},
      "nlp_result": {"corrected_text": null, "sequence_order": null}
    }
  ],
  "relations": [
    {
      "relation_id": "rel_1",
      "source_id": "text_1",
      "target_id": "text_2",
      "cv_relation": {"relation_type": "spatial_connects", "via_id": "arrow_1"},
      "nlp_relation": {"relation_type": "logical_flow", "is_confirmed": true}
    }
  ],
  "nlp_pipeline": {
    "mode": "model",
    "stages": {"text_correction": "completed", "semantic_analysis": "completed"},
    "model_id": "Qwen/Qwen3.5-4B",
    "prompt_version": "1",
    "input_type": "image_and_json",
    "inference_seconds": 15.0
  }
}
```

웹 파트는 corrected_text가 null이면 raw_text를 사용하고, 빈 문자열은 교정 결과로 존중합니다. sequence_order의 null을 임의의 확정 순서로 해석하지 않고, is_confirmed의 false와 null을 구분합니다. 비텍스트 요소의 교정문·순서는 null입니다. `completed`는 처리 완료 상태이며 정답 보장이 아닙니다.

웹 화면의 표시·사용자 수정·이미지/PDF 내보내기는 웹/출력 담당이 수행합니다. NLP의 현재 전달물은 그 작업에 필요한 구조화 JSON입니다. 이미지 연결과 수정 결과 저장 계약은 아직 합의하지 않았습니다.

### 향후 추가할 결과: 아직 미구현인 계약 초안

순서도 오류 탐지를 구현하면 결과 JSON 최상위에 `errors`를 추가하는 형식을 제안합니다. 평가기는 error_type과 element_ids를 이미 지원하며 reason과 suggestion은 웹 설명용 확장입니다.

```json
{
  "errors": [
    {
      "error_type": "missing_connection",
      "element_ids": ["text_1", "text_2"],
      "reason": "정의된 흐름 규칙에 필요한 연결이 누락된 것으로 판단됨",
      "suggestion": "두 요소 사이의 연결 여부 확인"
    }
  ]
}
```

현재 결과에 errors가 없는 것은 탐지 미구현이라는 뜻입니다. 정상 문서라는 뜻으로 사용하지 않습니다. 향후 탐지가 수행되고 오류가 없을 때만 `errors: []`를 출력합니다. 확인만 필요한 후보와 확정 오류는 별도 표현을 합의해야 합니다.

신규 관계 제안·순서도 노드 역할·비선형 읽기 순서·사용자 수정 정보도 계약 확정 후 추가합니다. 아직 구현된 출력 필드처럼 웹에 전달하지 않습니다.

### 별도 전달물: 평가 보고서 JSON

웹 편집용 문서 결과와 별도 파일로 보관합니다. 최상위 `runs` 안에 OCR/LLM/VLM 등 실행별 `summary`와 문서별 `documents`가 들어갑니다. CER·관계 점수·읽기 순서·오류 탐지 상태·추론 시간을 포함합니다. 현재 보고서는 [mock 평가 결과](../../data/evaluation/vlm_flow.metrics.json)이며, 실제 데이터 정답을 받으면 같은 평가기로 교체해 평가합니다.

## 다음 진행 순서

1. CV 담당에게 실제 JSON과 좌표 기준 이미지를 한 문서씩 짝지어 받는다.
2. 현재 관계 ID 중복 응답을 안정화하고 실제 입력의 클래스·좌표·관계 규칙을 점검한다.
3. 별도 정답을 받아 고정된 데이터에서 OCR 원본·LLM·VLM을 비교한다.
4. 순서도 오류 유형·신규 관계 제안·웹 출력 계약을 확정하고 남은 기능을 구현한다.
