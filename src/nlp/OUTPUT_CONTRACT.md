# NLP → 웹 결과 계약 v1

현재 코드가 생성하는 필드의 연동 기준입니다. 팀 간 API 합의와 웹 UI 구현은 별도로 진행해야 합니다. 파일 또는 API 응답 본문으로 같은 JSON 객체를 전달할 수 있습니다. 현재 구현은 파일 출력만 지원합니다.

## 공통 출력

- 입력의 `document_id`, `metadata`, `elements`, `relations`, OCR 원문, 좌표와 CV 필드를 보존합니다.
- `elements[].nlp_result`에 교정문·읽기 순서를 추가합니다.
- `relations[].nlp_relation`에 의미 유형과 확인 상태를 추가합니다.
- `nlp_pipeline`에 실행 정보와 단계 상태를 기록합니다. 신규 실행의 `output_schema_version`은 `"1"`입니다. 이전 샘플에는 이 필드가 없을 수 있습니다.
- 처리 실패는 CLI의 0이 아닌 종료 코드와 오류 문구로 반환하며 정상 결과 파일을 생성하지 않습니다. 실패를 빈 정상 결과로 변환하지 않습니다.
- 단일 파일 CLI는 기존 출력 파일을 성공 시 덮어쓸 수 있습니다. 실패하면 기존 파일은 유지되므로 실패한 실행의 결과로 사용하면 안 됩니다.
- 폴더 CLI는 빈 출력 폴더를 요구하고 모든 추론이 성공한 뒤 저장합니다. 파일 저장 중 디스크 오류까지 트랜잭션으로 보장하지는 않습니다.

| 값 | 웹 처리 |
|---|---|
| `corrected_text` 문자열 | 그 문자열을 표시. 빈 문자열도 교정 결과 |
| `corrected_text: null` | `cv_result.raw_text`를 표시하고 판독·교정 보류 상태로 취급 |
| `sequence_order: null` | 확정 순서가 없음. 배열 위치를 정답 순서로 사용하지 않음 |
| 관계 `is_confirmed: true` | 의미 분석이 확인한 관계 |
| 관계 `is_confirmed: false` | 부정된 관계. CV 원본을 삭제하지 않음 |
| 관계 `is_confirmed: null` | 보류 관계. 확인된 연결로 표시하지 않음 |

## 순서도 전용 출력

`metadata.document_type = "flowchart"`인 문서에만 다음 필드가 추가됩니다. 일반 문서는 이 필드를 새로 생성하지 않습니다.

| 필드 | 내용 |
|---|---|
| `flowchart.nodes[]` | element_id, 노드 문구, 연결된 Text ID, start/end/process/decision/unknown 역할, 역할 판단 출처, 좌표 |
| `flowchart.edges[]` | CV 연결을 노드로 투영한 방향 간선. relation_id, source/target, original_source/target, via_id, 상태 |
| `flowchart.contains[]` | 원래 포함 관계와 상태 |
| `flowchart.invalid_relations[]` | 없는 요소를 참조하는 관계 및 missing_ids |
| `flowchart.unmapped_relation_ids[]` | 그래프 노드에 대응되지 못한 관계 |
| `flowchart.rules` | 적용된 구조 검사 규칙 |
| `flowchart.status` | analyzed 또는 insufficient_nodes. analyzed가 문서 정상 판정을 뜻하지 않음 |
| `errors[]` | 규칙이 찾은 오류 후보와 모델 검토 결과 |
| `proposed_relations[]` | 사용자 승인 전의 연결 수정 제안 |

Text가 한 Box에만 포함된 경우 Text 연결을 Box 노드로 투영합니다. 여러 Box에 속하면 임의로 하나를 선택하지 않습니다. 확인되지 않은 CV 포함 관계를 쓴 투영은 잠정적이며 `projection_basis`에 기록합니다. 원래 element_id와 relations는 유지합니다.
간선 상태는 `confirmed`, `rejected`, `unreviewed`입니다. 구조 검사에서 보류 간선도 가능한 연결로 포함하므로 보류만으로 누락 오류를 만들지 않습니다. 순환 자체는 오류로 표시하지 않습니다.

### 오류 후보

```json
{
  "error_id": "flow_error_001",
  "error_type": "missing_connection",
  "element_ids": ["p"],
  "reason": "종료가 아닌 노드의 다음 연결을 확인하지 못했습니다.",
  "suggestion": "누락된 화살표 또는 연결 대상을 확인하세요.",
  "is_confirmed": null,
  "review_source": "rules"
}
```

`is_confirmed`는 true: 모델 확인, false: 모델 부정, null: 미검토/보류입니다. 웹은 후보·확인·부정을 구분해 표시합니다. 모델이 확인해도 사람이 작성한 정답이라는 뜻은 아닙니다. `errors: []`는 적용한 규칙에서 후보가 없다는 뜻이며 모든 종류의 오류가 없음을 보장하지 않습니다. 키가 없으면 순서도 분석을 적용하지 않은 출력입니다.

| error_type | 적용 기준 |
|---|---|
| invalid_reference | 검출 목록에 없는 요소를 참조하는 관계. element_ids에는 존재하는 관련 요소만 기록 |
| missing_start / missing_end | require_terminal_nodes가 true인데 해당 역할 노드를 찾지 못함. 전체 노드 ID를 관련 위치로 기록 |
| start_has_incoming | 시작 노드로 들어오는 간선 |
| end_has_outgoing | 종료 노드에서 나가는 간선 |
| missing_connection | start/process 노드의 나가는 간선이 없음 |
| decision_branch_missing | decision의 분기 연결 수가 최소값보다 적음. 같은 화살표의 중복 관계는 한 분기로 계산 |
| unreachable_node | 인식된 시작 노드들에서 도달하지 못하는 노드 |

역할 오인식·검출 누락도 같은 후보를 만들 수 있으므로 구조 오류를 바로 확정하지 않습니다. 알려지지 않은 process/decision 역할을 규칙만으로 임의 추정하지 않습니다. Yes/No 등 분기 라벨의 의미나 서로 다른 경로가 필요한지는 현재 규칙에서 검사하지 않습니다.

### 연결 수정 제안

```json
{
  "proposal_id": "flow_proposal_001",
  "source_id": "p",
  "target_id": "e",
  "relation_type": "logical_flow",
  "reason": "누락 연결에 대한 수정 제안",
  "related_error_ids": ["flow_error_001"],
  "status": "needs_user_confirmation",
  "purpose": "suggested_repair"
}
```

제안은 모델이 확인한 missing_connection/decision_branch_missing의 출발 노드 또는 unreachable_node의 도착 노드에 근거해 생성하고 related_error_ids에 해당 후보 ID를 기록합니다. 기존 노드만 참조하고 이미 있는 연결·자기 연결·중복·종료에서 출발·시작으로 도착하는 제안을 거부합니다. 실제 관측한 화살표로 취급하지 않으며 원본 relations에 자동 추가하지 않습니다.

## 웹 편집 후 저장 계약 초안

웹은 편집 결과를 새로운 문서 버전으로 저장하고 원래 document_id와 기존 요소 ID를 유지하는 방향을 권장합니다. 확정 API는 아직 없습니다.

- 텍스트 수정: 기존 element_id와 사용자가 확정한 문자열을 기록.
- 이동: 좌표 기준 이미지 크기를 유지하고 변경된 bounding_box를 기록.
- 추가·삭제: 새 ID는 중복 없이 생성하고 삭제 요소를 참조하는 연결·포함 관계도 정리.
- 연결 변경: 기존 relation_id 또는 새 고유 ID로 source/target/via를 기록.
- 오류 후보 해결·제안 승인: error_id/proposal_id와 사용자 처리 결과를 별도 수정 이력에 기록. 모델 확인값과 사용자 확인을 섞지 않음.

모델 결과 원본은 보관하고 사용자 수정 결과·수정 이력을 별도로 저장해야 수정 횟수와 오류 감소를 평가할 수 있습니다. 현재 저장 API·편집 명령 실행·이미지/PDF 출력은 구현하지 않았습니다.

## 실제 입력 준비와 실행

`metadata.document_type`, CV/OCR JSON, 좌표 기준 이미지를 함께 준비합니다. 이미지 크기·숫자 좌표·양수 영역·범위를 검증합니다. CV가 전처리 이미지를 별도로 저장해 전달해야 하며 NLP가 원본과 전처리 좌표를 변환하지 않습니다.

순서도 규칙은 다음처럼 문서별로 설정합니다. 기본값은 종료 노드 강제 false, 최소 분기 2, 도달 가능성 검사 true입니다. require_terminal_nodes를 true로 설정할지는 문서 규칙에 따라 팀에서 결정합니다.

```json
{
  "document_type": "flowchart",
  "flowchart_rules": {
    "require_terminal_nodes": false,
    "decision_min_branches": 2,
    "check_reachability": true
  }
}
```

실제 폴더 실행 예시입니다. examples의 입력 폴더는 실제 데이터를 받은 뒤 준비하며, outputs는 새 실행용 빈 폴더여야 합니다. JSON과 이미지는 상대 폴더 구조와 파일명이 같아야 합니다. 확장자는 PNG/JPG/JPEG 중 하나입니다.

```powershell
.\.venv\Scripts\python.exe scripts/run_nlp_batch.py --input-dir data/cv_results/test --image-dir data/processed_images/test --output-dir data/output/test_vlm_run_01 --use-model
py src/nlp/evaluation.py --reference data/ground_truth/test --prediction vlm=data/output/test_vlm_run_01 --output data/evaluation/test_vlm_run_01.json
```

JSON은 전체 document_id가 중복되지 않아야 하고 평가 정답과 예측의 문서 집합이 같아야 합니다. 모델은 배치당 한 번 로드하며 이미지는 한 문서씩 처리합니다. 입력 토큰 한도에 걸리는 큰 문서는 사전에 분할하되 좌표·ID와 문서 단위를 함께 관리합니다.
