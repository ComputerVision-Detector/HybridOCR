"""JSON -> 교정 -> 의미 분석 -> 결과 JSON. --use-model 및 --image로 LLM/VLM 실행."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

from nlp.correction.text_corrector import correct_text
from nlp.relation.semantic_analyzer import analyze_relations


def validate_document(document: dict) -> None:
    if not isinstance(document, dict):
        raise ValueError("입력은 JSON 객체여야 합니다.")
    if not isinstance(document.get("document_id"), str) or not document["document_id"].strip():
        raise ValueError("document_id는 비어 있지 않은 문자열이어야 합니다.")
    for field, result_field in (("elements", "nlp_result"), ("relations", "nlp_relation")):
        items = document.get(field)
        if not isinstance(items, list):
            raise ValueError(f"{field}는 배열이어야 합니다.")
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValueError(f"{field}[{index}]는 객체여야 합니다.")
            if result_field in item and not isinstance(item[result_field], dict):
                raise ValueError(f"{field}[{index}].{result_field}는 객체여야 합니다.")
    for field, id_field in (("elements", "element_id"), ("relations", "relation_id")):
        seen = set()
        for item in document[field]:
            item_id = item.get(id_field)
            if not isinstance(item_id, str) or not item_id.strip() or item_id in seen:
                raise ValueError(f"{id_field}는 중복 없는 비어 있지 않은 문자열이어야 합니다.")
            seen.add(item_id)
            cv_field = "cv_result" if field == "elements" else "cv_relation"
            if cv_field in item and not isinstance(item[cv_field], dict):
                raise ValueError(f"{cv_field}는 객체여야 합니다.")
            if field == "elements":
                if not isinstance(item.get("class_type"), str):
                    raise ValueError("class_type은 문자열이어야 합니다.")
                raw_text = item.get("cv_result", {}).get("raw_text")
                if raw_text is not None and not isinstance(raw_text, str):
                    raise ValueError("raw_text는 문자열 또는 null이어야 합니다.")
            else:
                for reference in (item.get("source_id"), item.get("target_id")):
                    if not isinstance(reference, str) or not reference.strip():
                        raise ValueError("source_id와 target_id는 비어 있지 않은 문자열이어야 합니다.")
                via_id = item.get("cv_relation", {}).get("via_id")
                if via_id is not None and not isinstance(via_id, str):
                    raise ValueError("via_id는 문자열 또는 null이어야 합니다.")


def run_pipeline(document: dict, model=None) -> dict:
    validate_document(document)
    result = deepcopy(document)
    started = time.perf_counter()
    result = correct_text(result, model)
    result = analyze_relations(result, model)
    stage_status = "completed" if model is not None else "not_implemented"
    result["nlp_pipeline"] = {
        "mode": "model" if model is not None else "scaffold",
        "stages": {
            "text_correction": stage_status,
            "semantic_analysis": stage_status,
        },
    }
    if model is not None:
        from nlp.qwen import MODEL_ID
        result["nlp_pipeline"].update({"model_id": MODEL_ID, "prompt_version": "1",
                                       "input_type": "image_and_json" if getattr(model, "image", None) is not None else "json",
                                       "inference_seconds": round(time.perf_counter() - started, 3)})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="CV/OCR 결과 JSON")
    parser.add_argument("--output", required=True, type=Path, help="NLP 결과 JSON")
    parser.add_argument("--use-model", action="store_true", help="Qwen3.5-4B를 로컬 CUDA GPU에서 실행")
    parser.add_argument("--image", type=Path, help="JSON 좌표와 일치하는 이미지. --use-model과 함께 사용")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("입력과 출력은 다른 파일이어야 합니다.")
    if args.image is not None:
        if not args.use_model:
            parser.error("--image는 --use-model과 함께 사용해야 합니다.")
        if args.image.resolve() == args.output.resolve():
            parser.error("이미지 입력과 출력은 다른 파일이어야 합니다.")
    try:
        document = json.loads(args.input.read_text(encoding="utf-8-sig"))
        validate_document(document)
        model = None
        if args.use_model:
            from nlp.qwen import QwenModel, load_document_image
            image = load_document_image(document, args.image) if args.image is not None else None
            model = QwenModel(image=image)
        result = run_pipeline(document, model)
        content = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    mode = "모델 처리 완료" if args.use_model else "구조 연결 확인 완료 (모델 미연결)"
    print(f"{mode}: {args.output}")


if __name__ == "__main__":
    main()
