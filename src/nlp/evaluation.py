"""NLP/VLM JSON 평가. 모델 없이 Python 표준 라이브러리로 실행합니다."""

import argparse
from itertools import chain
import json
from pathlib import Path
import unicodedata

RELATION_TYPES = {"connects": "connects", "spatial_connects": "connects", "logical_flow": "connects",
                  "contains": "contains", "spatial_contains": "contains", "semantic_grouping": "contains"}


def normalize_text(text: str, ignore_whitespace: bool = False) -> str:
    text = unicodedata.normalize("NFC", text)
    return ("" if ignore_whitespace else " ").join(text.split())


def edit_distance(reference: str, prediction: str) -> int:
    # ponytail: O(n*m) 시간, 긴 문서 평가가 병목이면 검증된 편집거리 라이브러리로 교체.
    previous = list(range(len(prediction) + 1))
    for row, expected in enumerate(reference, start=1):
        current = [row]
        for column, actual in enumerate(prediction, start=1):
            current.append(min(previous[column] + 1, current[-1] + 1,
                               previous[column - 1] + (expected != actual)))
        previous = current
    return previous[-1]


def ratio(numerator: int, denominator: int):
    return numerator / denominator if denominator else None


def scores(tp: int, fp: int, fn: int) -> dict:
    return {"tp": tp, "fp": fp, "fn": fn, "precision": ratio(tp, tp + fp),
            "recall": ratio(tp, tp + fn), "f1": ratio(2 * tp, 2 * tp + fp + fn)}


def compare_sets(expected: set, actual: set) -> dict:
    return scores(len(expected & actual), len(actual - expected), len(expected - actual))


def require_id(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}는 비어 있지 않은 문자열이어야 합니다.")
    return value


def require_list(document: dict, field: str) -> list:
    if not isinstance(document.get(field), list):
        raise ValueError(f"{field}는 배열이어야 합니다.")
    return document[field]


def relation_key(item: dict, relation_type) -> tuple:
    if not isinstance(relation_type, str) or relation_type not in RELATION_TYPES:
        raise ValueError(f"지원하지 않는 관계 유형: {relation_type!r}")
    return (RELATION_TYPES[relation_type], require_id(item.get("source_id"), "source_id"),
            require_id(item.get("target_id"), "target_id"))


def error_keys(document: dict) -> set:
    result = set()
    for item in require_list(document, "errors"):
        if not isinstance(item, dict):
            raise ValueError("errors의 각 항목은 객체여야 합니다.")
        error_type = require_id(item.get("error_type"), "error_type")
        element_ids = require_list(item, "element_ids")
        if not element_ids or any(not isinstance(value, str) or not value.strip() for value in element_ids):
            raise ValueError("오류에는 관련 element_ids가 필요합니다.")
        if len(set(element_ids)) != len(element_ids):
            raise ValueError("오류의 element_ids는 중복될 수 없습니다.")
        result.add((error_type, tuple(sorted(element_ids))))
    return result


def evaluate_document(reference: dict, prediction: dict, ignore_whitespace: bool = False) -> dict:
    if not isinstance(reference, dict) or not isinstance(prediction, dict):
        raise ValueError("정답과 예측은 JSON 객체여야 합니다.")
    if reference.get("document_id") != prediction.get("document_id"):
        raise ValueError("정답과 예측의 document_id가 다릅니다.")
    document_id = require_id(reference.get("document_id"), "document_id")
    ids = require_list(reference, "element_ids")
    if len(set(require_id(value, "element_id") for value in ids)) != len(ids):
        raise ValueError("정답 element_ids는 중복될 수 없습니다.")
    known_ids = set(ids)
    texts = reference.get("texts")
    if not isinstance(texts, dict) or any(not isinstance(text, str) for text in texts.values()):
        raise ValueError("정답 texts는 요소 ID와 정답 문자열의 객체여야 합니다.")
    if not set(texts) <= known_ids:
        raise ValueError("정답 texts에 알 수 없는 요소 ID가 있습니다.")
    elements = {}
    for item in require_list(prediction, "elements"):
        if not isinstance(item, dict):
            raise ValueError("elements의 각 항목은 객체여야 합니다.")
        item_id = require_id(item.get("element_id"), "element_id")
        if item_id in elements:
            raise ValueError(f"예측 요소 ID 중복: {item_id}")
        elements[item_id] = item
    predicted_text_ids = {key for key, item in elements.items() if item.get("class_type") == "Text"}
    missing = set(texts) - predicted_text_ids
    extra = predicted_text_ids - set(texts)
    characters = raw_edits = corrected_edits = improved = worsened = corrected_count = 0
    for item_id in sorted(set(texts) | predicted_text_ids):
        expected = normalize_text(texts.get(item_id, ""), ignore_whitespace)
        item = elements.get(item_id, {}) if item_id in predicted_text_ids else {}
        cv, nlp = item.get("cv_result", {}), item.get("nlp_result", {})
        if not isinstance(cv, dict) or not isinstance(nlp, dict):
            raise ValueError("cv_result와 nlp_result는 객체여야 합니다.")
        raw, corrected = cv.get("raw_text"), nlp.get("corrected_text")
        if any(value is not None and not isinstance(value, str) for value in (raw, corrected)):
            raise ValueError("OCR 원문·교정문은 문자열 또는 null이어야 합니다.")
        raw = normalize_text(raw if raw is not None else "", ignore_whitespace)
        corrected_count += item_id in texts and corrected is not None
        # 교정 보류(null)는 원문을 사용하는 실제 후속 출력 기준으로 평가.
        corrected = normalize_text(corrected, ignore_whitespace) if corrected is not None else raw
        before, after = edit_distance(expected, raw), edit_distance(expected, corrected)
        characters += len(expected)
        raw_edits += before
        corrected_edits += after
        improved += after < before
        worsened += after > before
    cer = {"reference_characters": characters, "raw_edit_distance": raw_edits,
           "corrected_edit_distance": corrected_edits, "raw_cer": ratio(raw_edits, characters),
           "corrected_cer": ratio(corrected_edits, characters),
           "relative_reduction": ratio(raw_edits - corrected_edits, raw_edits),
           "improved_elements": improved, "worsened_elements": worsened,
           "corrected_text_coverage": ratio(corrected_count, len(texts)),
           "missing_text_elements": sorted(missing), "extra_text_elements": sorted(extra)}

    expected_relations = set()
    for item in require_list(reference, "relations"):
        if not isinstance(item, dict):
            raise ValueError("정답 relations의 각 항목은 객체여야 합니다.")
        key = relation_key(item, item.get("relation_type"))
        if not {key[1], key[2]} <= known_ids:
            raise ValueError("정답 관계에 알 수 없는 요소 참조가 있습니다.")
        if key in expected_relations:
            raise ValueError("정답 관계가 중복됩니다.")
        expected_relations.add(key)
    cv_relations, nlp_relations = set(), set()
    unconfirmed = 0
    for item in require_list(prediction, "relations"):
        if not isinstance(item, dict):
            raise ValueError("예측 relations의 각 항목은 객체여야 합니다.")
        cv, nlp = item.get("cv_relation", {}), item.get("nlp_relation", {})
        if not isinstance(cv, dict) or not isinstance(nlp, dict):
            raise ValueError("cv_relation와 nlp_relation는 객체여야 합니다.")
        if cv.get("relation_type") is not None:
            cv_relations.add(relation_key(item, cv["relation_type"]))
        confirmed = nlp.get("is_confirmed")
        if confirmed is not None and type(confirmed) is not bool:
            raise ValueError("is_confirmed는 boolean 또는 null이어야 합니다.")
        if confirmed is True:
            nlp_relations.add(relation_key(item, nlp.get("relation_type")))
        unconfirmed += confirmed is None
    relations = {}
    for name, actual in (("cv", cv_relations), ("nlp_confirmed", nlp_relations)):
        relations[name] = compare_sets(expected_relations, actual)
        relations[name]["by_type"] = {kind: compare_sets(
            {key for key in expected_relations if key[0] == kind},
            {key for key in actual if key[0] == kind}) for kind in ("connects", "contains")}
    relations["unconfirmed_candidates"] = unconfirmed

    reading_order = {"status": "reference_missing"}
    if "reading_order_pairs" in reference:
        pairs = require_list(reference, "reading_order_pairs")
        seen_pairs = set()
        correct = resolved = 0
        for pair in pairs:
            if not isinstance(pair, list) or len(pair) != 2 or any(not isinstance(value, str) for value in pair):
                raise ValueError("읽기 순서 정답은 [앞 요소 ID, 뒤 요소 ID] 배열이어야 합니다.")
            first, second = pair
            if first == second or not {first, second} <= set(texts):
                raise ValueError("읽기 순서는 서로 다른 정답 Text 요소를 참조해야 합니다.")
            if (first, second) in seen_pairs or (second, first) in seen_pairs:
                raise ValueError("읽기 순서 정답이 중복되거나 상충합니다.")
            seen_pairs.add((first, second))
            orders = []
            for item_id in pair:
                nlp = elements[item_id].get("nlp_result", {}) if item_id in predicted_text_ids else {}
                if not isinstance(nlp, dict):
                    raise ValueError("nlp_result는 객체여야 합니다.")
                order = nlp.get("sequence_order")
                if order is not None and (type(order) is not int or order < 1):
                    raise ValueError("읽기 순서는 양의 정수 또는 null이어야 합니다.")
                orders.append(order)
            if None not in orders and orders[0] != orders[1]:
                resolved += 1
                correct += orders[0] < orders[1]
        reading_order = {"status": "evaluated", "reference_pairs": len(pairs), "correct_pairs": correct,
                         "resolved_pairs": resolved, "pair_accuracy": ratio(correct, len(pairs)),
                         "coverage": ratio(resolved, len(pairs)), "accuracy_on_resolved": ratio(correct, resolved)}

    errors = {"status": "reference_missing"}
    if "errors" in reference:
        expected_errors = error_keys(reference)
        if any(not set(key[1]) <= known_ids for key in expected_errors):
            raise ValueError("정답 오류에 알 수 없는 요소 참조가 있습니다.")
        errors = {"status": "prediction_missing"}
        if "errors" in prediction:
            actual_errors = error_keys(prediction)
            errors = {"status": "evaluated", **compare_sets(expected_errors, actual_errors)}
            kinds = sorted({key[0] for key in expected_errors | actual_errors})
            errors["by_type"] = {kind: compare_sets({key for key in expected_errors if key[0] == kind},
                                                     {key for key in actual_errors if key[0] == kind}) for kind in kinds}
    pipeline = prediction.get("nlp_pipeline", {})
    if not isinstance(pipeline, dict):
        raise ValueError("nlp_pipeline은 객체여야 합니다.")
    inference = pipeline.get("inference_seconds")
    if inference is not None and (type(inference) not in (int, float) or inference < 0):
        raise ValueError("inference_seconds는 음수가 아닌 수여야 합니다.")
    return {"document_id": document_id, "cer": cer, "relations": relations,
            "reading_order": reading_order, "errors": errors, "inference_seconds": inference}


def aggregate(documents: list) -> dict:
    characters = sum(item["cer"]["reference_characters"] for item in documents)
    raw = sum(item["cer"]["raw_edit_distance"] for item in documents)
    corrected = sum(item["cer"]["corrected_edit_distance"] for item in documents)

    def micro(metrics):
        return scores(*(sum(item[field] for item in metrics) for field in ("tp", "fp", "fn")))

    result = {"documents": len(documents), "cer": {
        "reference_characters": characters, "raw_edit_distance": raw, "corrected_edit_distance": corrected,
        "raw_cer": ratio(raw, characters), "corrected_cer": ratio(corrected, characters),
        "relative_reduction": ratio(raw - corrected, raw),
        "improved_elements": sum(item["cer"]["improved_elements"] for item in documents),
        "worsened_elements": sum(item["cer"]["worsened_elements"] for item in documents),
    }, "relations": {}}
    for name in ("cv", "nlp_confirmed"):
        result["relations"][name] = micro([item["relations"][name] for item in documents])
        result["relations"][name]["by_type"] = {kind: micro([
            item["relations"][name]["by_type"][kind] for item in documents]) for kind in ("connects", "contains")}
    orders = [item["reading_order"] for item in documents if item["reading_order"]["status"] == "evaluated"]
    total_pairs = sum(item["reference_pairs"] for item in orders)
    correct_pairs = sum(item["correct_pairs"] for item in orders)
    resolved_pairs = sum(item["resolved_pairs"] for item in orders)
    result["reading_order"] = {"evaluated_documents": len(orders), "reference_pairs": total_pairs,
                               "correct_pairs": correct_pairs, "resolved_pairs": resolved_pairs,
                               "pair_accuracy": ratio(correct_pairs, total_pairs),
                               "coverage": ratio(resolved_pairs, total_pairs)}
    evaluated_errors = [item["errors"] for item in documents if item["errors"]["status"] == "evaluated"]
    result["errors"] = {"evaluated_documents": len(evaluated_errors),
                        "missing_prediction_documents": sum(item["errors"]["status"] == "prediction_missing" for item in documents)}
    missing_error_predictions = result["errors"]["missing_prediction_documents"]
    result["errors"]["status"] = "incomplete" if missing_error_predictions else "evaluated" if evaluated_errors else "reference_missing"
    if evaluated_errors and not missing_error_predictions:
        result["errors"].update(micro(evaluated_errors))
        kinds = sorted(set(chain.from_iterable(item["by_type"] for item in evaluated_errors)))
        result["errors"]["by_type"] = {kind: micro([
            item["by_type"].get(kind, scores(0, 0, 0)) for item in evaluated_errors]) for kind in kinds}
    times = [item["inference_seconds"] for item in documents if item["inference_seconds"] is not None]
    result["timing"] = {"measured_documents": len(times), "mean_inference_seconds": sum(times) / len(times) if times else None}
    return result


def load_documents(path: Path) -> dict:
    paths = sorted(path.rglob("*.json")) if path.is_dir() else [path]
    if not paths:
        raise ValueError(f"JSON 파일이 없습니다: {path}")
    result = {}
    for file in paths:
        document = json.loads(file.read_text(encoding="utf-8-sig"))
        if not isinstance(document, dict):
            raise ValueError(f"JSON 객체가 필요합니다: {file}")
        document_id = require_id(document.get("document_id"), "document_id")
        if document_id in result:
            raise ValueError(f"중복 document_id: {document_id}")
        result[document_id] = document
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path, help="정답 JSON 파일 또는 전용 폴더")
    parser.add_argument("--prediction", required=True, action="append", help="이름=예측파일/폴더. 반복 지정하여 비교")
    parser.add_argument("--output", required=True, type=Path, help="평가 보고서 JSON")
    parser.add_argument("--ignore-whitespace", action="store_true", help="CER 계산 시 공백 제거")
    args = parser.parse_args()
    try:
        references = load_documents(args.reference)
        report = {"normalization": {"unicode": "NFC", "whitespace": "remove" if args.ignore_whitespace else "collapse",
                                    "case_and_punctuation": "preserve"},
                  "matching": "document_id and stable element_id; no detection IoU matching", "runs": {}}
        protected_paths = [args.reference.resolve()]
        for specification in args.prediction:
            if "=" not in specification:
                raise ValueError("--prediction은 이름=경로 형식이어야 합니다.")
            name, path = specification.split("=", 1)
            if not name.strip() or name in report["runs"]:
                raise ValueError("비교 이름은 비어 있지 않고 중복되지 않아야 합니다.")
            prediction_path = Path(path)
            protected_paths.append(prediction_path.resolve())
            predictions = load_documents(prediction_path)
            if set(predictions) != set(references):
                raise ValueError(f"{name}: 정답/예측 문서 집합 불일치. 누락={sorted(set(references) - set(predictions))}, 추가={sorted(set(predictions) - set(references))}")
            documents = [evaluate_document(reference, predictions[key], args.ignore_whitespace)
                         for key, reference in sorted(references.items())]
            report["runs"][name] = {"summary": aggregate(documents), "documents": documents}
        for path in protected_paths:
            if args.output.resolve() == path or (path.is_dir() and path in args.output.resolve().parents):
                raise ValueError("평가 보고서는 입력 파일이나 입력 폴더 안에 저장할 수 없습니다.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"평가 완료: {args.output}")


if __name__ == "__main__":
    main()
