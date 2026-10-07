"""순서도 정상·오류 JSON 및 독립 정답 생성. 이미지가 없는 규칙 테스트용."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cases():
    # (ID, 표시 문구, CV 역할 힌트). expected_errors는 규칙 실행 결과에서 만들지 않습니다.
    base = [("s", "Start", "start"), ("p", "Process", "process"), ("e", "End", "end")]
    decision = [("s", "Start", "start"), ("d", "Check", "decision"), ("p", "Process", "process"), ("e", "End", "end")]
    definitions = [
        ("normal", base, [("s", "p"), ("p", "e")], []),
        ("missing_connection", base, [("s", "p")], [("missing_connection", ["p"]), ("unreachable_node", ["e"])]),
        ("reversed_connection", base, [("s", "p"), ("e", "p")], [("missing_connection", ["p"]), ("end_has_outgoing", ["e"]), ("unreachable_node", ["e"])]),
        ("branch_missing", decision, [("s", "d"), ("d", "p"), ("p", "e")], [("decision_branch_missing", ["d"])]),
        ("normal_branch", decision, [("s", "d"), ("d", "p"), ("d", "e"), ("p", "e")], []),
        ("normal_loop", base, [("s", "p"), ("p", "p"), ("p", "e")], []),
        ("empty_ocr", [("s", "Start", "start"), ("p", "", "unknown"), ("e", "End", "end")], [("s", "p"), ("p", "e")], []),
        ("isolated_node", base + [("u", "Other", "process")], [("s", "p"), ("p", "e")], [("missing_connection", ["u"]), ("unreachable_node", ["u"])]),
        ("no_relations", base, [], [("missing_connection", ["s"]), ("missing_connection", ["p"]), ("unreachable_node", ["p"]), ("unreachable_node", ["e"])]),
        ("missing_end", base[:2], [("s", "p")], [("missing_end", ["s", "p"]), ("missing_connection", ["p"])]),
    ]
    result = []
    for name, nodes, edges, expected_errors in definitions:
        document = {"document_id": f"flow_{name}", "metadata": {
            "document_type": "flowchart", "synthetic": True, "image_width": 800, "image_height": 600,
            "flowchart_rules": {"require_terminal_nodes": True}}, "elements": [], "relations": []}
        for index, (item_id, text, role) in enumerate(nodes):
            document["elements"].append({"element_id": item_id, "class_type": "Text",
                "bounding_box": {"x": 50 + (index % 2) * 350, "y": 50 + (index // 2) * 150, "width": 200, "height": 50},
                "cv_result": {"raw_text": text, "additional_features": {"flow_role": role}}})
        for index, (source, target) in enumerate(edges, 1):
            document["relations"].append({"relation_id": f"r{index}", "source_id": source, "target_id": target,
                "cv_relation": {"relation_type": "spatial_connects", "via_id": None, "is_valid_geometry": True}})
        reference = {"document_id": document["document_id"], "synthetic": True,
            "element_ids": [node[0] for node in nodes], "texts": {node[0]: node[1] for node in nodes},
            "relations": [{"source_id": source, "target_id": target, "relation_type": "connects"} for source, target in edges],
            "errors": [{"error_type": kind, "element_ids": ids} for kind, ids in expected_errors]}
        result.append((name, document, reference))
    # 일반 문서는 오류 라벨 자체를 붙이지 않습니다.
    document = {"document_id": "ordinary_note", "metadata": {"document_type": "note", "synthetic": True},
                "elements": [{"element_id": "note", "class_type": "Text", "cv_result": {"raw_text": "메모"}}], "relations": []}
    result.append(("ordinary_note", document, {"document_id": "ordinary_note", "element_ids": ["note"], "texts": {"note": "메모"}, "relations": []}))
    return result


def create_mock(root=ROOT):
    for name, document, reference in cases():
        for folder, value in (("mock_json", document), ("ground_truth", reference)):
            path = root / "data" / folder / "flowchart_cases" / f"{name}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return len(cases())


if __name__ == "__main__":
    print(f"순서도/일반 문서 mock {create_mock()}개 생성")
