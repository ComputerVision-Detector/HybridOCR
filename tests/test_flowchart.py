"""그래프 투영·규칙 후보·모델 검토·입력 보호 및 폴더 CLI 검증."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from create_flowchart_mock import cases
from nlp.evaluation import aggregate, evaluate_document
from nlp.relation.flowchart import build_graph, repair_pairs
from nlp_pipeline import run_pipeline
from test_model_pipeline import FakeModel


class FlowchartTest(unittest.TestCase):
    def test_all_curated_cases_and_candidate_evaluation(self):
        evaluated = []
        for name, document, reference in cases():
            with self.subTest(name=name):
                original = deepcopy(document)
                result = run_pipeline(document)
                self.assertEqual(document, original)
                self.assertEqual(result["relations"], [dict(r, nlp_relation={"is_confirmed": None}) for r in original["relations"]])
                if name == "ordinary_note":
                    self.assertNotIn("errors", result)
                    self.assertNotIn("flowchart", result)
                    continue
                actual = {(e["error_type"], tuple(e["element_ids"])) for e in result["errors"]}
                expected = {(e["error_type"], tuple(sorted(e["element_ids"]))) for e in reference["errors"]}
                self.assertEqual(actual, expected)
                self.assertTrue(all(e["is_confirmed"] is None for e in result["errors"]))
                metrics = evaluate_document(reference, result)
                self.assertEqual(metrics["errors"]["tp"], 0)
                self.assertEqual(metrics["errors"]["candidate_generation"]["fn"], 0)
                self.assertEqual(metrics["errors"]["candidate_generation"]["fp"], 0)
                evaluated.append(metrics)
        summary = aggregate(evaluated)["errors"]
        self.assertEqual(summary["candidate_generation"]["f1"], 1)
        self.assertEqual(summary["normal_document_candidate_alarm_rate"], 0)
        self.assertEqual(summary["f1"], 0)  # 미검토 후보를 확인된 오류로 평가하지 않음.

    def test_containment_projection_and_relation_states(self):
        document = {"document_id": "box", "elements": [
            {"element_id": "b", "class_type": "Box"},
            {"element_id": "t", "class_type": "Text", "cv_result": {"raw_text": "Start"}},
            {"element_id": "e", "class_type": "Text", "cv_result": {"raw_text": "End"}},
        ], "relations": [
            {"relation_id": "contains", "source_id": "b", "target_id": "t", "cv_relation": {"relation_type": "spatial_contains"}},
            {"relation_id": "link", "source_id": "t", "target_id": "e", "cv_relation": {"relation_type": "spatial_connects"}, "nlp_relation": {"is_confirmed": False}},
        ]}
        graph = build_graph(document)
        self.assertEqual([n["element_id"] for n in graph["nodes"]], ["b", "e"])
        self.assertEqual(graph["nodes"][0]["role"], "start")
        self.assertEqual(graph["edges"][0]["source_id"], "b")
        self.assertEqual(graph["edges"][0]["original_source_id"], "t")
        self.assertEqual(graph["edges"][0]["status"], "rejected")
        document["elements"].append({"element_id": "b2", "class_type": "Box"})
        document["relations"].append({"relation_id": "contains2", "source_id": "b2", "target_id": "t", "cv_relation": {"relation_type": "spatial_contains"}})
        self.assertIn("t", [n["element_id"] for n in build_graph(document)["nodes"]])

    def test_abstention_cycles_parallel_branches_and_unknown_roles(self):
        _, document, _ = next(item for item in cases() if item[0] == "normal_loop")
        for r in document["relations"]:
            r["nlp_relation"] = {"is_confirmed": None}
        self.assertEqual(run_pipeline(document)["errors"], [])
        _, document, _ = next(item for item in cases() if item[0] == "branch_missing")
        document["relations"].append({"relation_id": "parallel", "source_id": "d", "target_id": "p", "cv_relation": {"relation_type": "spatial_connects"}})
        self.assertEqual(run_pipeline(document)["errors"], [])
        _, document, _ = next(item for item in cases() if item[0] == "empty_ocr")
        self.assertEqual(run_pipeline(document)["flowchart"]["nodes"][1]["role"], "unknown")

    def responses(self):
        _, document, reference = next(item for item in cases() if item[0] == "missing_connection")
        correction = {"elements": [{"element_id": e["element_id"], "corrected_text": e["cv_result"]["raw_text"]} for e in document["elements"]]}
        semantic = {"elements": [{"element_id": e["element_id"], "sequence_order": i} for i, e in enumerate(document["elements"], 1)],
                    "relations": [{"relation_id": "r1", "relation_type": "logical_flow", "is_confirmed": True}]}
        roles = {"nodes": [{"element_id": e["element_id"], "role": e["cv_result"]["additional_features"]["flow_role"]} for e in document["elements"]]}
        review = {"errors": [{"error_id": f"flow_error_{i:03d}", "is_confirmed": True, "reason": "연결이 없습니다.", "suggestion": "연결을 확인하세요."} for i in (1, 2)],
                  "proposed_relations": [{"source_id": "p", "target_id": "e", "reason": "사용자가 누락 연결을 추가하도록 제안"}]}
        return document, reference, [correction, semantic, roles, review]

    def test_model_review_and_proposals_do_not_mutate_cv(self):
        document, reference, responses = self.responses()
        original = deepcopy(document)
        result = run_pipeline(document, FakeModel(responses))
        self.assertEqual(document, original)
        self.assertEqual(len(result["relations"]), 1)
        self.assertEqual(result["proposed_relations"][0]["status"], "needs_user_confirmation")
        self.assertEqual(evaluate_document(reference, result)["errors"]["f1"], 1)
        self.assertEqual(result["nlp_pipeline"]["stages"]["flowchart_analysis"], "model_reviewed")

    def test_isolated_node_repairs_require_confirmed_error_evidence(self):
        _, document, _ = next(item for item in cases() if item[0] == "isolated_node")
        result = run_pipeline(document)
        graph, candidates = result["flowchart"], result["errors"]
        self.assertIn(("p", "u"), repair_pairs(graph, candidates))
        self.assertEqual(repair_pairs(graph, candidates, confirmed_only=True), {})
        for candidate in candidates:
            candidate["is_confirmed"] = True
        allowed = repair_pairs(graph, candidates, confirmed_only=True)
        self.assertEqual(allowed[("p", "u")], ["flow_error_002"])
        self.assertEqual(allowed[("u", "e")], ["flow_error_001"])
        self.assertNotIn(("e", "u"), allowed)
        self.assertNotIn(("p", "e"), allowed)
        self.assertNotIn(("u", "s"), allowed)

    def test_invalid_reviews_are_rejected(self):
        for failure in ("role", "duplicate", "ghost", "unsupported_repair", "confirmation", "missing_proposals"):
            document, _, responses = self.responses()
            if failure == "role":
                responses[2]["nodes"][0]["role"] = "decision"
            elif failure == "duplicate":
                responses[3]["errors"][1]["error_id"] = "flow_error_001"
            elif failure == "ghost":
                responses[3]["proposed_relations"][0]["target_id"] = "ghost"
            elif failure == "unsupported_repair":
                responses[3]["proposed_relations"][0]["source_id"] = "e"
            elif failure == "confirmation":
                responses[3]["errors"][0]["is_confirmed"] = "true"
            else:
                responses[3].pop("proposed_relations")
            original = deepcopy(document)
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                run_pipeline(document, FakeModel(responses))
            self.assertEqual(document, original)

    def test_coordinate_validation_and_invalid_reference(self):
        _, document, _ = cases()[0]
        for value in (-1, True, float("nan"), float("inf"), 900):
            invalid = deepcopy(document)
            invalid["elements"][0]["bounding_box"]["x"] = value
            with self.subTest(x=value), self.assertRaises(ValueError):
                run_pipeline(invalid)
        invalid = deepcopy(document)
        invalid["metadata"]["flowchart_rules"]["decision_min_branches"] = True
        with self.assertRaises(ValueError):
            run_pipeline(invalid)
        document["relations"][0]["target_id"] = "missing"
        result = run_pipeline(document)
        self.assertEqual(result["flowchart"]["invalid_relations"][0]["missing_ids"], ["missing"])
        self.assertIn("invalid_reference", {e["error_type"] for e in result["errors"]})

    def test_batch_cli_and_output_protection(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / "input", root / "output"
            source.mkdir()
            for name, document, _ in cases()[:2]:
                (source / f"{name}.json").write_text(json.dumps(document), encoding="utf-8")
            command = [sys.executable, str(ROOT / "scripts/run_nlp_batch.py"), "--input-dir", str(source), "--output-dir"]
            result = subprocess.run(command + [str(output)], capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(list(output.glob("*.json"))), 2)
            original = (output / "normal.json").read_bytes()
            self.assertNotEqual(subprocess.run(command + [str(output)], capture_output=True).returncode, 0)
            self.assertEqual((output / "normal.json").read_bytes(), original)
            self.assertNotEqual(subprocess.run(command + [str(source / "nested")], capture_output=True).returncode, 0)


if __name__ == "__main__":
    unittest.main()
