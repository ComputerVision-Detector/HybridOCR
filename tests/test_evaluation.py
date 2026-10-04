"""모델 없이 정답 대비 지표 계산과 평가 CLI를 검증합니다."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nlp.evaluation import aggregate, edit_distance, evaluate_document, normalize_text


class EvaluationTest(unittest.TestCase):
    def setUp(self):
        self.reference = {"document_id": "doc", "element_ids": ["a", "b", "box"],
                          "texts": {"a": "Start", "b": "End"}, "reading_order_pairs": [["a", "b"]],
                          "relations": [{"source_id": "a", "target_id": "b", "relation_type": "connects"}],
                          "errors": []}
        self.prediction = {"document_id": "doc", "elements": [
            {"element_id": "a", "class_type": "Text", "cv_result": {"raw_text": "Strat"},
             "nlp_result": {"corrected_text": "Start", "sequence_order": 1}},
            {"element_id": "b", "class_type": "Text", "cv_result": {"raw_text": "End"},
             "nlp_result": {"corrected_text": "End", "sequence_order": 2}},
        ], "relations": [{"source_id": "a", "target_id": "b",
                          "cv_relation": {"relation_type": "spatial_connects"},
                          "nlp_relation": {"relation_type": "logical_flow", "is_confirmed": True}}],
            "nlp_pipeline": {"inference_seconds": 2.0}}

    def test_cer_and_korean_normalization(self):
        self.assertEqual(edit_distance("kitten", "sitting"), 3)
        self.assertEqual(edit_distance("", "abc"), 3)
        self.assertEqual(edit_distance("abc", ""), 3)
        self.assertEqual(normalize_text("  한\t글!\n"), "한 글!")
        self.assertEqual(normalize_text(" 한 글! ", True), "한글!")
        result = evaluate_document(self.reference, self.prediction)
        self.assertEqual(result["cer"]["raw_cer"], 0.25)
        self.assertEqual(result["cer"]["corrected_cer"], 0)
        self.assertEqual(result["cer"]["relative_reduction"], 1)
        self.assertEqual(result["cer"]["improved_elements"], 1)
        self.assertEqual(result["relations"]["nlp_confirmed"]["f1"], 1)
        self.assertEqual(result["reading_order"]["pair_accuracy"], 1)
        self.assertEqual(result["errors"]["status"], "prediction_missing")

    def test_abstention_missing_extra_and_worsening(self):
        prediction = deepcopy(self.prediction)
        prediction["elements"][0]["nlp_result"]["corrected_text"] = None
        prediction["elements"][1]["nlp_result"]["sequence_order"] = None
        prediction["relations"][0]["nlp_relation"]["is_confirmed"] = None
        result = evaluate_document(self.reference, prediction)
        self.assertEqual(result["cer"]["corrected_cer"], 0.25)
        self.assertEqual(result["cer"]["corrected_text_coverage"], 0.5)
        self.assertEqual(result["reading_order"]["pair_accuracy"], 0)
        self.assertEqual(result["reading_order"]["coverage"], 0)
        self.assertEqual(result["relations"]["nlp_confirmed"]["fn"], 1)
        self.assertEqual(result["relations"]["nlp_confirmed"]["f1"], 0)
        prediction["elements"] = [prediction["elements"][0],
            {"element_id": "extra", "class_type": "Text", "cv_result": {"raw_text": "XYZ"}}]
        result = evaluate_document(self.reference, prediction)
        self.assertEqual(result["cer"]["missing_text_elements"], ["b"])
        self.assertEqual(result["cer"]["extra_text_elements"], ["extra"])
        self.assertEqual(result["cer"]["corrected_edit_distance"], 8)
        prediction = deepcopy(self.prediction)
        prediction["elements"][1]["nlp_result"]["corrected_text"] = ""
        self.assertEqual(evaluate_document(self.reference, prediction)["cer"]["worsened_elements"], 1)

    def test_relation_direction_type_and_empty_labels(self):
        prediction = deepcopy(self.prediction)
        prediction["relations"][0]["source_id"], prediction["relations"][0]["target_id"] = "b", "a"
        result = evaluate_document(self.reference, prediction)
        self.assertEqual(result["relations"]["cv"]["fp"], 1)
        self.assertEqual(result["relations"]["cv"]["fn"], 1)
        prediction["relations"][0]["nlp_relation"]["relation_type"] = "semantic_grouping"
        result = evaluate_document(self.reference, prediction)
        self.assertEqual(result["relations"]["nlp_confirmed"]["by_type"]["contains"]["fp"], 1)
        self.assertEqual(result["relations"]["nlp_confirmed"]["by_type"]["connects"]["fn"], 1)
        reference = {"document_id": "empty", "element_ids": [], "texts": {}, "relations": [], "errors": []}
        prediction = {"document_id": "empty", "elements": [], "relations": [], "errors": []}
        result = evaluate_document(reference, prediction)
        self.assertIsNone(result["cer"]["corrected_cer"])
        self.assertIsNone(result["relations"]["cv"]["f1"])
        self.assertIsNone(result["errors"]["f1"])

    def test_errors_and_weighted_aggregation(self):
        reference = deepcopy(self.reference)
        reference["errors"] = [{"error_type": "wrong_connection", "element_ids": ["a", "b"]}]
        prediction = deepcopy(self.prediction)
        prediction["errors"] = [{"error_type": "wrong_connection", "element_ids": ["b", "a"]},
                                {"error_type": "missing_connection", "element_ids": ["a", "box"]}]
        first = evaluate_document(reference, prediction)
        self.assertEqual(first["errors"]["tp"], 1)
        self.assertEqual(first["errors"]["fp"], 1)
        self.assertAlmostEqual(first["errors"]["f1"], 2 / 3)
        second_reference = {"document_id": "long", "element_ids": ["x"], "texts": {"x": "a" * 92},
                            "relations": []}
        second_prediction = {"document_id": "long", "elements": [
            {"element_id": "x", "class_type": "Text", "cv_result": {"raw_text": "a" * 92}}], "relations": []}
        summary = aggregate([first, evaluate_document(second_reference, second_prediction)])
        self.assertEqual(summary["cer"]["reference_characters"], 100)
        self.assertEqual(summary["cer"]["raw_cer"], 0.02)
        self.assertEqual(summary["timing"]["mean_inference_seconds"], 2)
        missing = evaluate_document(reference, self.prediction)
        partial = aggregate([first, missing])["errors"]
        self.assertEqual(partial["status"], "incomplete")
        self.assertNotIn("f1", partial)

    def test_cli_comparison_and_document_alignment(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            reference = root / "gold.json"
            prediction = root / "prediction.json"
            report = root / "metrics.json"
            reference.write_text(json.dumps(self.reference), encoding="utf-8")
            prediction.write_text(json.dumps(self.prediction), encoding="utf-8")
            command = [sys.executable, str(ROOT / "src/nlp/evaluation.py"), "--reference", str(reference),
                       "--prediction", f"llm={prediction}", "--prediction", f"vlm={prediction}", "--output"]
            result = subprocess.run(command + [str(report)], capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(set(json.loads(report.read_text(encoding="utf-8"))["runs"]), {"llm", "vlm"})
            original = prediction.read_bytes()
            rejected = subprocess.run(command + [str(prediction)], capture_output=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertEqual(prediction.read_bytes(), original)
            wrong = deepcopy(self.prediction)
            wrong["document_id"] = "another"
            prediction.write_text(json.dumps(wrong), encoding="utf-8")
            rejected = subprocess.run(command + [str(report)], capture_output=True)
            self.assertNotEqual(rejected.returncode, 0)


if __name__ == "__main__":
    unittest.main()
