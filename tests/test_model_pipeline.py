"""실제 모델 다운로드 없이 응답 병합·잘못된 응답 거부를 검사합니다."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nlp.qwen import parse_model_json
from nlp_pipeline import run_pipeline


class FakeModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.payloads = []
        self.instructions = []

    def generate_json(self, instruction, payload):
        self.payloads.append(payload)
        self.instructions.append(instruction)
        return next(self.responses)


class ModelPipelineTest(unittest.TestCase):
    def setUp(self):
        self.document = {
            "document_id": "test",
            "elements": [
                {"element_id": "text_1", "class_type": "Text", "cv_result": {"raw_text": "시작"}},
                {"element_id": "text_2", "class_type": "Text", "cv_result": {"raw_text": "끗"}},
                {"element_id": "arrow_1", "class_type": "Arrow", "cv_result": {"raw_text": None}},
            ],
            "relations": [
                {"relation_id": "rel_1", "source_id": "text_1", "target_id": "text_2",
                 "cv_relation": {"relation_type": "spatial_connects", "via_id": "arrow_1"}},
                {"relation_id": "rel_bad", "source_id": "missing", "target_id": "text_2",
                 "cv_relation": {"relation_type": "spatial_contains"}},
            ],
        }
        self.correction = {"elements": [
            {"element_id": "text_2", "corrected_text": "끝"},
            {"element_id": "text_1", "corrected_text": "시작"},
        ]}
        self.analysis = {
            "elements": [
                {"element_id": "text_2", "sequence_order": 2},
                {"element_id": "text_1", "sequence_order": 1},
            ],
            "relations": [{"relation_id": "rel_1", "relation_type": "logical_flow", "is_confirmed": None}],
        }

    def test_merge_by_id_and_skip_dangling_relation(self):
        original = deepcopy(self.document)
        model = FakeModel([self.correction, self.analysis])
        result = run_pipeline(self.document, model)
        self.assertEqual(self.document, original)
        self.assertEqual(result["elements"][1]["cv_result"]["raw_text"], "끗")
        self.assertEqual(result["elements"][1]["nlp_result"]["corrected_text"], "끝")
        self.assertEqual(result["elements"][0]["nlp_result"]["sequence_order"], 1)
        self.assertIsNone(result["elements"][2]["nlp_result"]["sequence_order"])
        self.assertEqual(result["relations"][0]["cv_relation"], original["relations"][0]["cv_relation"])
        self.assertIsNone(result["relations"][1]["nlp_relation"]["is_confirmed"])
        self.assertEqual(result["relations"][1]["nlp_relation"]["validation_error"], "missing_element_reference")
        self.assertEqual([r["relation_id"] for r in model.payloads[1]["relations"]], ["rel_1"])
        self.assertEqual(result["nlp_pipeline"]["mode"], "model")

    def test_reject_invalid_model_output(self):
        mutations = [
            ("elements", "element_id", "unknown"),
            ("elements", "sequence_order", True),
            ("relations", "is_confirmed", "true"),
            ("relations", "relation_type", "invented"),
        ]
        for field, key, value in mutations:
            with self.subTest(field=field, key=key):
                analysis = deepcopy(self.analysis)
                analysis[field][0][key] = value
                with self.assertRaises(ValueError):
                    run_pipeline(self.document, FakeModel([self.correction, analysis]))
        for invalid_elements in ([], [self.correction["elements"][0]] * 2,
                                 [{"element_id": "text_1", "corrected_text": 42}, self.correction["elements"][0]]):
            with self.subTest(elements=invalid_elements):
                with self.assertRaises(ValueError):
                    run_pipeline(self.document, FakeModel([{"elements": invalid_elements}, self.analysis]))

    def test_image_mode_reaches_both_stages(self):
        model = FakeModel([self.correction, self.analysis])
        model.image = object()
        result = run_pipeline(self.document, model)
        self.assertEqual(result["nlp_pipeline"]["input_type"], "image_and_json")
        self.assertTrue(all("attached document image" in instruction for instruction in model.instructions))
        self.assertTrue(all("There is no image" not in instruction for instruction in model.instructions))

    def test_parse_json_and_validate_input(self):
        self.assertEqual(parse_model_json('```json\n{"elements": []}\n```'), {"elements": []})
        for invalid in ('{"elements":', '[]', 'explanation {"elements": []}'):
            with self.subTest(response=invalid), self.assertRaises(ValueError):
                parse_model_json(invalid)
        duplicate = deepcopy(self.document)
        duplicate["elements"].append(duplicate["elements"][0])
        with self.assertRaises(ValueError):
            run_pipeline(duplicate)


if __name__ == "__main__":
    unittest.main()
