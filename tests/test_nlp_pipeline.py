"""Run from the repository root: py -m unittest discover -s tests"""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nlp_pipeline import run_pipeline


class PipelineTest(unittest.TestCase):
    def test_json_round_trip_and_input_protection(self):
        document = {
            "document_id": "mock_001",
            "elements": [{
                "element_id": "text_001",
                "class_type": "Text",
                "bounding_box": {"x": 10, "y": 20, "width": 100, "height": 30},
                "cv_result": {"raw_text": "테스트 원문", "confidence_score": 0.8},
            }],
            "relations": [],
            "custom_metadata": {"keep": True},
        }
        original = deepcopy(document)
        result = run_pipeline(document)
        self.assertEqual(document, original)
        self.assertEqual(result["elements"][0]["cv_result"], document["elements"][0]["cv_result"])
        self.assertIsNone(result["elements"][0]["nlp_result"]["corrected_text"])
        self.assertIsNone(result["elements"][0]["nlp_result"]["sequence_order"])
        self.assertEqual(result["custom_metadata"], document["custom_metadata"])
        self.assertEqual(result["nlp_pipeline"]["mode"], "scaffold")
        with self.assertRaises(ValueError):
            run_pipeline({"document_id": "bad", "elements": [], "relations": None})
        with TemporaryDirectory() as folder:
            source = Path(folder) / "mock.json"
            output = Path(folder) / "output" / "result.json"
            source.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8-sig")
            command = [sys.executable, str(ROOT / "src" / "nlp_pipeline.py"), "--input", str(source)]
            completed = subprocess.run(command + ["--output", str(output)], capture_output=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), result)
            rejected = subprocess.run(command + ["--output", str(source)], capture_output=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertEqual(json.loads(source.read_text(encoding="utf-8-sig")), document)


if __name__ == "__main__":
    unittest.main()
