"""이미지 읽기·좌표 기준 확인 및 VLM CLI 입력 보호."""

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nlp.qwen import load_document_image

try:
    from PIL import Image
except ImportError:
    Image = None


@unittest.skipIf(Image is None, "Pillow 필요: requirements.nlp.txt 설치")
class VLMInputTest(unittest.TestCase):
    def test_image_validation_and_cli_protection(self):
        with TemporaryDirectory() as folder:
            image_path = Path(folder) / "page.png"
            Image.new("RGB", (64, 64), "white").save(image_path)
            document = {"document_id": "image_test", "metadata": {"image_width": 64, "image_height": 64},
                        "elements": [], "relations": []}
            loaded = load_document_image(document, image_path)
            self.assertEqual(loaded.size, (64, 64))
            self.assertEqual(loaded.mode, "RGB")
            for metadata in (None, {}, {"image_width": True, "image_height": 64},
                             {"image_width": 128, "image_height": 64}):
                with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                    load_document_image(dict(document, metadata=metadata), image_path)
            rotated = Path(folder) / "rotated.jpg"
            exif = Image.Exif()
            exif[274] = 6
            Image.new("RGB", (64, 64)).save(rotated, exif=exif)
            with self.assertRaises(ValueError):
                load_document_image(document, rotated)
            source = Path(folder) / "page.json"
            source.write_text(json.dumps(document), encoding="utf-8")
            original_image = image_path.read_bytes()
            command = [sys.executable, str(ROOT / "src/nlp_pipeline.py"), "--input", str(source),
                       "--image", str(image_path), "--output", str(image_path)]
            for flags in ([], ["--use-model"]):
                with self.subTest(flags=flags):
                    result = subprocess.run(command + flags, capture_output=True)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(image_path.read_bytes(), original_image)
            image_path.write_bytes(b"not an image")
            with self.assertRaises(OSError):
                load_document_image(document, image_path)


if __name__ == "__main__":
    unittest.main()
