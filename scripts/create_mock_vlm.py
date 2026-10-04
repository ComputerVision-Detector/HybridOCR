"""이미지와 좌표가 일치하는 합성 순서도 mock 생성. 저장소 루트에서 실행."""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def create_mock(root: Path = ROOT) -> tuple[Path, Path, Path]:
    image = Image.new("RGB", (768, 384), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 36)
    except OSError:
        font = ImageFont.load_default(size=36)
    elements = []

    def add_element(element_id, class_type, bounds, raw_text=None, features=None):
        x1, y1, x2, y2 = bounds
        elements.append({
            "element_id": element_id, "class_type": class_type,
            "bounding_box": {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1},
            "cv_result": {"raw_text": raw_text, "confidence_score": 0.99,
                          "additional_features": features or {}},
        })

    for index, (bounds, text, raw_text) in enumerate([
        ((64, 128, 288, 256), "Start", "Strat"),
        ((480, 128, 704, 256), "End", "End"),
    ], start=1):
        draw.rounded_rectangle(bounds, radius=16, outline="black", width=4)
        center = ((bounds[0] + bounds[2]) // 2, (bounds[1] + bounds[3]) // 2)
        draw.text(center, text, fill="black", font=font, anchor="mm")
        add_element(f"box_{index}", "Box", bounds)
        add_element(f"text_{index}", "Text", draw.textbbox(center, text, font=font, anchor="mm"), raw_text)
    draw.line((288, 192, 480, 192), fill="black", width=4)
    draw.polygon([(480, 192), (460, 178), (460, 206)], fill="black")
    add_element("arrow_1", "Arrow", (288, 178, 480, 206), features={"direction": "right"})
    relations = []
    for index, (source, target, kind, via) in enumerate([
        ("box_1", "box_2", "spatial_connects", "arrow_1"),
        ("box_1", "text_1", "spatial_contains", None),
        ("box_2", "text_2", "spatial_contains", None),
    ], start=1):
        relations.append({"relation_id": f"rel_{index}", "source_id": source, "target_id": target,
                          "cv_relation": {"relation_type": kind, "via_id": via, "is_valid_geometry": True}})
    document = {"document_id": "mock_vlm_flow", "metadata": {
        "image_width": 768, "image_height": 384, "synthetic": True,
    }, "elements": elements, "relations": relations}
    image_path = root / "data/mock_images/vlm_flow.png"
    json_path = root / "data/mock_json/vlm_flow.json"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path)
    json_path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # 정답은 그림의 의도한 내용에서 작성하며 모델 입력에는 포함하지 않습니다.
    reference = {"document_id": document["document_id"], "element_ids": [e["element_id"] for e in elements],
                 "texts": {"text_1": "Start", "text_2": "End"},
                 "reading_order_pairs": [["text_1", "text_2"]],
                 "relations": [{"source_id": r["source_id"], "target_id": r["target_id"],
                                "relation_type": "connects" if r["cv_relation"]["relation_type"] == "spatial_connects" else "contains"}
                               for r in relations], "errors": [], "synthetic": True}
    reference_path = root / "data/ground_truth/vlm_flow.json"
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    reference_path.write_text(json.dumps(reference, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return json_path, image_path, reference_path


if __name__ == "__main__":
    for path in create_mock():
        print(path)
