# 파일 위치: src/cv/ocr/
# 파일 명: baseline_evaluator.py

import json
import numpy as np
from datasets import load_dataset
from paddleocr import PaddleOCR
from datetime import datetime, timezone

class BaselineEvaluator:
    def __init__(self, repo_id: str):
        self.repo_id = repo_id
        self.ocr = PaddleOCR(use_angle_cls=True, lang='korean')

    def process_image(self, index: int = 0) -> dict:
        print(f"[{self.repo_id}] 데이터셋 로드 및 {index}번 이미지 추론 시작...")
        dataset = load_dataset(self.repo_id, split="train")
        image_obj = dataset[index]["image"]
        
        # PIL Image를 numpy 배열(RGB)로 변환
        image_np = np.array(image_obj.convert('RGB'))

        # 오류 수정: cls=True 파라미터 제거
        result = self.ocr.ocr(image_np)

        document_id = f"doc_{index:03d}"
        elements = []

        if result and result[0]:
            for i, line in enumerate(result[0]):
                box_coords = line[0]
                text = line[1][0]
                confidence = line[1][1]

                # PaddleOCR의 4점 좌표 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]를 x, y, width, height로 변환
                x_coords = [pt[0] for pt in box_coords]
                y_coords = [pt[1] for pt in box_coords]
                x = min(x_coords)
                y = min(y_coords)
                w = max(x_coords) - x
                h = max(y_coords) - y

                element = {
                    "element_id": f"elem_{i+1:03d}",
                    "class_type": "Text",
                    "bounding_box": {
                        "x": round(float(x), 2),
                        "y": round(float(y), 2),
                        "width": round(float(w), 2),
                        "height": round(float(h), 2)
                    },
                    "cv_result": {
                        "raw_text": text,
                        "confidence_score": round(float(confidence), 4),
                        "additional_features": {}
                    },
                    "nlp_result": {
                        "corrected_text": None,
                        "sequence_order": None,
                        "additional_features": {}
                    }
                }
                elements.append(element)

        # NLP 파트와 약속된 JSON 스키마로 조립
        output_json = {
            "document_id": document_id,
            "metadata": {
                "image_width": image_obj.width,
                "image_height": image_obj.height,
                "processing_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            },
            "elements": elements,
            "relations": []
        }

        return output_json

if __name__ == "__main__":
    REPO_ID = "hakuuung/HybridOCR_CustomNotes"
    evaluator = BaselineEvaluator(REPO_ID)
    
    # 첫 번째 이미지에 대한 JSON 생성
    json_result = evaluator.process_image(index=0)
    
    # 결과 출력 및 저장
    output_path = "../../data/output/baseline_result_000.json"
    print(json.dumps(json_result, ensure_ascii=False, indent=2))
    
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_result, f, ensure_ascii=False, indent=2)