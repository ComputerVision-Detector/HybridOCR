# File location: src/cv/detection/detector.py
import cv2
import numpy as np
from typing import List, Dict, Any
# 사전학습 모델 적용을 위해 ultralytics YOLO 사용을 가정한 뼈대 코드
from ultralytics import YOLO

class ElementDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        """
        사전학습된 객체 탐지 모델을 초기화합니다.
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
        # 탐지 모델의 클래스 인덱스를 스키마의 class_type으로 맵핑
        # 모델 학습 데이터셋의 라벨링 순서에 맞추어 수정이 필요합니다.
        self.class_map = {
            0: "Text",
            1: "Arrow",
            2: "Box"
        }

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        이미지에서 Text, Arrow, Box를 검출하여 사전 정의된 JSON 스키마 형태의 리스트로 반환합니다.
        """
        results = self.model(image, conf=self.conf_threshold, verbose=False)
        elements = []

        if not results or len(results) == 0:
            return elements

        # 첫 번째 배치 결과 추출
        boxes = results[0].boxes

        for idx, box in enumerate(boxes):
            # xyxy (좌상단 우하단) 좌표를 좌상단 기준 x, y, width, height로 변환
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x = float(x1)
            y = float(y1)
            width = float(x2 - x1)
            height = float(y2 - y1)

            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            class_type = self.class_map.get(class_id, "Unknown")

            # JSON 스키마의 elements 배열 원소 규격에 맞추어 객체 생성
            element_data = {
                "element_id": f"elem_{idx+1:03d}",
                "class_type": class_type,
                "bounding_box": {
                    "x": round(x, 2),
                    "y": round(y, 2),
                    "width": round(width, 2),
                    "height": round(height, 2)
                },
                "cv_result": {
                    "raw_text": None,  # OCR 모듈 파이프라인에서 이후 갱신됨
                    "confidence_score": round(confidence, 4),
                    "additional_features": {}
                },
                "nlp_result": {
                    "corrected_text": None,
                    "sequence_order": None,
                    "additional_features": {}
                }
            }
            elements.append(element_data)

        return elements