# File location: src/cv/ocr/recognizer.py
import cv2
import numpy as np
import easyocr
from typing import List, Dict, Any

class TextRecognizer:
    def __init__(self, gpu: bool = True):
        """
        한글 및 영문 인식을 위한 EasyOCR 모델을 초기화합니다.
        """
        # 'ko', 'en' 언어팩 로드. GPU 사용 여부는 환경에 따라 설정합니다.
        self.reader = easyocr.Reader(['ko', 'en'], gpu=gpu)

    def crop_image(self, image: np.ndarray, bbox: Dict[str, float]) -> np.ndarray:
        """
        바운딩 박스 좌표를 기반으로 이미지에서 해당 영역을 잘라냅니다.
        """
        img_h, img_w = image.shape[:2]
        
        x = int(bbox["x"])
        y = int(bbox["y"])
        w = int(bbox["width"])
        h = int(bbox["height"])
        
        # 이미지 경계를 벗어나지 않도록 좌표 보정
        start_y = max(0, y)
        end_y = min(img_h, y + h)
        start_x = max(0, x)
        end_x = min(img_w, x + w)
        
        return image[start_y:end_y, start_x:end_x]

    def process(self, image: np.ndarray, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        elements 배열을 순회하며 'Text' 클래스인 경우 OCR을 수행하고 결과를 업데이트합니다.
        """
        for element in elements:
            if element.get("class_type") != "Text":
                continue
                
            bbox = element.get("bounding_box", {})
            if not bbox:
                continue

            cropped_img = self.crop_image(image, bbox)
            
            # 잘라낸 영역이 유효하지 않은 경우 건너뜀
            if cropped_img.size == 0:
                continue

            # OCR 수행 (detail=1은 바운딩 박스, 텍스트, 신뢰도를 모두 반환)
            ocr_results = self.reader.readtext(cropped_img, detail=1)
            
            if ocr_results:
                # 여러 줄이 인식될 경우 문자열 결합 및 신뢰도 평균 계산
                texts = []
                confidences = []
                for result in ocr_results:
                    _, text, conf = result
                    texts.append(text)
                    confidences.append(conf)
                
                merged_text = " ".join(texts)
                avg_confidence = sum(confidences) / len(confidences)
                
                # cv_result 갱신
                element["cv_result"]["raw_text"] = merged_text
                # Detection 모듈의 신뢰도와 OCR 모델의 신뢰도를 어떻게 처리할지에 따라 로직 수정 가능
                element["cv_result"]["additional_features"]["ocr_confidence"] = round(float(avg_confidence), 4)
            else:
                # 인식된 텍스트가 없을 경우
                element["cv_result"]["raw_text"] = ""
                element["cv_result"]["additional_features"]["ocr_confidence"] = 0.0

        return elements