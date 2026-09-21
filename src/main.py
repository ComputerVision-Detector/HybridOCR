# File location: src/main.py
import json
import os
from datetime import datetime, timezone
from typing import Dict, Any

# CV 모듈 임포트
from cv.preprocessing.image_processor import ImageProcessor
from cv.detection.detector import ElementDetector
from cv.ocr.recognizer import TextRecognizer
from cv.relation.spatial_analyzer import SpatialRelationAnalyzer

class HybridOCRPipeline:
    def __init__(self, detection_model_path: str):
        """
        CV 파이프라인의 각 모듈을 초기화합니다.
        """
        self.preprocessor = ImageProcessor(blur_threshold=100.0)
        self.detector = ElementDetector(model_path=detection_model_path, conf_threshold=0.5)
        self.recognizer = TextRecognizer(gpu=True)
        self.relation_analyzer = SpatialRelationAnalyzer(
            containment_threshold=0.8,
            connection_distance_threshold=100.0
        )

    def run(self, image_path: str, document_id: str) -> Dict[str, Any]:
        """
        입력 이미지에 대해 전처리, 객체 탐지, OCR, 공간 관계 분석을 순차적으로 실행하여
        CV 결과가 포함된 JSON 형태의 딕셔너리를 반환합니다.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"입력 이미지를 찾을 수 없습니다: {image_path}")

        # 1. 전처리 (Preprocessing)
        prep_result = self.preprocessor.process(image_path)
        
        if prep_result["status"] == "RETAKE_REQUIRED":
            return {
                "document_id": document_id,
                "error": "RETAKE_REQUIRED",
                "message": f"이미지가 너무 흐립니다. (variance: {prep_result['variance']:.2f})"
            }

        processed_image = prep_result["processed_image"]
        img_height, img_width = processed_image.shape[:2]

        # 2. 요소 검출 (Detection)
        elements = self.detector.detect(processed_image)

        # 3. 텍스트 인식 (OCR)
        elements = self.recognizer.process(processed_image, elements)

        # 4. 공간 관계 분석 (Relation Analysis)
        relations = self.relation_analyzer.analyze(elements)

        # 5. 결과 스키마 조립
        result_json = {
            "document_id": document_id,
            "metadata": {
                "image_width": img_width,
                "image_height": img_height,
                "processing_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            },
            "elements": elements,
            "relations": relations
        }

        return result_json

if __name__ == "__main__":
    # 실행 설정
    INPUT_IMAGE_PATH = "../data/raw_images/sample_001.jpg"
    OUTPUT_JSON_PATH = "../data/output/cv_result_001.json"
    MODEL_PATH = "../data/models/yolov8_custom.pt" # 사전학습된 탐지 모델 경로

    pipeline = HybridOCRPipeline(detection_model_path=MODEL_PATH)
    
    try:
        final_result = pipeline.run(image_path=INPUT_IMAGE_PATH, document_id="doc_001")
        
        if "error" in final_result:
            print(f"처리 중단: {final_result['message']}")
        else:
            # 최종 결과를 파일로 저장 (NLP 모듈의 입력으로 사용)
            os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
            with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
                json.dump(final_result, f, ensure_ascii=False, indent=2)
            
            print(f"CV 파이프라인 처리가 완료되었습니다. 결과가 저장되었습니다: {OUTPUT_JSON_PATH}")
            
    except Exception as e:
        print(f"오류 발생: {e}")