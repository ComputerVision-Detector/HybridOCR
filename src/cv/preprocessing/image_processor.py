# src/cv/preprocessing/image_processor.py

import cv2
import numpy as np

class ImageProcessor:
    def __init__(self, blur_threshold=100.0):
        # 라플라시안 분산 임계값 설정 (이하일 경우 재촬영 요망)
        self.blur_threshold = blur_threshold

    def check_quality(self, image):
        """이미지의 흐림 정도를 측정하여 재촬영 필요 여부 반환"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_blurry = variance < self.blur_threshold
        return is_blurry, variance

    def correct_skew(self, image):
        """텍스트 영역의 윤곽선을 탐지하여 이미지 기울기 보정"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 텍스트 획을 강조하기 위한 이진화
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 픽셀 좌표를 추출하여 최소 면적 사각형 계산
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) == 0:
            return image
            
        angle = cv2.minAreaRect(coords)[-1]

        # 각도 정규화
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # 회전 변환 행렬 생성 및 적용
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        return rotated

    def adjust_illumination(self, image):
        """CLAHE를 이용한 국소적 대비 및 조명 균일화"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        limg = cv2.merge((cl, a, b))
        adjusted = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return adjusted

    def process(self, image_path):
        """전처리 파이프라인 통합 실행"""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")

        # 1. 품질 검사 (심한 흐림은 전처리 중단 및 플래그 반환)
        is_blurry, blur_variance = self.check_quality(image)
        if is_blurry:
            return {
                "status": "RETAKE_REQUIRED",
                "variance": blur_variance,
                "processed_image": None
            }

        # 2. 기울기 보정
        deskewed_image = self.correct_skew(image)

        # 3. 조명/음영 보정
        final_image = self.adjust_illumination(deskewed_image)

        return {
            "status": "SUCCESS",
            "variance": blur_variance,
            "processed_image": final_image
        }