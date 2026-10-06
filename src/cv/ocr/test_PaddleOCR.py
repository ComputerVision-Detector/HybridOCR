# from paddleocr import PaddleOCR
#
# ocr = PaddleOCR(use_angle_cls=True, lang='korean')
# result = ocr.predict('IMG_2728.JPG')
#
# print(type(result))
# print(result)

from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np

IMAGE_PATH = "IMG_2728.JPG"
OUTPUT_PATH = "ocr_result.jpg"

# --------------------------------------------------
# 1. PaddleOCR 초기화
# --------------------------------------------------
ocr = PaddleOCR(
    lang="korean",
    use_textline_orientation=True
)

# --------------------------------------------------
# 2. 이미지 불러오기
#    EXIF Orientation 정보를 실제 픽셀에 적용
# --------------------------------------------------
image = Image.open(IMAGE_PATH)
image = ImageOps.exif_transpose(image).convert("RGB")

# --------------------------------------------------
# 3. OCR 추론
#    시각화에 사용하는 이미지와 동일한 이미지 사용
# --------------------------------------------------
image_np = np.array(image)
result = ocr.predict(image_np)

# --------------------------------------------------
# 4. 이미지에 OCR 결과 표시
# --------------------------------------------------
draw = ImageDraw.Draw(image)

# Windows 기본 한글 폰트
font_path = "C:/Windows/Fonts/malgun.ttf"
font = ImageFont.truetype(font_path, 32)

for res in result:
    rec_texts = res["rec_texts"]
    rec_scores = res["rec_scores"]
    rec_boxes = res["rec_boxes"]

    for text, score, box in zip(
        rec_texts,
        rec_scores,
        rec_boxes
    ):
        # ------------------------------------------
        # Bounding Box 좌표
        # [x_min, y_min, x_max, y_max]
        # ------------------------------------------
        x1, y1, x2, y2 = map(int, box)

        # ------------------------------------------
        # Bounding Box 그리기
        # ------------------------------------------
        draw.rectangle(
            [(x1, y1), (x2, y2)],
            outline=(255, 0, 0),
            width=4
        )

        # ------------------------------------------
        # OCR 결과 + confidence
        # ------------------------------------------
        label = f"{text} ({score:.2f})"

        # 텍스트 크기 계산
        text_bbox = draw.textbbox(
            (0, 0),
            label,
            font=font
        )

        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # bbox 위에 표시
        label_y = max(
            0,
            y1 - text_height - 10
        )

        # ------------------------------------------
        # 텍스트 배경
        # ------------------------------------------
        draw.rectangle(
            [
                (x1, label_y),
                (
                    x1 + text_width + 10,
                    label_y + text_height + 10
                )
            ],
            fill=(255, 0, 0)
        )

        # ------------------------------------------
        # OCR 텍스트
        # ------------------------------------------
        draw.text(
            (x1 + 5, label_y + 3),
            label,
            font=font,
            fill=(255, 255, 255)
        )

# --------------------------------------------------
# 5. 결과 저장
# --------------------------------------------------
image.save(OUTPUT_PATH)

print(f"OCR 시각화 결과 저장 완료: {OUTPUT_PATH}")

# --------------------------------------------------
# 6. 결과 이미지 화면 출력
# --------------------------------------------------
image.show()