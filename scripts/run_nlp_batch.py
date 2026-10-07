"""같은 실행 조건의 CV/OCR JSON 폴더 처리. 모델은 한 번 로드합니다."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nlp_pipeline import run_pipeline, validate_document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="이번 실행용 비어 있는 폴더")
    parser.add_argument("--use-model", action="store_true")
    parser.add_argument("--image-dir", type=Path, help="입력 JSON과 같은 상대 경로·파일명인 PNG/JPG/JPEG 폴더")
    args = parser.parse_args()
    try:
        source, destination = args.input_dir.resolve(), args.output_dir.resolve()
        if source == destination or source in destination.parents or destination in source.parents:
            raise ValueError("입력과 출력은 서로 포함하지 않는 별도 폴더여야 합니다.")
        if args.image_dir is not None:
            images_root = args.image_dir.resolve()
            if images_root == destination or images_root in destination.parents or destination in images_root.parents:
                raise ValueError("이미지와 출력도 별도 폴더여야 합니다.")
        if destination.exists() and any(destination.iterdir()):
            raise ValueError("출력 폴더가 비어 있지 않습니다. 새로운 실행 폴더를 지정하세요.")
        paths = sorted(source.rglob("*.json"))
        if not paths:
            raise ValueError("입력 JSON이 없습니다.")
        if args.image_dir is not None and not args.use_model:
            raise ValueError("--image-dir는 --use-model과 함께 사용합니다.")
        documents, images, seen = [], [], set()
        for path in paths:
            document = json.loads(path.read_text(encoding="utf-8-sig"))
            validate_document(document)
            if document["document_id"] in seen:
                raise ValueError("입력 document_id가 중복됩니다.")
            seen.add(document["document_id"])
            documents.append(document)
            if args.image_dir is not None:
                from nlp.qwen import load_document_image
                matches = [args.image_dir / path.relative_to(source).with_suffix(suffix) for suffix in (".png", ".jpg", ".jpeg")]
                matches = [match for match in matches if match.is_file()]
                if len(matches) != 1:
                    raise ValueError(f"이미지 한 개가 대응해야 합니다: {path.name}")
                load_document_image(document, matches[0]).close()
                images.append(matches[0])
        model = None
        if args.use_model:
            from nlp.qwen import QwenModel
            model = QwenModel(image=load_document_image(documents[0], images[0]) if images else None)
        outputs = []
        for index, (path, document) in enumerate(zip(paths, documents)):
            if model is not None and images:
                model.image.close()
                model.image = load_document_image(document, images[index])
            print(f"처리: {document['document_id']}", flush=True)
            outputs.append((destination / path.relative_to(source), run_pipeline(document, model)))
        # ponytail: 결과 JSON을 메모리에 보관. 대량 문서는 임시 디렉터리에 저장 후 이동하는 방식으로 확장.
        # 모든 추론이 성공한 뒤에 저장해 추론 실패 결과를 평가 폴더에 섞지 않습니다.
        for path, result in outputs:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(f"완료: {len(outputs)}문서 → {destination}")


if __name__ == "__main__":
    main()
