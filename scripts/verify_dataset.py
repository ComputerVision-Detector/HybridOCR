# File location: HybridOCR/scripts/verify_dataset.py
# File name: verify_dataset.py

from datasets import load_dataset
import random

def verify_uploaded_dataset(repo_id):
    print(f"[{repo_id}] 데이터셋 로드 시도 중...")
    
    # 1. 데이터셋 다운로드 및 로드 (캐시 활용)
    dataset = load_dataset(repo_id, split="train")
    
    total_count = len(dataset)
    print(f"정상 로드 완료. 총 이미지 수: {total_count}장\n")
    
    if total_count == 0:
        print("데이터셋에 이미지가 존재하지 않습니다.")
        return

    # 2. 첫 번째 이미지와 무작위 2장의 이미지 정보 출력
    print("--- 이미지 정보 ---")

    random_indices = random.sample(range(1, total_count), min(2, total_count - 1))
    sample_indices = [0] + random_indices

    for i in sample_indices:
        image_obj = dataset[i]["image"]
        print(f"Index {i}: 크기 {image_obj.size}, 모드 {image_obj.mode}, 포맷 {image_obj.format}")

    # 3. 이미지 화면 출력 검증
    print("\n이미지를 화면에 출력합니다.")
    
    sample_image = []

    for i in sample_indices:
        sample_image.append(dataset[i]["image"])
        sample_image[-1].show()

if __name__ == "__main__":
    REPO_ID = "hakuuung/HybridOCR_CustomNotes"
    verify_uploaded_dataset(REPO_ID)