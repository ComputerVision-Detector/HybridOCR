# File location: HybridOCR/scripts/
# File name: push_raw_images.py

from datasets import load_dataset

def upload_raw_images():
    # 1. 이미지가 모여있는 로컬 디렉토리 경로
    local_data_dir = "C:/Users/hk100/Desktop/HybridOCR/data/raw_images/custom_notes_temp"
    
    # 2. 본인의 Hugging Face 저장소 이름
    repo_id = "hakuuung/HybridOCR_CustomNotes"
    
    # 3. metadata.jsonl 없이 이미지만 로드 (자동으로 'image' 컬럼만 생성됨)
    dataset = load_dataset("imagefolder", data_dir=local_data_dir)
    
    # 4. Hub에 비공개 업로드
    dataset.push_to_hub(repo_id, private=True)
    
    print("원본 이미지 업로드 완료.")

if __name__ == "__main__":
    upload_raw_images()