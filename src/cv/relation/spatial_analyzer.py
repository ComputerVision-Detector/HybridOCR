# File location: src/cv/relation/spatial_analyzer.py
import math
from typing import List, Dict, Any, Tuple

class SpatialRelationAnalyzer:
    def __init__(self, containment_threshold: float = 0.8, connection_distance_threshold: float = 100.0):
        """
        공간 좌표 기반의 물리적 관계를 추론하는 분석기를 초기화합니다.
        containment_threshold: 박스가 다른 요소를 포함했다고 판단할 최소 면적 비율 (Intersection over Area)
        connection_distance_threshold: 화살표 끝점과 연결 대상 간의 최대 허용 거리 (픽셀)
        """
        self.containment_threshold = containment_threshold
        self.connection_distance_threshold = connection_distance_threshold
        self.relation_counter = 1

    def _generate_relation_id(self) -> str:
        rel_id = f"rel_{self.relation_counter:03d}"
        self.relation_counter += 1
        return rel_id

    def _calculate_ioa(self, container_box: Dict[str, float], target_box: Dict[str, float]) -> float:
        """Target Box의 전체 면적 중 Container Box 내부에 포함된 면적의 비율(IoA)을 계산합니다."""
        x_left = max(container_box["x"], target_box["x"])
        y_top = max(container_box["y"], target_box["y"])
        x_right = min(container_box["x"] + container_box["width"], target_box["x"] + target_box["width"])
        y_bottom = min(container_box["y"] + container_box["height"], target_box["y"] + target_box["height"])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        target_area = target_box["width"] * target_box["height"]
        
        if target_area == 0:
            return 0.0
            
        return intersection_area / target_area

    def _get_element_center(self, bbox: Dict[str, float]) -> Tuple[float, float]:
        """바운딩 박스의 중심점 좌표를 반환합니다."""
        return (bbox["x"] + bbox["width"] / 2, bbox["y"] + bbox["height"] / 2)

    def _get_arrow_endpoints(self, bbox: Dict[str, float], direction: str) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        바운딩 박스와 방향 정보를 바탕으로 화살표의 시작점과 끝점 좌표를 추정합니다.
        (정밀도를 위해 향후 픽셀 기반 스켈레톤 추출 알고리즘으로 대체 가능)
        """
        x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        
        if direction == "right":
            return ((x, y + h / 2), (x + w, y + h / 2))
        elif direction == "left":
            return ((x + w, y + h / 2), (x, y + h / 2))
        elif direction == "down":
            return ((x + w / 2, y), (x + w / 2, y + h))
        elif direction == "up":
            return ((x + w / 2, y + h), (x + w / 2, y))
        
        # 방향이 불명확한 경우 중심점 반환
        center = self._get_element_center(bbox)
        return (center, center)

    def _calculate_distance(self, pt1: Tuple[float, float], pt2: Tuple[float, float]) -> float:
        """두 점 사이의 유클리디안 거리를 계산합니다."""
        return math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])

    def analyze(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """요소들의 바운딩 박스 정보를 바탕으로 시각적 관계(포함, 연결)를 추출합니다."""
        relations = []
        
        # 1. Box 포함 관계 (spatial_contains) 분석
        boxes = [e for e in elements if e.get("class_type") == "Box"]
        non_boxes = [e for e in elements if e.get("class_type") != "Box"]

        for box in boxes:
            for target in non_boxes:
                ioa = self._calculate_ioa(box["bounding_box"], target["bounding_box"])
                if ioa >= self.containment_threshold:
                    relations.append({
                        "relation_id": self._generate_relation_id(),
                        "source_id": box["element_id"],
                        "target_id": target["element_id"],
                        "cv_relation": {
                            "relation_type": "spatial_contains",
                            "via_id": None,
                            "overlap_ratio": round(ioa, 4),
                            "is_valid_geometry": True
                        },
                        "nlp_relation": {
                            "relation_type": None,
                            "is_confirmed": None
                        }
                    })

        # 2. Arrow 연결 관계 (spatial_connects) 분석
        arrows = [e for e in elements if e.get("class_type") == "Arrow"]
        candidates = [e for e in elements if e.get("class_type") in ["Text", "Box"]]

        for arrow in arrows:
            direction = arrow.get("cv_result", {}).get("additional_features", {}).get("direction", "unknown")
            start_pt, end_pt = self._get_arrow_endpoints(arrow["bounding_box"], direction)
            
            closest_source = None
            closest_target = None
            min_source_dist = float('inf')
            min_target_dist = float('inf')

            for candidate in candidates:
                cand_center = self._get_element_center(candidate["bounding_box"])
                
                dist_to_start = self._calculate_distance(start_pt, cand_center)
                if dist_to_start < min_source_dist:
                    min_source_dist = dist_to_start
                    closest_source = candidate
                    
                dist_to_end = self._calculate_distance(end_pt, cand_center)
                if dist_to_end < min_target_dist:
                    min_target_dist = dist_to_end
                    closest_target = candidate

            # 거리가 임계값 이내이고 출발지와 도착지가 서로 다른 객체일 경우 관계 성립
            if (min_source_dist <= self.connection_distance_threshold and 
                min_target_dist <= self.connection_distance_threshold and 
                closest_source and closest_target and 
                closest_source["element_id"] != closest_target["element_id"]):
                
                avg_distance = (min_source_dist + min_target_dist) / 2
                
                relations.append({
                    "relation_id": self._generate_relation_id(),
                    "source_id": closest_source["element_id"],
                    "target_id": closest_target["element_id"],
                    "cv_relation": {
                        "relation_type": "spatial_connects",
                        "via_id": arrow["element_id"],
                        "distance": round(avg_distance, 2),
                        "is_valid_geometry": True
                    },
                    "nlp_relation": {
                        "relation_type": None,
                        "is_confirmed": None
                    }
                })

        return relations