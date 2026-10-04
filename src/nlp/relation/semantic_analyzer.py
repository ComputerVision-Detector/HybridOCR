"""텍스트·좌표·CV 관계를 바탕으로 읽기 순서와 의미 관계를 추론합니다."""

from nlp.qwen import index_predictions


def analyze_relations(document: dict, model=None) -> dict:
    for element in document["elements"]:
        element["nlp_result"].setdefault("sequence_order", None)
    for relation in document["relations"]:
        relation.setdefault("nlp_relation", {}).setdefault("is_confirmed", None)
    if model is None or not document["elements"]:
        return document
    element_ids = {e["element_id"] for e in document["elements"]}
    text_ids = {e["element_id"] for e in document["elements"] if e.get("class_type") == "Text"}
    valid_relations = []
    for relation in document["relations"]:
        references = [relation.get("source_id"), relation.get("target_id")]
        via_id = relation.get("cv_relation", {}).get("via_id")
        if via_id is not None:
            references.append(via_id)
        if any(reference not in element_ids for reference in references):
            relation["nlp_relation"]["is_confirmed"] = None
            relation["nlp_relation"]["validation_error"] = "missing_element_reference"
        else:
            valid_relations.append(relation)
    image_instruction = (
        "Use the attached document image to verify arrow direction, endpoints and box containment. "
        if getattr(model, "image", None) is not None
        else "There is no image: do not claim to have visually verified handwriting or arrows. "
    )
    response = model.generate_json(
        image_instruction + "Infer reading order from element coordinates and text, and review supplied spatial relations. "
        "Return a completed copy of output_template. Fill only sequence_order, relation_type and is_confirmed. "
        "Template strings are placeholders: replace them with your predictions of the required JSON types. "
        "Preserve every ID and both array lengths exactly; never omit, duplicate or combine entries. "
        "The output elements array MUST contain ONLY the IDs in text_element_ids, each exactly once. "
        "Arrow and Box elements are context only and MUST NOT appear in the output elements array. "
        "Include every supplied relation exactly once, even when several relations share an endpoint. "
        "relation_id identifies a relation; source_id, target_id and via_id identify elements and must not replace it. "
        "Do not invent IDs or relations. "
        "sequence_order is a unique positive integer for clearly ordered Text elements; use null for ambiguous order. "
        "relation_type is logical_flow, semantic_grouping or null. is_confirmed is true, false or null. "
        "A true is_confirmed requires a non-null relation_type. "
        "Use null when the supplied data is insufficient.",
        {"metadata": document.get("metadata", {}), "text_element_ids": sorted(text_ids),
         "elements": [{"element_id": e["element_id"], "class_type": e.get("class_type"),
                       "bounding_box": e.get("bounding_box"), "cv_result": e.get("cv_result"),
                       "nlp_result": e["nlp_result"]} for e in document["elements"]],
         "relations": [{"relation_id": r["relation_id"], "source_id": r.get("source_id"),
                        "target_id": r.get("target_id"), "cv_relation": r.get("cv_relation")} for r in valid_relations],
         "output_template": {
             "elements": [{"element_id": e["element_id"], "sequence_order": "<positive integer or null>"}
                          for e in document["elements"] if e["element_id"] in text_ids],
             "relations": [{"relation_id": r["relation_id"], "relation_type": "<logical_flow, semantic_grouping or null>",
                            "is_confirmed": "<boolean or null>"}
                           for r in valid_relations],
         }},
    )
    orders = index_predictions(response, "elements", "element_id", text_ids)
    relations = index_predictions(response, "relations", "relation_id", {r["relation_id"] for r in valid_relations})
    assigned_orders = set()
    for element in document["elements"]:
        if element["element_id"] not in text_ids:
            element["nlp_result"]["sequence_order"] = None
            continue
        prediction = orders[element["element_id"]]
        if "sequence_order" not in prediction:
            raise ValueError("sequence_order 필드가 필요합니다.")
        order = prediction["sequence_order"]
        if order is not None:
            if type(order) is not int or order < 1 or element.get("class_type") != "Text" or order in assigned_orders:
                raise ValueError("sequence_order는 Text 요소의 중복 없는 양의 정수 또는 null이어야 합니다.")
            assigned_orders.add(order)
        element["nlp_result"]["sequence_order"] = order
    for relation in valid_relations:
        prediction = relations[relation["relation_id"]]
        if "is_confirmed" not in prediction or (prediction["is_confirmed"] is not None and type(prediction["is_confirmed"]) is not bool):
            raise ValueError("is_confirmed는 boolean 또는 null이어야 합니다.")
        if "relation_type" not in prediction or prediction["relation_type"] not in (None, "logical_flow", "semantic_grouping"):
            raise ValueError("알 수 없는 의미 관계 유형입니다.")
        if prediction["is_confirmed"] is True and prediction["relation_type"] is None:
            raise ValueError("확인된 관계에는 relation_type이 필요합니다.")
        relation["nlp_relation"].update({key: prediction[key] for key in ("relation_type", "is_confirmed")})
        relation["nlp_relation"].pop("validation_error", None)
    return document
