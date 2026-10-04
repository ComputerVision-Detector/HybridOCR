"""OCR 원문을 보존하며 LLM 교정 결과를 NLP 필드에 기록합니다."""

from nlp.qwen import index_predictions


def correct_text(document: dict, model=None) -> dict:
    for element in document["elements"]:
        result = element.setdefault("nlp_result", {})
        result.setdefault("corrected_text", None)
    text_elements = [element for element in document["elements"] if element.get("class_type") == "Text"]
    if model is None or not text_elements:
        return document
    image_instruction = (
        "Use the attached document image to verify text inside each supplied bounding box. "
        if getattr(model, "image", None) is not None else "No image is supplied; use text context only. "
    )
    response = model.generate_json(
        image_instruction + "Correct Korean/English OCR conservatively using the surrounding text. "
        "Preserve names, numbers and formulas unless there is clear evidence. "
        "Return {\"elements\":[{\"element_id\":\"...\",\"corrected_text\":\"...\"}]}. "
        "Include every supplied Text element exactly once. Use null when unreadable; "
        "return the original text when no correction is justified.",
        {"metadata": document.get("metadata", {}),
         "elements": [{"element_id": e["element_id"], "bounding_box": e.get("bounding_box"),
                       "raw_text": e.get("cv_result", {}).get("raw_text")} for e in text_elements]},
    )
    predictions = index_predictions(response, "elements", "element_id", {e["element_id"] for e in text_elements})
    for element in text_elements:
        prediction = predictions[element["element_id"]]
        if "corrected_text" not in prediction or (prediction["corrected_text"] is not None and not isinstance(prediction["corrected_text"], str)):
            raise ValueError("corrected_text는 문자열 또는 null이어야 합니다.")
        element["nlp_result"]["corrected_text"] = prediction["corrected_text"]
    return document
