"""순서도 그래프와 규칙 기반 오류 후보. CV 결과를 수정하지 않습니다."""

from collections import deque

from nlp.qwen import index_predictions

ROLES = {"start", "end", "process", "decision", "unknown"}


def relation_status(relation):
    confirmed = relation.get("nlp_relation", {}).get("is_confirmed")
    if confirmed is not None and type(confirmed) is not bool:
        raise ValueError("is_confirmed는 boolean 또는 null이어야 합니다.")
    return "confirmed" if confirmed is True else "rejected" if confirmed is False else "unreviewed"


def build_graph(document):
    elements = {e["element_id"]: e for e in document["elements"]}
    contains, connections, invalid = [], [], []
    for relation in document["relations"]:
        cv = relation.get("cv_relation", {})
        refs = [relation["source_id"], relation["target_id"]]
        if cv.get("via_id") is not None:
            refs.append(cv["via_id"])
        if any(ref not in elements for ref in refs):
            invalid.append({"relation_id": relation["relation_id"], "missing_ids": sorted(set(refs) - elements.keys()),
                            "element_ids": sorted(set(refs) & elements.keys())})
            continue
        edge = {"relation_id": relation["relation_id"], "source_id": refs[0], "target_id": refs[1],
                "via_id": cv.get("via_id"), "status": relation_status(relation)}
        if cv.get("relation_type") == "spatial_contains":
            contains.append(edge)
        elif cv.get("relation_type") == "spatial_connects":
            connections.append(edge)
    parents = {}
    for edge in contains:
        if edge["status"] != "rejected" and elements[edge["source_id"]]["class_type"] == "Box" and elements[edge["target_id"]]["class_type"] == "Text":
            parents.setdefault(edge["target_id"], set()).add(edge["source_id"])
    # 여러 박스에 속한 Text는 임의로 한 박스에 귀속하지 않습니다.
    owners = {key: next(iter(values)) for key, values in parents.items() if len(values) == 1}
    nodes = []
    for item_id, element in elements.items():
        if element["class_type"] not in ("Box", "Text") or item_id in owners:
            continue
        text_ids = [key for key, owner in owners.items() if owner == item_id]
        if element["class_type"] == "Text":
            text_ids = [item_id]
        labels = []
        for text_id in text_ids:
            item = elements[text_id]
            corrected = item.get("nlp_result", {}).get("corrected_text")
            labels.append(corrected if corrected is not None else item.get("cv_result", {}).get("raw_text") or "")
        text = " ".join(labels).strip()
        features = element.get("cv_result", {}).get("additional_features", {})
        role = features.get("flow_role", "unknown")
        if not isinstance(role, str) or role not in ROLES:
            raise ValueError("flow_role은 start/end/process/decision/unknown이어야 합니다.")
        source = "cv_hint" if role != "unknown" else "unknown"
        if role == "unknown":
            label = text.casefold().strip()
            if label in ("start", "시작"):
                role, source = "start", "text_rule"
            elif label in ("end", "stop", "끝", "종료"):
                role, source = "end", "text_rule"
            elif features.get("shape") == "diamond":
                role, source = "decision", "shape_rule"
        nodes.append({"element_id": item_id, "text": text, "text_element_ids": text_ids,
                      "role": role, "role_source": source, "bounding_box": element.get("bounding_box")})
    node_ids = {n["element_id"] for n in nodes}
    for edge in connections:
        edge["original_source_id"], edge["original_target_id"] = edge["source_id"], edge["target_id"]
        edge["source_id"] = owners.get(edge["source_id"], edge["source_id"])
        edge["target_id"] = owners.get(edge["target_id"], edge["target_id"])
    usable = [edge for edge in connections if {edge["source_id"], edge["target_id"]} <= node_ids]
    return {"nodes": nodes, "edges": usable, "contains": contains, "invalid_relations": invalid,
            "unmapped_relation_ids": [e["relation_id"] for e in connections if e not in usable],
            "projection_basis": "non-rejected CV containment; unreviewed membership is provisional"}


def find_candidates(graph, rules):
    nodes = {n["element_id"]: n for n in graph["nodes"]}
    outgoing, incoming = {key: set() for key in nodes}, {key: set() for key in nodes}
    branches = {key: set() for key in nodes}
    # 보류 관계도 가능한 연결로 포함해 보류만으로 누락 오류를 만들지 않습니다.
    for edge in graph["edges"]:
        if edge["status"] != "rejected":
            outgoing[edge["source_id"]].add(edge["target_id"])
            incoming[edge["target_id"]].add(edge["source_id"])
            branches[edge["source_id"]].add(edge["via_id"] or edge["relation_id"])
    candidates = []

    def add(kind, ids, reason, suggestion):
        candidates.append({"error_id": f"flow_error_{len(candidates) + 1:03d}", "error_type": kind,
                           "element_ids": sorted(ids), "reason": reason, "suggestion": suggestion,
                           "is_confirmed": None, "review_source": "rules"})

    starts = [key for key, node in nodes.items() if node["role"] == "start"]
    ends = [key for key, node in nodes.items() if node["role"] == "end"]
    for relation in graph["invalid_relations"]:
        if relation["element_ids"]:
            add("invalid_reference", relation["element_ids"], "관계가 검출 목록에 없는 요소를 참조합니다.", "검출 결과와 관계 참조를 확인하세요.")
    if nodes and rules["require_terminal_nodes"]:
        if not starts:
            add("missing_start", nodes.keys(), "시작 역할의 노드를 확인하지 못했습니다.", "시작 노드 또는 역할 인식 결과를 확인하세요.")
        if not ends:
            add("missing_end", nodes.keys(), "종료 역할의 노드를 확인하지 못했습니다.", "종료 노드 또는 문서 규칙을 확인하세요.")
    for key, node in nodes.items():
        role = node["role"]
        if role == "start" and incoming[key]:
            add("start_has_incoming", [key], "시작 노드로 들어오는 연결이 있습니다.", "시작 역할과 연결 방향을 확인하세요.")
        if role == "end" and outgoing[key]:
            add("end_has_outgoing", [key], "종료 노드에서 나가는 연결이 있습니다.", "종료 역할과 연결 방향을 확인하세요.")
        if role == "decision" and len(branches[key]) < rules["decision_min_branches"]:
            add("decision_branch_missing", [key], "조건 분기의 연결 수가 규칙보다 적습니다.", "분기 연결 또는 도착 노드 인식 결과를 확인하세요.")
        elif role in ("start", "process") and not outgoing[key]:
            add("missing_connection", [key], "종료가 아닌 노드의 다음 연결을 확인하지 못했습니다.", "누락된 화살표 또는 연결 대상을 확인하세요.")
    if starts and rules["check_reachability"]:
        reached = set(starts)
        pending = deque(starts)
        while pending:
            current = pending.popleft()
            for target in outgoing[current] - reached:
                reached.add(target)
                pending.append(target)
        for key in sorted(nodes.keys() - reached):
            add("unreachable_node", [key], "시작 노드에서 도달하는 경로를 확인하지 못했습니다.", "연결 누락·방향·시작 역할을 확인하세요.")
    return candidates


def flowchart_rules(metadata):
    rules = {"require_terminal_nodes": False, "decision_min_branches": 2, "check_reachability": True}
    supplied = metadata.get("flowchart_rules", {})
    if not isinstance(supplied, dict) or set(supplied) - rules.keys():
        raise ValueError("지원하지 않는 flowchart_rules입니다.")
    rules.update(supplied)
    if any(type(rules[k]) is not bool for k in ("require_terminal_nodes", "check_reachability")):
        raise ValueError("순서도 규칙의 활성화 값은 boolean이어야 합니다.")
    if type(rules["decision_min_branches"]) is not int or rules["decision_min_branches"] < 2:
        raise ValueError("decision_min_branches는 2 이상의 정수여야 합니다.")
    return rules


def repair_pairs(graph, candidates, confirmed_only=False):
    nodes = {node["element_id"]: node for node in graph["nodes"]}
    existing = {(e["source_id"], e["target_id"]) for e in graph["edges"] if e["status"] != "rejected"}
    result = {}
    for error in candidates:
        if confirmed_only and error["is_confirmed"] is not True:
            continue
        if error["error_type"] in ("missing_connection", "decision_branch_missing"):
            sources, targets = error["element_ids"], nodes.keys()
        elif error["error_type"] == "unreachable_node":
            sources, targets = nodes.keys(), error["element_ids"]
        else:
            continue
        # ponytail: 후보×노드 목록. 큰 순서도는 문맥 영역별 분할이 필요합니다.
        for source in sources:
            for target in targets:
                if source != target and source in nodes and target in nodes and nodes[source]["role"] != "end" and nodes[target]["role"] != "start" and (source, target) not in existing:
                    result.setdefault((source, target), []).append(error["error_id"])
    return result


def analyze_flowchart(document, model=None):
    metadata = document.get("metadata", {})
    if metadata.get("document_type") != "flowchart":
        return None
    rules = flowchart_rules(metadata)
    graph = build_graph(document)
    graph.update({"rules": rules, "rules_version": "1", "status": "analyzed" if graph["nodes"] else "insufficient_nodes"})
    if model is not None and graph["nodes"]:
        response = model.generate_json(
            "Classify supplied flowchart nodes using their text, boxes, graph and the attached image if present. "
            "Preserve CV role hints. Complete output_template, keeping every ID and array length unchanged. "
            "Replace the template strings with your role predictions. "
            "Include every node_id exactly once. Roles: start, end, process, decision, unknown. "
            "Use unknown when evidence is insufficient. Do not invent nodes.",
            {"node_ids": [n["element_id"] for n in graph["nodes"]], "graph": graph,
             "output_template": {"nodes": [{"element_id": n["element_id"], "role": "<start/end/process/decision/unknown>"} for n in graph["nodes"]]}},
        )
        roles = index_predictions(response, "nodes", "element_id", {n["element_id"] for n in graph["nodes"]})
        for node in graph["nodes"]:
            role = roles[node["element_id"]].get("role")
            if not isinstance(role, str) or role not in ROLES or (node["role_source"] == "cv_hint" and role != node["role"]):
                raise ValueError("모델의 순서도 역할이 올바르지 않거나 CV 역할 힌트와 충돌합니다.")
            node.update({"role": role, "role_source": "model"})
    candidates = find_candidates(graph, rules)
    proposals = []
    if model is not None and candidates:
        allowed_repairs = [{"source_id": source, "target_id": target, "error_ids": error_ids}
                           for (source, target), error_ids in sorted(repair_pairs(graph, candidates).items())]
        response = model.generate_json(
            "Review ONLY the supplied structural error candidates against the graph, document data and attached image if present. "
            "These are hypotheses, not ground truth. Cycles and unreviewed relations are not errors by themselves. "
            "Complete output_template, keeping every error_id and the errors array length unchanged. "
            "Replace template strings with your judgments and explanatory text. "
            "Optional proposed_relations entries have source_id, target_id and reason string fields. "
            "Return each error_id exactly once; confirmation is true, false or null. Use null when uncertain. "
            "proposed_relations may be empty. Propose only repairs to confirmed missing_connection, decision_branch_missing or unreachable_node errors. "
            "Each proposed source_id and target_id MUST be an allowed pair in allowed_repairs. "
            "Do not reverse the source and target. Return [] if no supported repair is available. "
            "Use existing node IDs only. A proposed repair is not an observed arrow. Never invent text or nodes.",
            {"graph": graph, "candidates": candidates, "error_ids": [e["error_id"] for e in candidates], "allowed_repairs": allowed_repairs,
             "output_template": {"errors": [{"error_id": e["error_id"], "is_confirmed": "<boolean or null>",
                                              "reason": "<evidence>", "suggestion": "<suggestion>"} for e in candidates],
                                 "proposed_relations": []}},
        )
        reviews = index_predictions(response, "errors", "error_id", {e["error_id"] for e in candidates})
        for candidate in candidates:
            review = reviews[candidate["error_id"]]
            confirmed = review.get("is_confirmed")
            if "is_confirmed" not in review or (confirmed is not None and type(confirmed) is not bool):
                raise ValueError("오류 확인값은 boolean 또는 null이어야 합니다.")
            if any(not isinstance(review.get(key), str) or not review[key].strip() for key in ("reason", "suggestion")):
                raise ValueError("오류 검토에는 근거와 수정 제안 문자열이 필요합니다.")
            candidate.update({key: review[key] for key in ("is_confirmed", "reason", "suggestion")})
            candidate["review_source"] = "model"
        proposed = response.get("proposed_relations")
        if not isinstance(proposed, list):
            raise ValueError("proposed_relations는 배열이어야 합니다.")
        allowed = repair_pairs(graph, candidates, confirmed_only=True)
        seen = set()
        for item in proposed:
            if not isinstance(item, dict):
                raise ValueError("제안 관계는 객체여야 합니다.")
            source, target = item.get("source_id"), item.get("target_id")
            if not isinstance(source, str) or not isinstance(target, str) or (source, target) not in allowed or (source, target) in seen:
                raise ValueError("제안 관계의 참조·오류 근거·중복을 확인하세요.")
            if not isinstance(item.get("reason"), str) or not item["reason"].strip():
                raise ValueError("제안 관계에는 근거 문자열이 필요합니다.")
            seen.add((source, target))
            proposals.append({"proposal_id": f"flow_proposal_{len(proposals) + 1:03d}", "source_id": source,
                              "target_id": target, "relation_type": "logical_flow", "reason": item["reason"],
                              "related_error_ids": allowed[(source, target)],
                              "status": "needs_user_confirmation", "purpose": "suggested_repair"})
    document.update({"flowchart": graph, "errors": candidates, "proposed_relations": proposals})
    return "model_reviewed" if model is not None else "rules_only"
