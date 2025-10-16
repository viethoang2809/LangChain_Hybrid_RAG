# app/utils/hybrid_helpers.py
import os
import json
from typing import List, Dict, Any
from openai import OpenAI

from app.retrievers.vector_tools import VectorClient, Passage



# Load rule tổng hợp câu trả lời answer_synthesis.txt
def load_answer_rule(path: str = "app/prompts/answer_synthesis.txt") -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Không tìm thấy rule tổng hợp câu trả lời: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()



# Tạo map ID -> Record (từ Neo4j)
def build_id_map_from_graph_records(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Tạo map id -> record (thuộc tính từ Neo4j)."""
    id_map = {}
    for r in records or []:
        rid = str(r.get("id") or "").strip()
        if rid:
            id_map[rid] = r
    return id_map



# Fetch lại bài viết theo ID từ VectorDB
def vector_fetch_by_ids(vclient: VectorClient, ids: List[str], limit: int = 3) -> List[Passage]:
    """Truy xuất lại các bài theo ID từ VectorDB."""
    vs = vclient._load_vs()
    results: List[Passage] = []
    wanted = set([str(x).strip() for x in ids if x])
    try:
        for _, doc in (vs.docstore._dict or {}).items():
            mid = (doc.metadata or {}).get("id")
            if mid and str(mid).strip() in wanted:
                results.append(
                    Passage(id=str(mid).strip(), text=doc.page_content or "", score=None, metadata=doc.metadata or {})
                )
                if len(results) >= limit:
                    break
    except Exception:
        pass
    return results



# Chọn top 3 bài ưu tiên giữa Graph và Vector
def select_topN_by_priority(
    graph_ids: List[str],
    vector_passages: List[Passage],
    vclient: VectorClient,
    graph_id_map: Dict[str, Dict[str, Any]],
    fill_limit: int = 3
) -> List[Passage]:
    """Chọn top 3 bài ưu tiên trùng ID giữa Graph và Vector."""
    picked: List[Passage] = []
    used_ids = set()
    graph_ids = [str(x).strip() for x in graph_ids if str(x).strip()]
    vector_by_id = {str(p.id).strip(): p for p in vector_passages if p.id}

    # 1 Overlap giữa Graph & Vector
    for gid in graph_ids:
        if gid in vector_by_id and gid not in used_ids:
            picked.append(vector_by_id[gid])
            used_ids.add(gid)
            if len(picked) >= fill_limit:
                return picked

    # 2 Graph có ID nhưng Vector chưa có → fetch thủ công
    missing_from_vector = [gid for gid in graph_ids if gid not in used_ids and gid not in vector_by_id]
    if missing_from_vector:
        fetched = vector_fetch_by_ids(vclient, missing_from_vector, limit=(fill_limit - len(picked)))
        for p in fetched:
            if p.id and p.id not in used_ids:
                picked.append(p)
                used_ids.add(p.id)
                if len(picked) >= fill_limit:
                    return picked

    # 3 Bổ sung từ vector_passages còn lại
    for p in vector_passages:
        pid = str(p.id).strip() if p.id else None
        if pid and pid not in used_ids:
            picked.append(p)
            used_ids.add(pid)
        if len(picked) >= fill_limit:
            break

    return picked[:fill_limit]



# Chuẩn bị input tổng hợp (graph + vector)
def build_synthesis_input(chosen_passages: List[Passage], graph_id_map: Dict[str, Dict[str, Any]]) -> str:
    """Tạo text có cấu trúc để gửi LLM tổng hợp."""
    blocks = []
    for p in chosen_passages:
        pid = str(p.id).strip() if p.id else None
        graph_info = graph_id_map.get(pid) if pid else None
        block = {
            "id": pid or "N/A",
            "graph": graph_info or {},
            "vector_text": (p.text or "").strip(),
        }
        blocks.append(block)

    pretty = []
    for b in blocks:
        pretty.append(
            f"ID: {b['id']}\nGRAPH: {json.dumps(b['graph'], ensure_ascii=False)}\nTEXT: {b['vector_text']}"
        )
    return "\n\n---\n\n".join(pretty)


# CONFIDENCE SCORING
def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def compute_confidence(
    semantic_score: float,
    hop_distance: int = 0,
    relation_weight: float = 1.0,
    alpha: float = 0.6,
    beta: float = 0.3,
    gamma: float = 0.1,
) -> float:
    """
    Điểm tin cậy tổng hợp:
      - semantic_score: cosine similarity (0..1) từ VectorDB (p.score)
      - hop_distance: 0 nếu id trùng Graph, >0 nếu không trùng (ước lượng)
      - relation_weight: tầm quan trọng record (0.5..1.0)
      - alpha/beta/gamma: trọng số (có thể tinh chỉnh)

    S_final = α*S_sem + β*(1/(1+hop)) + γ*w_rel
    """
    s_sem = _clip(float(semantic_score or 0.0), 0.0, 1.0)
    s_graph = 1.0 / (1.0 + max(0, int(hop_distance)))
    s_rel = _clip(float(relation_weight or 1.0), 0.0, 1.0)
    return round(alpha * s_sem + beta * s_graph + gamma * s_rel, 4)


def estimate_relation_weight(graph_info: dict) -> float:
    """
    Ước lượng 'độ quan trọng' của record theo các thuộc tính có mặt trong Graph.
    Không cần chính xác tuyệt đối; mục tiêu là ưu tiên record giàu thông tin hơn.
    Trả về trong khoảng [0.5, 1.0].
    """
    if not graph_info:
        return 0.5

    w = 0.5
    # Có pháp lý 'sổ đỏ/chính chủ' -> tăng mạnh
    legal = (graph_info.get("legal_status") or [])
    legal_text = " ".join([str(x) for x in legal]).lower()
    if "sổ đỏ" in legal_text or "chính chủ" in legal_text:
        w += 0.2

    # Có loại hình, địa chỉ -> tăng nhẹ
    if graph_info.get("property_type"):
        w += 0.1
    if graph_info.get("full_address"):
        w += 0.05

    # Có tiện ích nội bộ / gần tiện ích -> tăng nhẹ
    if graph_info.get("internal_amenities"):
        w += 0.05
    if graph_info.get("near_facilities"):
        w += 0.05

    return _clip(w, 0.5, 1.0)


def attach_confidence_to_passages(
    passages: List[Passage],
    graph_id_map: Dict[str, Dict[str, Any]],
    default_hop_no_match: int = 2,
    alpha: float = 0.6,
    beta: float = 0.3,
    gamma: float = 0.1,
) -> List[Passage]:
    """
    Gắn các trường chấm điểm vào metadata của từng Passage:
      - semantic (từ p.score nếu có)
      - hop (0 nếu id trùng Graph, else default_hop_no_match)
      - rel_weight (ước lượng từ record Graph tương ứng)
      - confidence (điểm tổng hợp)

    Trả về danh sách passages (giữ nguyên object) để có thể tiếp tục dùng.
    """
    for p in passages:
        pid = str(p.id).strip() if p.id else None
        semantic = float(p.score) if isinstance(p.score, (int, float)) else 0.0
        hop = 0 if (pid and pid in graph_id_map) else default_hop_no_match
        rel_w = estimate_relation_weight(graph_id_map.get(pid) or {})

        conf = compute_confidence(semantic, hop, rel_w, alpha=alpha, beta=beta, gamma=gamma)

        # Lưu vào metadata để CLI/Streamlit debug dễ
        meta = p.metadata or {}
        meta["semantic"] = round(semantic, 4)
        meta["hop"] = hop
        meta["relation_weight"] = round(rel_w, 3)
        meta["confidence"] = conf
        p.metadata = meta
    return passages


def rerank_by_confidence(passages: List[Passage]) -> List[Passage]:
    """
    Sắp xếp lại theo confidence (giảm dần).
    Nếu passage thiếu confidence, coi như 0.
    """
    def _get_conf(p):
        try:
            return float((p.metadata or {}).get("confidence") or 0.0)
        except Exception:
            return 0.0
    return sorted(passages, key=_get_conf, reverse=True)





# Tổng hợp đầu ra cuối cùng bằng LLM
def llm_summarize_answer(
    client: OpenAI,
    user_query: str,
    synthesis_rule: str,
    synthesis_payload: str,
    model: str,
) -> str:
    """Gọi LLM để tổng hợp câu trả lời."""
    prompt = f"""{synthesis_rule}

Dữ liệu đầu vào:
{synthesis_payload}

Câu hỏi người dùng:
{user_query}
"""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    return resp.choices[0].message.content.strip()
