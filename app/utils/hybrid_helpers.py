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
