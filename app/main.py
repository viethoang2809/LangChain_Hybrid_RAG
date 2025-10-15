# app/main.py
# Streamlit UI cho Hybrid RAG: Neo4j (NL2Cypher) + FAISS, hợp nhất theo ID và tổng hợp câu trả lời
import os
import json
import traceback
from typing import List, Dict, Any

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# Local modules
from app.retrievers.hybrid_retriever import HybridRetriever
from app.retrievers.vector_tools import VectorClient, Passage

# Cấu hình
load_dotenv()

def get_var(key, default=None, section="general"):
    try:
        return st.secrets[section].get(key, default)
    except Exception:
        return os.getenv(key, default)
OPENAI_MODEL = get_var("OPENAI_MODEL", "gpt-4o-mini")
ANSWER_RULE_PATH = get_var("ANSWER_RULE_PATH", "app/prompts/answer_synthesis.txt")
OPENAI_API_KEY = get_var("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY)


# ===============================
# 🔧 Helper functions
# ===============================
def load_answer_rule(path: str = ANSWER_RULE_PATH) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy rule tổng hợp câu trả lời: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def build_id_map_from_graph_records(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Tạo map id -> record (thuộc tính từ Neo4j)."""
    id_map = {}
    for r in records or []:
        rid = str(r.get("id") or "").strip()
        if rid:
            id_map[rid] = r
    return id_map


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


def select_top3_by_priority(
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

    # 1️⃣ Overlap giữa Graph & Vector
    for gid in graph_ids:
        if gid in vector_by_id and gid not in used_ids:
            picked.append(vector_by_id[gid])
            used_ids.add(gid)
            if len(picked) >= fill_limit:
                return picked

    # 2️⃣ Graph có ID nhưng Vector chưa có → cố fetch theo ID
    missing_from_vector = [gid for gid in graph_ids if gid not in used_ids and gid not in vector_by_id]
    if missing_from_vector:
        fetched = vector_fetch_by_ids(vclient, missing_from_vector, limit=(fill_limit - len(picked)))
        for p in fetched:
            if p.id and p.id not in used_ids:
                picked.append(p)
                used_ids.add(p.id)
                if len(picked) >= fill_limit:
                    return picked

    # 3️⃣ Bổ sung từ vector_passages còn lại
    for p in vector_passages:
        pid = str(p.id).strip() if p.id else None
        if pid and pid not in used_ids:
            picked.append(p)
            used_ids.add(pid)
        if len(picked) >= fill_limit:
            break

    return picked[:fill_limit]


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


def llm_summarize_answer(client: OpenAI, user_query: str, synthesis_rule: str, synthesis_payload: str, model: str) -> str:
    """Tổng hợp đầu ra cuối cùng bằng LLM."""
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


# ===============================
# 🖥️ Streamlit UI
# ===============================
def main():
    st.set_page_config(page_title="Hybrid RAG - Bất động sản Hà Nội", page_icon="🏠", layout="wide")
    st.title("🏠 Hybrid RAG cho Bất động sản Hà Nội")
    st.caption("Kết hợp Neo4j (Graph) + FAISS (Vector) · Truy vấn 1 lần Graph duy nhất · Giới hạn 3 căn / câu trả lời")

    with st.sidebar:
        st.header("⚙️ Cài đặt")
        model = st.text_input("OPENAI_MODEL", value=OPENAI_MODEL)
        top_k = st.slider("Số kết quả Vector (k)", min_value=5, max_value=20, value=10)
        limit_ids = st.slider("Giới hạn ID trả lời", min_value=1, max_value=5, value=3)
        show_debug = st.checkbox("🧩 Hiển thị debug (IDs & mô tả)", value=True)

    # Nhập câu hỏi
    user_query = st.text_input("💬 Nhập câu hỏi của bạn:", placeholder="Ví dụ: Tìm nhà 5 tầng sổ đỏ chính chủ tại Thanh Xuân")
    run = st.button("🔎 Tìm kiếm")

    if run and user_query.strip():
        try:
            client = OpenAI()
            synth_rule = load_answer_rule()
            hybrid = HybridRetriever()
            vclient = hybrid.vector

            # 1️⃣ Hybrid Search (Graph + Vector, chỉ query Graph 1 lần)
            st.info("⏳ Đang truy vấn dữ liệu từ Neo4j và FAISS...")
            hybrid_result = hybrid.search(user_query=user_query, top_k=top_k)
            graph_records = hybrid_result["graph_records"]
            graph_ids = hybrid_result["graph_ids"]
            vector_passages = hybrid_result["vector_passages"]

            # 2️⃣ Kết hợp dữ liệu
            graph_id_map = build_id_map_from_graph_records(graph_records)
            chosen_passages = select_top3_by_priority(
                graph_ids, vector_passages, vclient, graph_id_map, fill_limit=limit_ids
            )

            # Debug
            if show_debug:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📊 IDs từ Graph")
                    st.write(graph_ids[:20])
                with col2:
                    st.subheader("📚 IDs trong Vector (từ k kết quả)")
                    st.write([p.id for p in vector_passages if p.id][:20])

                st.subheader("✅ ID được chọn (ưu tiên trùng, tối đa N)")
                st.write([p.id for p in chosen_passages])

                st.subheader("📝 Snippet mô tả (Vector)")
                for p in chosen_passages:
                    st.markdown(f"- **ID {p.id or 'N/A'}** · _{(p.text or '')[:200]}{'...' if p.text and len(p.text)>200 else ''}_")

            # 3️⃣ Chuẩn bị dữ liệu cho LLM
            synthesis_payload = build_synthesis_input(chosen_passages, graph_id_map)

            # 4️⃣ Gọi LLM để tổng hợp câu trả lời
            st.write("🧠 Đang tổng hợp câu trả lời...")
            answer = llm_summarize_answer(client, user_query, synth_rule, synthesis_payload, model)

            # Hiển thị kết quả
            st.markdown("---")
            st.subheader("✨ Câu trả lời")
            st.write(answer)

            # 5️⃣ Bảng dữ liệu chi tiết
            with st.expander("📋 Xem dữ liệu đã hợp nhất (debug)"):
                merged_rows = []
                for p in chosen_passages:
                    pid = str(p.id).strip() if p.id else None
                    row = {"id": pid, "text_len": len(p.text or "")}
                    row.update(graph_id_map.get(pid, {}))
                    merged_rows.append(row)
                try:
                    import pandas as pd
                    st.dataframe(pd.DataFrame(merged_rows))
                except Exception:
                    st.json(merged_rows)

        except Exception as e:
            st.error("❌ Lỗi khi xử lý truy vấn.")
            st.exception(e)
            st.text(traceback.format_exc())

    st.markdown("---")
    st.caption("© Hybrid RAG • Neo4j + FAISS • LangChain-style pipeline")


if __name__ == "__main__":
    main()
