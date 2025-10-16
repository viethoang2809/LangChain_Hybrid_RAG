# app/main.py
# Streamlit UI cho Hybrid RAG: Neo4j (NL2Cypher) + FAISS, chạy song song và hiển thị debug chi tiết
import os
import json
import traceback
import asyncio
import time
from typing import List, Dict, Any

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# Local modules
from app.retrievers.hybrid_retriever import HybridRetrieverParallel
from app.retrievers.vector_tools import VectorClient, Passage
from app.utils.hybrid_helpers import (
    load_answer_rule,
    build_id_map_from_graph_records,
    select_topN_by_priority,
    build_synthesis_input,
    llm_summarize_answer,
)


# Cấu hình hệ thống
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



# Giao diện chính
def main():
    st.set_page_config(page_title="Hybrid RAG - Bất động sản Hà Nội", page_icon="🏠", layout="wide")
    st.title("🏠 Hybrid RAG cho Bất động sản Hà Nội (Parallel)")
    st.caption("Kết hợp Neo4j (Graph) + FAISS (Vector) · Chạy song song · Tổng hợp bằng GPT")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Cài đặt")
        model = st.text_input("OPENAI_MODEL", value=OPENAI_MODEL)
        top_k = st.slider("Số kết quả Vector (k)", min_value=5, max_value=20, value=10)
        limit_ids = st.slider("Giới hạn ID trả lời", min_value=1, max_value=5, value=3)
        show_debug = st.checkbox("🧩 Hiển thị debug (IDs & mô tả)", value=True)

    # Input
    user_query = st.text_input(
        "💬 Nhập câu hỏi của bạn:",
        placeholder="Ví dụ: Tìm nhà 5 tầng sổ đỏ chính chủ tại Thanh Xuân"
    )
    run = st.button("🔎 Tìm kiếm")


    # Xử lý khi người dùng nhấn tìm kiếm
    if run and user_query.strip():
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            synth_rule = load_answer_rule()
            hybrid = HybridRetrieverParallel()
            vclient = hybrid.vector

            # 1 Chạy truy vấn song song Graph + Vector
            st.info("⏳ Đang truy vấn dữ liệu song song từ Neo4j và FAISS...")
            start = time.time()
            hybrid_result = asyncio.run(hybrid.search(user_query=user_query, top_k=top_k))
            took = int((time.time() - start) * 1000)

            graph_records = hybrid_result["graph_records"]
            graph_ids = hybrid_result["graph_ids"]
            vector_passages = hybrid_result["vector_passages"]

            # 📜 Hiển thị Cypher Query nếu có
            if "cypher_query" in hybrid_result and hybrid_result["cypher_query"]:
                st.markdown("---")
                st.subheader("📜 Truy vấn Cypher được sinh ra")
                st.code(hybrid_result["cypher_query"], language="cypher")


            # 2 Kết hợp dữ liệu
            graph_id_map = build_id_map_from_graph_records(graph_records)
            chosen_passages = select_topN_by_priority(
                graph_ids, vector_passages, vclient, graph_id_map, fill_limit=limit_ids
            )


            # Debug
            if show_debug:
                st.markdown("---")
                st.subheader("🧩 DEBUG THÔNG TIN")

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
                    st.markdown(
                        f"- **ID {p.id or 'N/A'}** · _{(p.text or '')[:200]}{'...' if p.text and len(p.text)>200 else ''}_"
                    )

                st.info(f"⏱ Tổng thời gian truy vấn song song: **{took} ms**")

            # 3 Chuẩn bị dữ liệu cho LLM
            synthesis_payload = build_synthesis_input(chosen_passages, graph_id_map)

            # 4 Gọi LLM để tổng hợp câu trả lời
            st.write("🧠 Đang tổng hợp câu trả lời...")
            answer = llm_summarize_answer(client, user_query, synth_rule, synthesis_payload, model)

            # 5 Hiển thị kết quả
            st.markdown("---")
            st.subheader("✨ Câu trả lời")
            st.write(answer)

            # 6 Bảng dữ liệu chi tiết
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

    # Footer
    st.markdown("---")
    st.caption("© Hybrid RAG • Neo4j + FAISS • Chạy song song bằng asyncio")


if __name__ == "__main__":
    main()
