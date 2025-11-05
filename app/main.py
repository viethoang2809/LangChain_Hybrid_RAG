# app/main.py
# Hybrid RAG + Chatbot tư vấn thông minh
# Cache pipeline, lưu text gốc, giữ nguyên giao diện truy vấn thường

import os
import json
import traceback
import asyncio
import time
from datetime import datetime
import sys
from typing import List

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import threading

# ==== Import nội bộ ====
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.retrievers.hybrid_retriever import HybridRetrieverParallel
from app.retrievers.vector_tools import VectorClient, Passage
from app.utils.hybrid_helpers import (
    load_answer_rule,
    build_id_map_from_graph_records,
    select_topN_by_priority,
    build_synthesis_input,
    llm_summarize_answer,
)

from app.utils.vietmap_utils import enrich_last_chat_record

# Cấu hình hệ thống
load_dotenv()

def get_var(key, default=None, section="general"):
    try:
        return st.secrets[section].get(key, default)
    except Exception:
        return os.getenv(key, default)

OPENAI_MODEL_NORMAL = get_var("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_MODEL_ADVANCED = get_var("OPENAI_MODEL", "gpt-4o")
OPENAI_API_KEY = get_var("OPENAI_API_KEY")
ANSWER_RULE_PATH = get_var("ANSWER_RULE_PATH", "app/prompts/answer_synthesis.txt")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY



# Lịch sử chat
HISTORY_PATH = "data/chat_history.jsonl"
os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)

def save_chat_record(query: str, answer: str, passages: List[Passage]):
    """
    Lưu lịch sử hội thoại gồm:
    - Câu hỏi (query)
    - Câu trả lời (answer)
    - Văn bản gốc (raw_text) trích từ các Passage của RAG
    - Danh sách ID và nội dung từng Passage (để debug hoặc tư vấn lại)

    Dữ liệu được lưu vào: data/chat_history.jsonl
    Mỗi dòng là 1 JSON record (append mode)
    """

    # Gộp toàn bộ text gốc từ các Passage đã chọn
    raw_text = "\n\n".join(
        [
            f"(ID {p.id}) {p.text.strip()}"
            for p in passages
            if getattr(p, "text", None)
        ]
    )

    # Ghi record theo thứ tự hợp lý: raw_text đặt ngay sau answer
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "query": query,
        "answer": answer,
        "raw_text": raw_text,  # 👈 để ngay sau answer cho dễ đọc & dễ truy xuất
        "property_ids": [
            p.id for p in passages if getattr(p, "id", None)
        ],
        "passages": [
            {"id": p.id, "text": p.text}
            for p in passages
            if getattr(p, "text", None)
        ],
    }

    # Ghi append từng dòng JSON
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    with open(HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def load_chat_history(limit: int = 2):
    if not os.path.exists(HISTORY_PATH):
        return []
    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()[-limit:]
    return [json.loads(l) for l in lines]



# Giao diện chính
def main():
    st.set_page_config(page_title="Hybrid RAG - Bất động sản Hà Nội", page_icon="🏠", layout="wide")
    st.title("🏠 Hybrid RAG cho Bất động sản Hà Nội")
    st.caption("Kết hợp Neo4j (Graph) + FAISS (Vector) · Tổng hợp bằng GPT")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Cài đặt")
        model = st.text_input("OPENAI_MODEL", value=OPENAI_MODEL_NORMAL)
        top_k = st.slider("Số kết quả Vector (k)", 5, 20, 10)
        limit_ids = st.slider("Giới hạn ID trả lời", 1, 5, 3)
        show_debug = st.checkbox("🧩 Hiển thị debug", value=True)
        chatbot_mode = st.toggle("🤖 Chatbot Mode", value=False)

        st.markdown("---")
        st.subheader("💾 Lịch sử gần đây")
        history = load_chat_history(limit=10)
        if history:
            for h in reversed(history[-5:]):
                with st.expander(f"📅 {h['timestamp']} — {h['query'][:40]}..."):
                    st.write(h["answer"])
                    st.caption(f"IDs: {', '.join(h['property_ids'])}")
        else:
            st.info("Chưa có lịch sử nào.")



    # CHATBOT MODE (Cache + UI 2 bên)
    if chatbot_mode:
        st.markdown("""
            <h3>💬 Chatbot Tư vấn Bất động sản thông minh</h3>
        """, unsafe_allow_html=True)

        # Cache pipeline: chỉ load 1 lần
        if "hybrid_pipeline" not in st.session_state:
            with st.spinner("⏳ Đang tải pipeline Hybrid RAG (Neo4j + FAISS)..."):
                st.session_state.hybrid_pipeline = HybridRetrieverParallel()
                st.success("✅ Đã tải xong pipeline RAG!")
        hybrid = st.session_state.hybrid_pipeline
        vclient = hybrid.vector

        if "chat_ui" not in st.session_state:
            st.session_state.chat_ui = []

        # Hiển thị hội thoại cũ
        for msg in st.session_state.chat_ui:
            if msg["role"] == "user":
                st.markdown(
                    f"""
                    <div style='text-align:right;'>
                        <div style='background-color:#f0f2f6; display:inline-block;
                                    padding:10px 14px; border-radius:12px; margin:4px 0;
                                    max-width:70%; color:#333;'>
                            <b>Khách hàng:</b> {msg['content']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div style='text-align:left;'>
                        <div style='background-color:#e8f5e9; display:inline-block;
                                    padding:10px 14px; border-radius:12px; margin:4px 0;
                                    max-width:70%; color:#111;'>
                            <b>Tư vấn viên:</b> {msg['content']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        user_input = st.chat_input("Nhập câu hỏi (VD: Có căn nào 4 tầng ở Cầu Giấy không?)")

        if user_input:
            # hiển thị tin nhắn khách hàng
            st.session_state.chat_ui.append({"role": "user", "content": user_input})
            st.markdown(
                f"""
                <div style='text-align:right;'>
                    <div style='background-color:#f0f2f6;
                                display:inline-block; padding:10px 14px;
                                border-radius:12px; margin:4px 0;
                                max-width:70%; color:#333;'>
                        <b>Khách hàng:</b> {user_input}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            loading_box = st.empty()
            loading_box.markdown(
                """
                <div style='text-align:left;'>
                    <div style='background-color:#fff3cd;
                                display:inline-block; padding:10px 14px;
                                border-radius:12px; margin:4px 0;
                                max-width:70%; color:#333;'>
                        <b>Tư vấn viên:</b> ⏳ Đang xử lý, vui lòng chờ một chút...
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            try:
                client = OpenAI(api_key=OPENAI_API_KEY)
                synth_rule = load_answer_rule()

                #Lấy ngữ cảnh hội thoại gần nhất (session)
                session_history = ""
                for msg in st.session_state.chat_ui[-3:]:
                    role = "Khách hàng" if msg["role"] == "user" else "Tư vấn viên"
                    session_history += f"{role}: {msg['content']}\n"

                # === PHÂN LOẠI Ý ĐỊNH (chuẩn, chi tiết)
                intent_prompt = f"""
                Bạn là **bộ phân loại ý định câu hỏi** cho chatbot tư vấn bất động sản tại Hà Nội.

                Nhiệm vụ của bạn:
                Dựa trên **ngữ cảnh hội thoại gần nhất** và **câu hỏi hiện tại**, hãy xác định **ý định chính** của khách hàng.

                ---

                **Ngữ cảnh hội thoại gần đây:**
                {session_history}

                Hãy dựa vào câu hỏi và ngữ cảnh để phân loại thành đúng **1 trong 4 loại** sau:
                1. **SEARCH** — người dùng muốn tìm thêm căn nhà mới (chưa từng được đề cập trước đó).  
                Ví dụ:
                - "Tìm nhà 5 tầng ở Cầu Giấy"
                - "Có căn nào dưới 5 tỷ ở Hà Đông không?"
                - "Cho tôi thêm vài căn khu Thanh Xuân"

                2. **LISTING** — người dùng hỏi **các tiện ích hoặc địa điểm cụ thể quanh 1 căn đã có**.  
                Ví dụ:
                - "Gần căn 1 có bệnh viện nào không?"
                - "Xung quanh căn ở Kim Giang có công viên gì?"
                - "Căn thứ 2 có gần trường học không?"
                - "Liệt kê giúp tôi các siêu thị quanh căn Giáp Nhất"

                3. **COMPARE** — người dùng muốn **so sánh giữa 2 hoặc nhiều căn đã tư vấn**.  
                Ví dụ:
                - "So sánh căn 1 và căn 2 xem căn nào tiện hơn"
                - "Căn nào gần trung tâm hơn?"
                - "Nếu để đầu tư thì nên chọn căn nào trong 3 căn trên?"


                4. **ANALYZE** — người dùng muốn **đánh giá tổng quan, tư vấn sâu hoặc phân tích tiềm năng**.  
                Ví dụ:
                - "Khu Thanh Xuân có tiềm năng tăng giá không?"
                - "Căn ở Kim Giang có đáng mua không?"
                - "Căn số 1 có đáng để mua không ?"
                - "Khu vực này có đáng sống không?"
                - "Nếu ở gia đình 4 người thì căn nào hợp lý hơn?"

                ---

                **QUY TẮC PHÂN LOẠI:**
                - Nếu câu hỏi nói về *tìm căn mới* → luôn là **SEARCH**.
                - Nếu có từ khóa "gần", "xung quanh", "có gì", "liệt kê", "ở quanh" → **LISTING**.
                - Nếu có từ khóa "so sánh", "căn 1", "căn 2", "căn thứ", "nên chọn", "căn nào" → **COMPARE**.
                - Nếu câu hỏi thiên về "đánh giá", "phù hợp", "tiềm năng", "đáng mua", "đáng sống" → **ANALYZE**.
                - Nếu câu hỏi có nhiều đặc điểm, chọn loại có **độ chuyên biệt cao hơn** theo thứ tự ưu tiên:
                `COMPARE > LISTING > ANALYZE > SEARCH`.

                ---

                **Đầu ra:**
                - Trả về đúng **một từ duy nhất** trong bốn loại:  
                `"SEARCH"` hoặc `"LISTING"` hoặc `"COMPARE"` hoặc `"ANALYZE"`.
                - Không giải thích, không dấu câu, không xuống dòng.

                ---

                CÂU HỎI:
                "{user_input}"
                """


                intent_resp = client.chat.completions.create(
                    model=OPENAI_MODEL_NORMAL,
                    messages=[{"role": "system", "content": intent_prompt}],
                    temperature=0,
                )
                intent = intent_resp.choices[0].message.content.strip().upper()

                # === SEARCH ===
                if "SEARCH" in intent:
                    loading_box.markdown(
                        """
                        <div style='text-align:left;'>
                            <div style='background-color:#fff3cd;
                                        display:inline-block; padding:10px 14px;
                                        border-radius:12px; margin:4px 0;
                                        max-width:70%; color:#333;'>
                                <b>Tư vấn viên:</b> 💭 Bạn hãy đợi tôi chút để tôi tìm bất động sản phù hợp cho bạn nha...
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    hybrid_result = asyncio.run(hybrid.search(user_query=user_input, top_k=top_k))
                    graph_records = hybrid_result["graph_records"]
                    graph_ids = hybrid_result["graph_ids"]
                    vector_passages = hybrid_result["vector_passages"]

                    graph_id_map = build_id_map_from_graph_records(graph_records)
                    chosen_passages = select_topN_by_priority(
                        graph_ids, vector_passages, vclient, graph_id_map, fill_limit=limit_ids
                    )
                    synthesis_payload = build_synthesis_input(chosen_passages, graph_id_map)
                    answer = llm_summarize_answer(client, user_input, synth_rule, synthesis_payload, OPENAI_MODEL_NORMAL)
                    save_chat_record(user_input, answer, chosen_passages)
                    threading.Thread(target=enrich_last_chat_record, daemon=True).start()

                else:
                    # === CHAT / LISTING / COMPARE / ANALYZE ===
                    history_data = load_chat_history(limit=10)

                    # 1 Lấy ngữ cảnh hội thoại gần nhất (session)
                    session_history = ""
                    for msg in st.session_state.chat_ui[-6:]:
                        role = "Khách hàng" if msg["role"] == "user" else "Tư vấn viên"
                        session_history += f"{role}: {msg['content']}\n"

                    # 2 Lấy dữ liệu thật từ chat_history
                    long_context = ""
                    vietmap_context = ""

                    for h in history_data:
                        q = h.get("query", "")
                        a = h.get("answer", "")
                        raw = h.get("raw_text", "")
                        long_context += f"💬 Q: {q}\n🤖 A: {a}\n📄 Dữ liệu gốc:\n{raw}\n\n"

                        # Thêm VietMap context theo nhóm tiện ích
                        props = h.get("properties", [])
                        if props:
                            for p in props:
                                address = p.get("address", "")
                                label = p.get("label", "Căn")
                                groups = p.get("nearby_groups", {}) or {}
                                if not groups:
                                    vietmap_context += f"🏠 {label} tại {address}, chưa có dữ liệu tiện ích.\n\n"
                                    continue
                                vietmap_context += f"🏠 {label} tại {address} có các tiện ích xung quanh:\n"
                                group_names = {
                                    "hospital": "🏥 Y tế",
                                    "primary_school": "🏫 Trường tiểu học",
                                    "secondary_school": "🏫 Trường trung học",
                                    "university": "🎓 Đại học",
                                    "supermarket": "🛒 Siêu thị",
                                    "market": "🛍️ Chợ",
                                    "park": "🌳 Công viên",
                                    "restaurant": "🍽️ Nhà hàng",
                                    "cafe": "☕ Cà phê",
                                }
                                for group, items in groups.items():
                                    if not items or isinstance(items, dict) and "error" in items:
                                        continue
                                    viet_group = group_names.get(group, group)
                                    vietmap_context += f"  {viet_group}:\n"
                                    for n in items[:2]:  # chỉ lấy 2 địa điểm gần nhất mỗi nhóm
                                        name = n.get("name", "")
                                        dist = n.get("distance_km", "")
                                        if name:
                                            vietmap_context += f"    • {name} ({dist} km)\n"
                                vietmap_context += "\n"

                    # 3 Chọn model & prompt theo intent
                    if intent == "LISTING":
                        model = OPENAI_MODEL_NORMAL
                        tone = (
                                "liệt kê cụ thể và chính xác các tiện ích xung quanh từng căn nhà dựa trên dữ liệu VietMap. "
                                "Tập trung vào việc trình bày rõ ràng từng nhóm tiện ích (trường học, bệnh viện, siêu thị, nhà hàng, công viên...) "
                                "Mỗi tiện ích cần có tên và khoảng cách thực tế, giúp khách hàng dễ hình dung vị trí. "
                                "Viết ngắn gọn, dễ đọc, sử dụng biểu tượng (emoji) nếu phù hợp để phân nhóm. "
                                "Không phân tích hay đánh giá, chỉ mô tả khách quan, chính xác và có cấu trúc rõ ràng."
                            )
                    elif intent == "COMPARE":
                        model = OPENAI_MODEL_ADVANCED
                        tone = (
                                "So sánh chi tiết giữa các căn nhà được nhắc đến, dựa trên vị trí, tiện ích xung quanh và giá trị sống. "
                                "Chỉ ra điểm mạnh – yếu của từng căn: ví dụ căn nào gần trường học, căn nào yên tĩnh hơn, "
                                "Căn nào thuận tiện hơn cho gia đình có con nhỏ hoặc người đi làm trung tâm. "
                                "Giữ giọng văn khách quan, thân thiện, giống như một chuyên gia tư vấn giúp khách hàng chọn căn phù hợp nhất. "
                                "Kết thúc nên có gợi ý nhẹ về căn đáng cân nhắc hơn (nhưng không ép lựa chọn)."
                            )
                    elif intent == "ANALYZE":
                        model = OPENAI_MODEL_ADVANCED
                        tone = (
                                "Đánh giá tổng quan và chuyên sâu về các căn nhà hoặc khu vực được nhắc đến. "
                                "Tập trung vào việc phân tích giá trị sống, tiềm năng tăng giá, quy hoạch khu vực, "
                                "Và sự hài hòa giữa vị trí, tiện ích, môi trường sống và giao thông. Hãy liệt kê ra rồi phân tích từng yếu tố một cách rõ ràng. "
                                "Diễn đạt như một chuyên gia bất động sản cao cấp tại Hà Nội: giọng văn tự tin, chuyên nghiệp nhưng gần gũi. "
                                "Nêu rõ vì sao khu vực hoặc căn này đáng chú ý, yếu tố nào giúp gia tăng giá trị hoặc phù hợp với từng nhóm khách hàng."
                            )
                    else:
                        model = OPENAI_MODEL_NORMAL
                        tone = (
                                "tư vấn nối tiếp thân thiện, tự nhiên, mang tính trò chuyện cá nhân. "
                                "Dựa vào lịch sử hội thoại và các dữ liệu thật trong hệ thống, "
                                "giải thích hoặc làm rõ thông tin cho khách hàng, không cần phân tích dài. "
                                "Giữ câu văn ngắn gọn, dễ hiểu, tránh lặp lại mô tả cũ. "
                                "Nếu khách hỏi cảm nhận hoặc xin lời khuyên, hãy phản hồi trung lập, có lý lẽ và niềm tin chuyên môn."
                            )

                    advisor_prompt = f"""
                    Bạn là **chuyên gia tư vấn bất động sản cao cấp tại Hà Nội**, am hiểu sâu về **vị trí, tiện ích xung quanh và tiềm năng phát triển khu vực**.

                    **Nhiệm vụ:** {tone}

                    ---

                    **Lịch sử hội thoại gần đây:**
                    {session_history}

                    **Dữ liệu thật về các căn nhà:**
                    {long_context}

                    **Tiện ích & vị trí theo VietMap (rất quan trọng):**
                    {vietmap_context}

                    **Câu hỏi của khách hàng:**
                    "{user_input}"

                    ---

                    **Yêu cầu phản hồi:**
                    - Viết bằng giọng tự nhiên, thân thiện, phong cách tư vấn chuyên nghiệp.
                    - Dựa vào dữ liệu thật, không bịa thông tin.
                    - Khi có nhiều căn, so sánh nhẹ nhàng, khách quan.
                    - Nêu rõ nhóm tiện ích và ý nghĩa của chúng (VD: gần trường học → tiện cho con nhỏ).
                    - Nếu thiếu dữ liệu ở nhóm nào, hãy nói rõ: “chưa có dữ liệu về ... quanh căn này”.
                    """




                    resp = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "system", "content": advisor_prompt}],
                        temperature=0.7,
                    )
                    answer = resp.choices[0].message.content.strip()

                # === HIỂN THỊ TRẢ LỜI
                loading_box.empty()
                st.session_state.chat_ui.append({"role": "assistant", "content": answer})
                st.markdown(
                    f"""
                    <div style='text-align:left;'>
                        <div style='background-color:#e8f5e9;
                                    display:inline-block; padding:10px 14px;
                                    border-radius:12px; margin:4px 0;
                                    max-width:70%; color:#111;'>
                            <b>Tư vấn viên:</b> {answer}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            except Exception as e:
                loading_box.empty()
                st.error(f"Lỗi: {e}")
                st.text(traceback.format_exc())

        st.stop()


 
    # GIAO DIỆN TRUY VẤN THƯỜNG
    user_query = st.text_input(
        "💬 Nhập câu hỏi của bạn:",
        placeholder="Ví dụ: Tìm nhà 5 tầng sổ đỏ chính chủ tại Thanh Xuân"
    )
    run = st.button("🔎 Tìm kiếm")

    if run and user_query.strip():
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            synth_rule = load_answer_rule()
            hybrid = HybridRetrieverParallel()
            vclient = hybrid.vector

            st.info("⏳ Đang truy vấn dữ liệu song song từ Neo4j và FAISS...")
            start = time.time()
            hybrid_result = asyncio.run(hybrid.search(user_query=user_query, top_k=top_k))
            took = int((time.time() - start) * 1000)

            graph_records = hybrid_result["graph_records"]
            graph_ids = hybrid_result["graph_ids"]
            vector_passages = hybrid_result["vector_passages"]

            # Hiển thị Cypher Query nếu có
            if "cypher_query" in hybrid_result and hybrid_result["cypher_query"]:
                st.markdown("---")
                st.subheader("📜 Truy vấn Cypher được sinh ra")
                st.code(hybrid_result["cypher_query"], language="cypher")

            # Kết hợp dữ liệu
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

            # Chuẩn bị dữ liệu cho LLM
            synthesis_payload = build_synthesis_input(chosen_passages, graph_id_map)

            # Gọi LLM để tổng hợp câu trả lời
            st.write("🧠 Đang tổng hợp câu trả lời...")
            answer = llm_summarize_answer(client, user_query, synth_rule, synthesis_payload, model)

            # Hiển thị kết quả
            st.markdown("---")
            st.subheader("✨ Câu trả lời")
            st.write(answer)

            # Lưu lại lịch sử + text gốc
            save_chat_record(user_query, answer, chosen_passages)
            threading.Thread(target=enrich_last_chat_record, daemon=True).start()

            # Bảng dữ liệu chi tiết
            with st.expander("📋 Xem dữ liệu đã hợp nhất (debug)"):
                merged_rows = []
                for p in chosen_passages:
                    pid = str(p.id).strip() if p.id else None
                    row = {"id": pid, "text_len": len(p.text or "")}
                    row.update(graph_id_map.get(pid, {}))
                    merged_rows.append(row)
                try:
                    st.dataframe(pd.DataFrame(merged_rows))
                except Exception:
                    st.json(merged_rows)

        except Exception as e:
            st.error("❌ Lỗi khi xử lý truy vấn.")
            st.exception(e)
            st.text(traceback.format_exc())

    st.markdown("---")
    st.caption("© Hybrid RAG • Neo4j + FAISS • Chatbot tư vấn thông minh (cache pipeline + lưu text gốc)")


if __name__ == "__main__":
    main()
