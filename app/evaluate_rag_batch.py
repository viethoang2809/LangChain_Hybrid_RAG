"""
Hybrid RAG BATCH: chạy nhiều câu hỏi giống hệt CLI và lưu kết quả ra CSV.
"""
import os, sys, csv, time, asyncio, traceback
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# === Thêm đường dẫn để import ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# === Import nội bộ ===
from app.retrievers.hybrid_retriever import HybridRetrieverParallel
from app.retrievers.vector_tools import VectorClient
from app.utils.hybrid_helpers import (
    load_answer_rule,
    build_id_map_from_graph_records,
    select_topN_by_priority,
    build_synthesis_input,
    llm_summarize_answer,
)

# === Cấu hình ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=OPENAI_API_KEY)

INPUT_PATH = "data/Question.csv"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
)


# =======================================================
# 🚀 CHẠY 1 CÂU HỎI GIỐNG HỆT CLI
# =======================================================
async def run_query_once(user_query: str, top_k: int = 10, limit: int = 3, show_debug: bool = False):
    synth_rule = load_answer_rule()
    hybrid = HybridRetrieverParallel()
    vclient = hybrid.vector

    print(f"\n❓ {user_query}\n")
    print("⏳ Đang truy vấn dữ liệu song song từ Neo4j và FAISS...\n")

    total_start = time.time()
    hybrid_start = time.time()
    hybrid_result = await hybrid.search(user_query=user_query, top_k=top_k)
    hybrid_time = int((time.time() - hybrid_start) * 1000)

    graph_records = hybrid_result["graph_records"]
    graph_ids = hybrid_result["graph_ids"]
    vector_passages = hybrid_result["vector_passages"]

    # Xây map Graph
    graph_id_map = build_id_map_from_graph_records(graph_records)

    # Chọn topN passage theo ID
    fusion_start = time.time()
    chosen_passages = select_topN_by_priority(
        graph_ids, vector_passages, vclient, graph_id_map, fill_limit=limit
    )
    fusion_time = int((time.time() - fusion_start) * 1000)

    # Debug nếu cần
    if show_debug:
        print("───────────────────────────────")
        print("🔍 DEBUG THÔNG TIN TRUY VẤN")
        print("───────────────────────────────")
        print(f"📊 Graph IDs ({len(graph_ids)}): {graph_ids[:10]}")
        print(f"📚 Vector IDs ({len(vector_passages)}): {[p.id for p in vector_passages[:10]]}")
        print(f"✅ Chosen IDs ({len(chosen_passages)}): {[p.id for p in chosen_passages]}")
        print("───────────────────────────────\n")

    # Chuẩn bị dữ liệu tổng hợp
    synthesis_payload = build_synthesis_input(chosen_passages, graph_id_map)

    # Gọi LLM tổng hợp 
    print("🧠 Đang tổng hợp câu trả lời bằng GPT...\n")
    llm_start = time.time()
    answer = llm_summarize_answer(client, user_query, synth_rule, synthesis_payload, OPENAI_MODEL)
    llm_time = int((time.time() - llm_start) * 1000)
    total_time = int((time.time() - total_start) * 1000)

    # === In ra giống CLI ===
    print("\n✨ CÂU TRẢ LỜI:\n───────────────────────────────")
    print(answer)
    print("───────────────────────────────")

    # In ra các snippet gợi nhớ
    print("\n📋 DỮ LIỆU HỢP NHẤT:")
    for p in chosen_passages:
        pid = str(p.id).strip() if p.id else "N/A"
        snippet = (p.text or "").strip().replace("\n", " ")
        if len(snippet) > 120:
            snippet = snippet[:120] + "..."
        print(f"• ID {pid}: {snippet}...")

    print("\n───────────────────────────────")
    print("⏱ THỜI GIAN XỬ LÝ")
    print("───────────────────────────────")
    print(f"🔸 Graph + Vector: {hybrid_time} ms")
    print(f"🔸 Fusion chọn topN: {fusion_time} ms")
    print(f"🔸 LLM tổng hợp: {llm_time} ms")
    print(f"⚡ Tổng thời gian: {total_time} ms")
    print("───────────────────────────────\n")

    return answer.strip()


# =======================================================
# 🔁 CHẠY NHIỀU CÂU HỎI TRONG FILE
# =======================================================
async def main():
    print("🏠 Hybrid RAG – Batch Mode (y hệt CLI)")
    print("=========================================================")
    print(f"📂 Đọc file câu hỏi: {INPUT_PATH}")

    # Đọc danh sách câu hỏi
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        questions = [row["question"] for row in reader if row.get("question")]

    print(f"✅ Tổng số câu hỏi: {len(questions)}\n")

    results = []

    for idx, query in enumerate(questions, 1):
        print(f"🔹 [{idx}/{len(questions)}] {query}")
        try:
            answer = await run_query_once(query, top_k=10, limit=3)
            results.append({"id": idx, "question": query, "answer": answer})
        except Exception as e:
            print(f"❌ Lỗi ở câu {idx}: {e}")
            print(traceback.format_exc())
            results.append({"id": idx, "question": query, "answer": f"ERROR: {e}"})

    # Ghi ra CSV
    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "question", "answer"])
        writer.writeheader()
        writer.writerows(results)

    print("\n✅ Hoàn tất! Kết quả lưu tại:")
    print(f"👉 {os.path.abspath(OUTPUT_PATH)}")


if __name__ == "__main__":
    asyncio.run(main())
