"""
Hybrid RAG CLI: chạy song song Graph (Neo4j) + Vector (FAISS)
"""
import os, sys, json, traceback, argparse, asyncio, time
from dotenv import load_dotenv
from openai import OpenAI

# Thêm đường dẫn
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Module nội bộ
from app.retrievers.hybrid_retriever import HybridRetrieverParallel
from app.retrievers.vector_tools import VectorClient
from app.utils.hybrid_helpers import (
    load_answer_rule,
    build_id_map_from_graph_records,
    select_topN_by_priority,
    build_synthesis_input,
    llm_summarize_answer,
)

# Load config
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=OPENAI_API_KEY)



# CHẠY 1 TRUY VẤN HYBRID RAG SONG SONG
def run_query_once(user_query: str, top_k: int = 10, limit: int = 3, show_debug: bool = False):
    """Chạy một truy vấn Hybrid RAG duy nhất (song song Graph + Vector)."""
    print(f"\n❓ {user_query}\n")

    synth_rule = load_answer_rule()
    hybrid = HybridRetrieverParallel()
    vclient = hybrid.vector

    print("⏳ Đang truy vấn dữ liệu song song từ Neo4j và FAISS...\n")

    # --- Đo thời gian tổng ---
    total_start = time.time()
    hybrid_start = time.time()
    hybrid_result = asyncio.run(hybrid.search(user_query=user_query, top_k=top_k))
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

   
    # Debug chi tiết
    if show_debug:
        print("───────────────────────────────")
        print("🔍 DEBUG THÔNG TIN TRUY VẤN")
        print("───────────────────────────────")
        print(f"📊 Graph IDs ({len(graph_ids)}): {graph_ids[:20]}")
        print(f"📚 Vector IDs ({len(vector_passages)}): {[p.id for p in vector_passages[:20]]}")
        print(f"✅ Chosen IDs ({len(chosen_passages)}): {[p.id for p in chosen_passages]}")
        print()
        print("📝 Snippet mô tả:")
        for p in chosen_passages:
            snippet = (p.text or "").strip().replace("\n", " ")
            if len(snippet) > 160:
                snippet = snippet[:160] + "..."
            print(f"• ID {p.id or 'N/A'} → {snippet}")
        print("───────────────────────────────")
        print(f"⚙️  Graph + Vector time: {hybrid_time} ms")
        print(f"⚙️  Fusion (chọn topN): {fusion_time} ms\n")

    # Chuẩn bị dữ liệu tổng hợp
    synthesis_payload = build_synthesis_input(chosen_passages, graph_id_map)

    # Gọi LLM tổng hợp 
    print("🧠 Đang tổng hợp câu trả lời bằng GPT...\n")
    llm_start = time.time()
    answer = llm_summarize_answer(client, user_query, synth_rule, synthesis_payload, OPENAI_MODEL)
    llm_time = int((time.time() - llm_start) * 1000)

    total_time = int((time.time() - total_start) * 1000)


    # Hiển thị kết quả
    print("\n✨ CÂU TRẢ LỜI:\n───────────────────────────────")
    print(answer)
    print("───────────────────────────────")

    print("\n📋 DỮ LIỆU HỢP NHẤT:")
    for p in chosen_passages:
        pid = str(p.id).strip() if p.id else "N/A"
        snippet = (p.text or "").strip()[:120].replace("\n", " ")
        print(f"• ID {pid}: {snippet}...")


    # Thống kê thời gian
    print("\n───────────────────────────────")
    print("⏱ THỜI GIAN XỬ LÝ")
    print("───────────────────────────────")
    print(f"🔸 Graph + Vector song song: {hybrid_time} ms")
    print(f"🔸 Fusion chọn topN:         {fusion_time} ms")
    print(f"🔸 LLM tổng hợp:             {llm_time} ms")
    print(f"⚡ Tổng thời gian:           {total_time} ms")
    print("───────────────────────────────\n")



# CLI CHÍNH
def main():
    parser = argparse.ArgumentParser(description="Hybrid RAG CLI cho Bất động sản Hà Nội (song song)")
    parser.add_argument("--query", type=str, help="Câu hỏi người dùng (nếu không có, sẽ bật chế độ nhập tay)")
    parser.add_argument("--k", type=int, default=10, help="Số lượng top-k kết quả vector")
    parser.add_argument("--limit", type=int, default=3, help="Giới hạn số căn để tổng hợp")
    parser.add_argument("--show-debug", action="store_true", help="Hiển thị debug chi tiết")
    args = parser.parse_args()

    print("🏠 Hybrid RAG – Bất động sản Hà Nội (CLI mode, Parallel)")
    print("========================================================")

    try:
        if args.query:
            run_query_once(args.query, args.k, args.limit, args.show_debug)
            return

        print("🗨️  Nhập câu hỏi của bạn (gõ 'exit' để thoát):\n")
        while True:
            user_query = input("❓> ").strip()
            if not user_query:
                continue
            if user_query.lower() in ["exit", "quit", "q"]:
                print("👋 Tạm biệt!")
                break
            run_query_once(user_query, args.k, args.limit, args.show_debug)

    except KeyboardInterrupt:
        print("\n🛑 Dừng chương trình.")
    except Exception as e:
        print("❌ Lỗi khi xử lý truy vấn:", e)
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
