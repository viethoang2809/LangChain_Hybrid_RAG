# app/main_cli.py
import os, sys, json, traceback, argparse
from dotenv import load_dotenv
from openai import OpenAI

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.retrievers.hybrid_retriever import HybridRetriever
from app.retrievers.vector_tools import VectorClient
from app.utils.hybrid_helpers import (
    load_answer_rule,
    build_id_map_from_graph_records,
    select_top3_by_priority,
    build_synthesis_input,
    llm_summarize_answer,
)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=OPENAI_API_KEY)


def run_query_once(user_query: str, top_k: int = 10, limit: int = 3, show_debug: bool = False):
    """Chạy một truy vấn Hybrid RAG duy nhất."""
    print(f"\n❓ {user_query}\n")

    synth_rule = load_answer_rule()
    hybrid = HybridRetriever()
    vclient = hybrid.vector

    print("⏳ Đang truy vấn dữ liệu từ Neo4j và FAISS...\n")
    hybrid_result = hybrid.search(user_query=user_query, top_k=top_k)
    graph_records = hybrid_result["graph_records"]
    graph_ids = hybrid_result["graph_ids"]
    vector_passages = hybrid_result["vector_passages"]

    graph_id_map = build_id_map_from_graph_records(graph_records)
    chosen_passages = select_top3_by_priority(graph_ids, vector_passages, vclient, graph_id_map, fill_limit=limit)

    if show_debug:
        print("📊 Graph IDs:", graph_ids)
        print("📚 Vector IDs:", [p.id for p in vector_passages])
        print("✅ Chosen IDs:", [p.id for p in chosen_passages])
        print()

    synthesis_payload = build_synthesis_input(chosen_passages, graph_id_map)
    print("🧠 Đang tổng hợp câu trả lời...\n")
    answer = llm_summarize_answer(client, user_query, synth_rule, synthesis_payload, OPENAI_MODEL)

    print("✨ CÂU TRẢ LỜI:\n------------------------------------------------")
    print(answer)
    print("------------------------------------------------")

    # Hiển thị snippet ngắn
    print("\n📋 DỮ LIỆU HỢP NHẤT:")
    for p in chosen_passages:
        pid = str(p.id).strip() if p.id else "N/A"
        snippet = (p.text or "").strip()[:120].replace("\n", " ")
        print(f"• ID {pid}: {snippet}...")


def main():
    parser = argparse.ArgumentParser(description="Hybrid RAG CLI cho Bất động sản Hà Nội")
    parser.add_argument("--query", type=str, help="Câu hỏi người dùng (nếu không có, sẽ bật chế độ nhập tay)")
    parser.add_argument("--k", type=int, default=10, help="Số lượng top-k kết quả vector")
    parser.add_argument("--limit", type=int, default=3, help="Giới hạn số căn để tổng hợp")
    parser.add_argument("--show-debug", action="store_true", help="Hiển thị debug IDs")
    args = parser.parse_args()

    print("🏠 Hybrid RAG – Bất động sản Hà Nội (CLI mode)")
    print("================================================")

    try:
        # Nếu có query -> chạy một lần rồi thoát
        if args.query:
            run_query_once(args.query, args.k, args.limit, args.show_debug)
            return

        # Không có query -> bật chế độ tương tác
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
