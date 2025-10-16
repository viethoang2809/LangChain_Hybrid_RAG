"""
🧪 Test script cho Hybrid Retriever (Neo4j + FAISS)
Chạy:
    python -m scripts.test_hybrid_retriever
"""

import os
import sys
from dotenv import load_dotenv

# đảm bảo import được app/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.retrievers.hybrid_retriever import HybridRetriever


# CHẠY TEST TỰ ĐỘNG
def test_hybrid_queries():
    print("\n🚀 BẮT ĐẦU TEST HYBRID RETRIEVER\n")
    retriever = HybridRetriever()

    # === Các câu hỏi mẫu ===
    test_queries = [
        "Tìm nhà 5 tầng sổ đỏ chính chủ tại Thanh Xuân",
        "Có nhà hướng Tây Bắc ở Cầu Giấy không",
        "Bất động sản gần Đại học Thương Mại giá khoảng 5 tỷ",
    ]

    for i, q in enumerate(test_queries, start=1):
        print(f"\n================= 🧠 TEST CASE {i} =================")
        print(f"❓ Câu hỏi: {q}")

        result = retriever.search(q, top_k=10)

        # --- In thống kê ---
        print("\n===== TỔNG HỢP KẾT QUẢ =====")
        print(f"🕐 Thời gian: {result['took_ms']}ms")
        print(f"📊 Graph IDs: {result['graph_ids']}")
        print(f"📚 Vector hits: {result['vector_hits']}")
        print(f"🔀 Fused top: {len(result['fused_results'])} kết quả\n")

        for j, p in enumerate(result["fused_results"][:5], start=1):
            score = f"{round(p.score,4):.4f}" if p.score else "—"
            print(f"({j}) id={p.id or 'N/A'} | score={score}")
            snippet = p.text.replace("\n", " ")
            print(f"    📝 {snippet[:200]}{'...' if len(snippet)>200 else ''}")
        print("===================================================")


# CHẾ ĐỘ NHẬP CÂU HỎI
def interactive_mode():
    print("\n🗨️  CHẾ ĐỘ NHẬP CÂU HỎI TƯƠNG TÁC (gõ 'exit' để thoát)\n")
    retriever = HybridRetriever()

    while True:
        q = input("❓ Nhập câu hỏi: ").strip()
        if not q:
            continue
        if q.lower() in ["exit", "quit", "q"]:
            print("👋 Thoát chế độ test hybrid retriever.")
            break

        result = retriever.search(q, top_k=10)
        print("\n===== KẾT QUẢ HỢP NHẤT =====")
        print(f"🕐 Thời gian: {result['took_ms']}ms")
        print(f"📊 Graph IDs: {result['graph_ids']}")
        print(f"📚 Vector hits: {result['vector_hits']}\n")

        for j, p in enumerate(result["fused_results"][:5], start=1):
            score = f"{round(p.score,4):.4f}" if p.score else "—"
            print(f"({j}) id={p.id or 'N/A'} | score={score}")
            print(f"    {p.text[:200]}{'...' if len(p.text)>200 else ''}")
        print("===================================================")



# MAIN
if __name__ == "__main__":
    print("🧪 TEST HYBRID RETRIEVER\n")
    load_dotenv()

    try:
        test_hybrid_queries()
        interactive_mode()
    except KeyboardInterrupt:
        print("\n🛑 Dừng test.")
    print("\n🎯 HOÀN TẤT TOÀN BỘ KIỂM TRA.")
