"""
 Test script cho module vector_tools.py
Chạy:
    python -m scripts.test_vector_tools
"""

import os
import sys
from dotenv import load_dotenv

# đảm bảo import được app/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.retrievers.vector_tools import VectorClient



# HÀM CHẠY TEST TỰ ĐỘNG
def test_vector_search_basic():
    print("🚀 TEST TÌM KIẾM SEMANTIC TRONG VECTOR DB\n")
    client = VectorClient()

    test_queries = [
        "Tìm nhà 5 tầng sổ đỏ chính chủ tại Thanh Xuân",
        "Có căn hộ 60m2 giá khoảng 3 tỷ ở Cầu Giấy không",
    ]

    for i, query in enumerate(test_queries, start=1):
        print(f"\n================= 🔎 TEST CASE {i} =================")
        print(f"❓ Câu hỏi: {query}")
        result = client.search(query, k=10, mmr=True)

        if result.error:
            print("❌ Lỗi:", result.error)
            continue

        print(f"✅ Tìm thấy {len(result.passages)} kết quả trong {result.took_ms}ms:\n")
        for j, p in enumerate(result.passages, start=1):
            print(f"({j}) 🧩 id={p.id or 'N/A'} | score={round(p.score,4) if p.score else '—'}")
            preview = p.text.replace("\n", " ")
            if len(preview) > 180:
                preview = preview[:180] + "..."
            print(f"    📝 {preview}")
            if p.metadata:
                print(f"    ℹ️  metadata keys: {list(p.metadata.keys())}")
            print()
        print("=====================================================")



# HÀM CHO PHÉP NHẬP CÂU HỎI
def interactive_mode():
    print("\n🗨️  CHẾ ĐỘ NHẬP CÂU HỎI TƯƠNG TÁC (gõ 'exit' để thoát)\n")
    client = VectorClient()

    while True:
        query = input("❓ Nhập câu hỏi: ").strip()
        if not query:
            continue
        if query.lower() in ["exit", "quit", "q"]:
            print("👋 Thoát chế độ test vector.")
            break

        result = client.search(query, k=5, mmr=True)
        if result.error:
            print("❌ Lỗi:", result.error)
            continue

        print(f"✅ Kết quả ({len(result.passages)}) trong {result.took_ms}ms:\n")
        for j, p in enumerate(result.passages, start=1):
            print(f"({j}) id={p.id or 'N/A'} | score={round(p.score,4) if p.score else '—'}")
            preview = p.text.replace("\n", " ")
            print(f"    {preview[:200]}{'...' if len(preview)>200 else ''}\n")



# MAIN
if __name__ == "__main__":
    print("🧪 BẮT ĐẦU TEST VECTOR TOOLS...\n")
    load_dotenv()

    try:
        test_vector_search_basic()
        interactive_mode()
    except KeyboardInterrupt:
        print("\n🛑 Dừng test.")
    print("\n🎯 HOÀN TẤT TOÀN BỘ KIỂM TRA.")
