"""
 Test script cho module graph_tools.py
Chạy:
    python -m scripts.test_graph_pipeline
"""

import os
import sys
from dotenv import load_dotenv

# Cho phép import module app/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.retrievers.graph_tools import GraphQueryPipeline, Neo4jExecutor


# TEST 1: KIỂM TRA KẾT NỐI NEO4J
def test_connection_only():
    print("🔌 Kiểm tra kết nối Neo4j...")
    executor = Neo4jExecutor()
    try:
        records = executor.run_query("MATCH (n) RETURN DISTINCT labels(n) AS labels LIMIT 5;")
        print("✅ Kết nối thành công! Một vài node:")
        for r in records:
            print("  -", r)
    except Exception as e:
        print("❌ Lỗi kết nối:", e)
    finally:
        executor.close()


# TEST 2: CHẠY PIPELINE VỚI DANH SÁCH CÂU HỎI MẪU
def run_predefined_tests(pipeline: GraphQueryPipeline):
    test_queries = [
        "Tìm nhà 5 tầng gần đại học Thương Mại",
        "Có nhà sổ đỏ chính chủ tại Thanh Xuân không",
        "Tìm bất động sản hướng Tây Tứ Trạch ở Long Biên",
    ]

    for i, query in enumerate(test_queries, start=1):
        print(f"\n================= 🧠 TEST CASE {i} =================")
        print(f"❓ Câu hỏi người dùng: {query}\n")

        show_semantic_examples(pipeline, query)
        run_query_and_show_result(pipeline, query)



# TEST 3: CHO PHÉP NHẬP CÂU HỎI THỦ CÔNG
def interactive_query_mode(pipeline: GraphQueryPipeline):
    print("\n🗨️  CHẾ ĐỘ NHẬP CÂU HỎI TƯƠNG TÁC (gõ 'exit' để thoát)\n")
    while True:
        user_query = input("❓ Nhập câu hỏi: ").strip()
        if not user_query:
            continue
        if user_query.lower() in ["exit", "quit", "q"]:
            print("👋 Thoát chế độ nhập tay.")
            break

        show_semantic_examples(pipeline, user_query)
        run_query_and_show_result(pipeline, user_query)


# HÀM PHỤ TRỢ
def show_semantic_examples(pipeline: GraphQueryPipeline, query: str):
    """Hiển thị top-3 câu hỏi gần nghĩa nhất và Cypher mẫu"""
    examples = pipeline.retriever.retrieve_examples(query, k=3)
    print("🔍 Top-3 câu hỏi gần nghĩa nhất (theo semantic search):")
    for j, ex in enumerate(examples, start=1):
        print(f"({j}) ❓ {ex['Question']}")
        print(f"    ↳ 📘 Cypher mẫu tương ứng:")
        print(f"{ex['Cypher'][:300]}{'...' if len(ex['Cypher'])>300 else ''}\n")


def run_query_and_show_result(pipeline: GraphQueryPipeline, query: str):
    """Chạy pipeline với query và in kết quả"""
    result = pipeline.run_pipeline(query)
    print("🔹 Cypher GPT sinh ra:")
    print(result.get("query", "(Không có truy vấn được sinh ra)"))

    if "error" in result:
        print("❌ Lỗi khi chạy Cypher:", result["error"])
    else:
        records = result.get("result", [])
        print(f"📊 Số bản ghi trả về: {len(records)}")
        if records:
            print("🔸 Ví dụ bản ghi đầu tiên:")
            print(records[0])
    print("===================================================")



# MAIN
if __name__ == "__main__":
    print("🧪 BẮT ĐẦU TEST GRAPH PIPELINE...\n")
    load_dotenv()

    # 1️⃣ Kiểm tra kết nối Neo4j
    test_connection_only()

    # 2️⃣ Khởi tạo pipeline
    print("\n🚀 KHỞI TẠO GRAPH PIPELINE...\n")
    pipeline = GraphQueryPipeline()

    # 3️⃣ Chạy test query mặc định
    run_predefined_tests(pipeline)

    # 4️⃣ Cho phép nhập câu hỏi tùy ý
    interactive_query_mode(pipeline)

    print("\n🎯 HOÀN TẤT TOÀN BỘ KIỂM TRA.")
