"""
Hybrid Retriever: kết hợp giữa Neo4j (Cypher) và VectorDB (FAISS)
"""
import os
import time
from typing import Dict, Any, List
from app.retrievers.graph_tools import GraphQueryPipeline
from app.retrievers.vector_tools import VectorClient, Passage
from openai import OpenAI



class HybridRetriever:
    """
    Retriever kết hợp giữa:
    - Graph (Neo4j): NL2Cypher → lấy property theo ngữ nghĩa
    - VectorDB (FAISS): tìm kiếm semantic trên văn bản
    """

    def __init__(self):
        self.graph = GraphQueryPipeline()
        self.vector = VectorClient()
        self.client = OpenAI()
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def search(self, user_query: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Truy vấn song song cả Graph và Vector, nhưng chỉ query Neo4j 1 lần.
        """
        # 1. Query Graph (NL2Cypher + thực thi)
        graph_result = self.graph.run_pipeline(user_query)
        graph_records = graph_result.get("result") or []
        graph_ids = [str(r.get("id")).strip() for r in graph_records if r.get("id")]

        # 2. Vector Search
        vector_result = self.vector.search(user_query, k=top_k, mmr=True)
        vector_passages = vector_result.passages if not vector_result.error else []

        # 3. Trả về hợp nhất
        return {
            "query": user_query,
            "graph_records": graph_records,   # đã có đủ dữ liệu (1 lần query)
            "graph_ids": graph_ids,
            "vector_passages": vector_passages,
            "vector_time_ms": vector_result.took_ms,
            "vector_error": vector_result.error,
        }



# TEST NHANH
if __name__ == "__main__":
    retriever = HybridRetriever()

    q = "Tìm nhà 5 tầng sổ đỏ chính chủ tại Thanh Xuân"
    print(f"\n❓ Câu hỏi: {q}")
    result = retriever.search(q, top_k=10)

    print("\n===== TỔNG HỢP KẾT QUẢ =====")
    print(f"🕐 Thời gian: {result['took_ms']}ms")
    print(f"📊 Graph IDs: {result['graph_ids']}")
    print(f"📚 Vector hits: {result['vector_hits']}")
    print("\n🔹 Kết quả hợp nhất (Top 5):")
    for i, p in enumerate(result["fused_results"][:5], start=1):
        print(f"({i}) id={p.id or 'N/A'} | score≈{round(p.score,4) if p.score else '—'}")
        preview = p.text[:200].replace("\n", " ")
        print(f"    {preview}{'...' if len(preview)>200 else ''}")
