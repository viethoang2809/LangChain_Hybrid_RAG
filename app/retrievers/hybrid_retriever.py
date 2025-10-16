"""
Hybrid Retriever: chạy song song Graph (Neo4j) + Vector (FAISS)
"""
import os
import time
import asyncio
from typing import Dict, Any
from app.retrievers.graph_tools import GraphQueryPipeline
from app.retrievers.vector_tools import VectorClient
from openai import OpenAI


class HybridRetrieverParallel:
    def __init__(self):
        self.graph = GraphQueryPipeline()
        self.vector = VectorClient()
        self.client = OpenAI()
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    async def _graph_search(self, query: str):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.graph.run_pipeline, query)

    async def _vector_search(self, query: str, top_k: int):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.vector.search, query, top_k, True)

    async def search(self, user_query: str, top_k: int = 10) -> Dict[str, Any]:
        """Chạy song song thật sự giữa Graph và Vector."""
        start = time.time()
        print("\n🚀 Đang chạy song song Graph + Vector...\n")

        # Chạy hai nhiệm vụ song song
        graph_task = asyncio.create_task(self._graph_search(user_query))
        vector_task = asyncio.create_task(self._vector_search(user_query, top_k))
        graph_result, vector_result = await asyncio.gather(graph_task, vector_task)

        took = int((time.time() - start) * 1000)

        graph_records = graph_result.get("result") or []
        graph_ids = [str(r.get("id")).strip() for r in graph_records if r.get("id")]
        vector_passages = vector_result.passages if not vector_result.error else []

        print(f"✅ Graph xong: {len(graph_records)} kết quả")
        print(f"✅ Vector xong: {len(vector_passages)} kết quả")
        print(f"⚡ Tổng thời gian song song: {took}ms")

        return {
            "query": user_query,
            "graph_records": graph_records,
            "graph_ids": graph_ids,
            "vector_passages": vector_passages,
            "vector_time_ms": vector_result.took_ms,
            "vector_error": vector_result.error,
            "took_ms": took,
        }


# === TEST ===
if __name__ == "__main__":
    import asyncio

    retriever = HybridRetrieverParallel()
    q = "Tìm nhà 5 tầng sổ đỏ chính chủ tại Thanh Xuân"

    print(f"\n❓ Câu hỏi: {q}\n")
    result = asyncio.run(retriever.search(q, top_k=10))

    print("\n===== KẾT QUẢ =====")
    print(f"🕐 Tổng thời gian: {result['took_ms']}ms")
    print(f"📊 Graph IDs: {result['graph_ids'][:10]}")
    print(f"📚 Vector hits: {len(result['vector_passages'])}")
