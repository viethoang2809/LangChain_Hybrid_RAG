# retrievers/vector_tools.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import os, time, math
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# Cấu hình
load_dotenv()

def get_var(key, default=None, section="general"):
    try:
        return st.secrets[section].get(key, default)
    except Exception:
        return os.getenv(key, default)
    
EMBED_MODEL = get_var("OPENAI_EMBED_MODEL", "text-embedding-3-small")
VECTOR_STORE_PATH = get_var("VECTOR_STORE_PATH", ".vector_store/text_embeddings")

@dataclass
class Passage:
    id: Optional[str]
    text: str
    score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class VectorResult:
    passages: List[Passage]
    took_ms: int
    error: Optional[str] = None

class VectorClient:
    def __init__(self,
                 index_path: str = VECTOR_STORE_PATH,
                 emb_model: str = EMBED_MODEL) -> None:
        self.index_path = index_path
        self.emb_model = emb_model
        self._vs = None
        self._emb = None

    def _get_embeddings(self):
        if self._emb is None:
            self._emb = OpenAIEmbeddings(model=self.emb_model)
        return self._emb

    def _load_vs(self):
        if self._vs is None:
            emb = self._get_embeddings()
            if not os.path.exists(self.index_path):
                raise FileNotFoundError(f"Vector store not found: {self.index_path}")
            self._vs = FAISS.load_local(self.index_path, emb, allow_dangerous_deserialization=True)
        return self._vs

    def search(self, query: str, k: int = 10, mmr: bool = True) -> VectorResult:
        start = time.time()
        passages: List[Passage] = []
        err = None
        try:
            vs = self._load_vs()
            if mmr:
                docs: List[Document] = vs.max_marginal_relevance_search(query, k=k, fetch_k=min(25, max(10, k*2)))
            else:
                docs: List[Document] = vs.similarity_search_with_score(query, k=k)  # returns (Document, score)
                # normalize to consistent structure
                passages = [
                    Passage(
                        id=(doc.metadata or {}).get("id"),
                        text=doc.page_content,
                        score=score if isinstance(score, (int, float)) else None,
                        metadata=doc.metadata or {},
                    )
                    for doc, score in docs
                ]
                took_ms = int((time.time() - start)*1000)
                return VectorResult(passages=passages, took_ms=took_ms, error=None)

            # for MMR path, FAISS doesn't return scores; do a second pass to get scores:
            # compute embedding for query and dot-product with stored vectors is not trivial here,
            # so we fallback to a similarity_search_with_score small k for scoring.
            docs_scored = vs.similarity_search_with_score(query, k=min(k, 10))
            score_map = {}
            for doc, sc in docs_scored:
                # lower score => closer (depending on distance metric), we convert to pseudo-sim
                try:
                    sim = 1.0 / (1.0 + float(sc))
                except Exception:
                    sim = None
                score_map.get("dummy", None)
                # index by text hash (rough), or id if available
                key = (doc.metadata or {}).get("id") or hash(doc.page_content)
                score_map[key] = sim

            # build result
            for d in docs:
                pid = (d.metadata or {}).get("id")
                key = pid or hash(d.page_content)
                passages.append(Passage(
                    id=pid,
                    text=d.page_content,
                    score=score_map.get(key),
                    metadata=d.metadata or {}
                ))
        except Exception as e:
            err = str(e)
        took_ms = int((time.time() - start) * 1000)
        return VectorResult(passages=passages, took_ms=took_ms, error=err)

    @staticmethod
    def rrf_fuse(ids_from_graph: List[str], vector_passages: List[Passage], k: int = 8) -> List[Passage]:
        """
        Reciprocal Rank Fusion: ưu tiên passages có id trùng với graph trước,
        sau đó trộn theo thứ hạng vector. rrf_score = 1/(k + rank).
        """
        # 1) ưu tiên theo match id
        id_set = set([i for i in ids_from_graph if i])
        base: List[Tuple[Passage, float]] = []
        for rank, p in enumerate(vector_passages, start=1):
            boost = 1.0 if (p.id and p.id in id_set) else 0.0
            rrf = 1.0 / (k + rank)
            score = rrf + boost
            base.append((p, score))
        base.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in base]
