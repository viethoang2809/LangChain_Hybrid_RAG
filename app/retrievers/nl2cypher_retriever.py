import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


class NL2CypherRetriever:
    """
    NL2Cypher Retriever:
    - So sánh semantic giữa user query và các mẫu 'Question' trong CSV.
    - Lấy 'Cypher' tương ứng làm ví dụ few-shot.
    - Ghép với schema/rule từ file prompt để sinh full prompt gửi GPT.
    """

    def __init__(
        self,
        csv_path="data/Cypher_template.csv",
        schema_path="app/prompts/nl2cypher_vi.txt",
        store_dir=".vector_store/nl2cypher_index",
        embed_model="text-embedding-3-small",
    ):
        load_dotenv()
        self.csv_path = csv_path
        self.schema_path = schema_path
        self.store_dir = store_dir
        self.embed_model = embed_model
        self.embeddings = OpenAIEmbeddings(model=self.embed_model)
        self.vdb = None

        os.makedirs(self.store_dir, exist_ok=True)
        self.schema_text = self._load_schema()
        self._load_or_build_index()

    # ======================
    # 🔹 LOAD SCHEMA PROMPT
    # ======================
    def _load_schema(self):
        if not os.path.exists(self.schema_path):
            raise FileNotFoundError(f"❌ Không tìm thấy file schema: {self.schema_path}")
        with open(self.schema_path, "r", encoding="utf-8") as f:
            schema_text = f.read().strip()
        print(f"📜 Đã load schema từ {self.schema_path}")
        return schema_text

    # ======================
    # 🔹 LOAD / BUILD INDEX
    # ======================
    def _load_or_build_index(self):
        faiss_path = os.path.join(self.store_dir, "index.faiss")
        if os.path.exists(faiss_path):
            print("📦 Đang load FAISS index có sẵn...")
            self.vdb = FAISS.load_local(
                folder_path=self.store_dir,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            print("🚀 Chưa có index — đang tạo mới từ CSV...")
            self._build_index()

    def _build_index(self):
        df = pd.read_csv(self.csv_path)
        if not {"Question", "Cypher"}.issubset(df.columns):
            raise ValueError("❌ CSV phải có 2 cột: 'Question' và 'Cypher'")

        texts = df["Question"].astype(str).tolist()
        metadatas = [{"Cypher": row["Cypher"]} for _, row in df.iterrows()]

        self.vdb = FAISS.from_texts(texts, embedding=self.embeddings, metadatas=metadatas)
        self.vdb.save_local(self.store_dir)
        print(f"✅ Đã tạo FAISS index từ {len(df)} ví dụ.")

    # ======================
    # 🔹 TRUY XUẤT VÍ DỤ
    # ======================
    def retrieve_examples(self, query: str, k: int = 3):
        """Tìm top-k ví dụ semantic gần nhất trong index"""
        if not self.vdb:
            raise RuntimeError("⚠️ VectorDB chưa được load hoặc build.")
        results = self.vdb.similarity_search(query, k=k)
        return [{"Question": r.page_content, "Cypher": r.metadata["Cypher"]} for r in results]

    def debug_retrieve(self, query: str, k: int = 3):
        """In ra ví dụ gần nghĩa nhất để debug"""
        examples = self.retrieve_examples(query, k)
        print(f"\n🔍 Top-{k} ví dụ semantic gần nhất cho: '{query}'\n")
        for i, ex in enumerate(examples, 1):
            print(f"--- Ví dụ {i} ---")
            print("❓ Question:", ex["Question"])
            print("💬 Cypher:", ex["Cypher"])
            print()

    # ======================
    # 🔹 TẠO PROMPT CHO GPT
    # ======================
    def build_prompt(self, user_query: str, k: int = 3):
        """Ghép prompt hoàn chỉnh để gửi GPT"""
        examples = self.retrieve_examples(user_query, k=k)
        few_shot_text = "\n\n".join(
            [
                f"(Ví dụ {i+1})\n"
                f"Hỏi (ngữ nghĩa tương tự): {ex['Question']}\n"
                f"Truy vấn Cypher tương ứng:\n{ex['Cypher']}"
                for i, ex in enumerate(examples)
            ]
        )

        prompt = f"""
{self.schema_text}

Dưới đây là một vài ví dụ tương tự (retrieved bằng semantic search):
{few_shot_text}

Câu hỏi người dùng:
{user_query}

Chỉ trả về DUY NHẤT code block chứa truy vấn Cypher hợp lệ.
"""
        return prompt


# ======================
# 🔹 DEMO
# ======================
if __name__ == "__main__":
    retriever = NL2CypherRetriever()
    query = "Tìm nhà sổ đỏ chính chủ tại Thanh Xuân"
    retriever.debug_retrieve(query)
    print("\n🧠 PROMPT GỬI GPT:\n")
    print(retriever.build_prompt(query, k=3))
