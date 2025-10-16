import os
from neo4j import GraphDatabase
from openai import OpenAI
from dotenv import load_dotenv
from app.retrievers.nl2cypher_retriever import NL2CypherRetriever
import streamlit as st


# Cấu hình
load_dotenv()


def get_var(key: str, default=None, section="general"):
    try:
        # ưu tiên đọc từ secrets trên Streamlit Cloud
        return st.secrets[section].get(key, default)
    except Exception:
        # fallback về .env khi chạy local
        return os.getenv(key, default)
    
NEO4J_URI = get_var("NEO4J_URI")
NEO4J_USER = get_var("NEO4J_USER")
NEO4J_PASSWORD = get_var("NEO4J_PASSWORD")
OPENAI_MODEL = get_var("OPENAI_MODEL", "gpt-4o-mini")



# NEO4J EXECUTOR
# Kết nối với Neo4j và thực thi Cypher
class Neo4jExecutor:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def run_query(self, cypher_query: str):
        """Thực thi Cypher và trả kết quả dạng list[dict]"""
        with self.driver.session() as session:
            result = session.run(cypher_query)
            return [record.data() for record in result]

    def close(self):
        self.driver.close()



# GRAPH QUERY PIPELINE
class GraphQueryPipeline:
    # Khởi tạo các thành phần
    def __init__(self):
        self.retriever = NL2CypherRetriever()
        self.client = OpenAI()
        self.neo4j = Neo4jExecutor()

    # Làm sạch kết quả LLM trả về
    def clean_cypher(self, text: str) -> str:
        """Làm sạch kết quả LLM (loại bỏ ```cypher...)"""
        if not text:
            return ""
        return (
            text.replace("```cypher", "")
            .replace("```", "")
            .strip()
        )

    # Gửi prompt đến LLM để sinh Cypher
    def generate_cypher(self, user_query: str, k: int = 3) -> str:
        """Dùng LLM để sinh Cypher từ câu hỏi"""
        # Build prompt từ nl2cypher
        prompt = self.retriever.build_prompt(user_query, k=k)

        print("\n📤 GỬI PROMPT ĐẾN OPENAI...\n")
        response = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )

        cypher = self.clean_cypher(response.choices[0].message.content)
        print("\n✅ Cypher sinh ra:\n", cypher)
        return cypher
    
    # Thực thi pineline nhận câu hỏi => Cypher => Kết quả
    def run_pipeline(self, user_query: str):
        """Full pipeline: NL → Cypher → Query → Result"""
        cypher_query = self.generate_cypher(user_query)
        print("\n⚙️ Đang chạy truy vấn trên Neo4j...\n")
        try:
            records = self.neo4j.run_query(cypher_query)
            print(f"📊 Trả về {len(records)} kết quả.")
            return {"query": cypher_query, "result": records}
        except Exception as e:
            print("❌ Lỗi khi chạy Cypher:", e)
            return {"query": cypher_query, "error": str(e)}



# DEMO CHẠY THỬ
if __name__ == "__main__":
    pipeline = GraphQueryPipeline()
    question = "Tìm nhà 5 tầng sổ đỏ chính chủ đầy đủ nội thất tại Thanh Xuân"
    result = pipeline.run_pipeline(question)

    print("\n===== KẾT QUẢ TRẢ VỀ =====")
    print(result)



# Test bug
print(" DEBUG NEO4J CONFIG ")
print("NEO4J_URI:", repr(NEO4J_URI))
print("NEO4J_USER:", repr(NEO4J_USER))
print("NEO4J_PASSWORD:", "SET" if NEO4J_PASSWORD else "None")
print("=======================================")

