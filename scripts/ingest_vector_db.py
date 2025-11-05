import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS, Chroma

# Load biến môi trường
load_dotenv()

DATA_PATH = "data/project-text-semantic.csv"
VDB_DIR = os.getenv("VECTOR_DB_DIR", ".vector_store")
BACKEND = os.getenv("VECTOR_DB_BACKEND", "faiss").lower()
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

os.makedirs(VDB_DIR, exist_ok=True)

print(f"📂 Đang đọc dữ liệu từ: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# Kiểm tra dữ liệu 
if not {"id", "text"}.issubset(df.columns):
    raise ValueError("❌ File CSV phải có 2 cột: 'id' và 'text'")

print(f"Số bài rao cần embedding: {len(df)}")

# Chuẩn bị dữ liệu embedding 
texts = df["text"].astype(str).tolist()
metadatas = [{"id": str(row["id"])} for _, row in df.iterrows()]

# Tạo embedding 
print(f"Đang tạo embedding bằng model: {EMBED_MODEL}")
emb = OpenAIEmbeddings(model=EMBED_MODEL)

if BACKEND == "faiss":
    vdb = FAISS.from_texts(texts, embedding=emb, metadatas=metadatas)
    save_path = os.path.join(VDB_DIR, "text_embeddings")
    vdb.save_local(save_path)
    print(f"Đã lưu FAISS vào: {save_path}")
else:
    save_path = os.path.join(VDB_DIR, "text_embeddings_chroma")
    vdb = Chroma.from_texts(
        texts,
        embedding=emb,
        metadatas=metadatas,
        persist_directory=save_path
    )
    vdb.persist()
    print(f"Đã lưu Chroma vào: {save_path}")

print("Hoàn tất embedding text dataset!")
