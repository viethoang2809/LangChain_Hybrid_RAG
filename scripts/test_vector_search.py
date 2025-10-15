from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

emb = OpenAIEmbeddings(model="text-embedding-3-small")
vdb = FAISS.load_local(".vector_store/text_embeddings", emb, allow_dangerous_deserialization=True)

query = "Nhà 5 tầng ở Cầu Giấy"
docs = vdb.similarity_search(query, k=3)
for i, d in enumerate(docs, 1):
    print(f"#{i} ID: {d.metadata['id']}")
    print(d.page_content[:500], "\n")
