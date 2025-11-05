import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

print("🔍 Kiểm tra API key:", api_key[:8] + "..." if api_key else "❌ Không có key!")

try:
    client = OpenAI(api_key=api_key)

    # 1 Test model chat
    print("\nĐang test model chat:", model)
    chat_response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Hello, are you working?"}],
    )
    print("Chat model trả lời:", chat_response.choices[0].message.content)

    # 2 Test embedding
    print("\nĐang test embedding:", embed_model)
    emb_response = client.embeddings.create(
        model=embed_model,
        input="Xin chào Hà Nội, đây là test embedding."
    )
    print("Nhận được vector độ dài:", len(emb_response.data[0].embedding))

    print("\nOpenAI API hoạt động bình thường!")
except Exception as e:
    print("Lỗi khi gọi API OpenAI:", str(e))
