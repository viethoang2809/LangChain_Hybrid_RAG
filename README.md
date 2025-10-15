# 🏠 LangChain Hybrid RAG – Bất Động Sản Hà Nội

## 🚀 Giới thiệu

Dự án **LangChain Hybrid RAG** giúp xây dựng hệ thống hỏi đáp thông minh về **bất động sản Hà Nội**, kết hợp hai hướng truy xuất:

- **Neo4j Graph Database** → lưu trữ và truy vấn dữ liệu có cấu trúc (theo node, quan hệ)
- **FAISS VectorDB** → tìm kiếm ngữ nghĩa từ văn bản mô tả nhà đất
- **LangChain + OpenAI API** → tổng hợp kết quả và sinh câu trả lời tự nhiên

Ứng dụng được triển khai trên **Streamlit Cloud** cho phép người dùng hỏi trực tiếp các câu như:

> “Tìm nhà 5 tầng sổ đỏ chính chủ đầy đủ nội thất tại Thanh Xuân”

và hệ thống sẽ:
1. Sinh ra truy vấn **Cypher** tương ứng  
2. Thực thi trên Neo4j  
3. Kết hợp với dữ liệu embedding để tạo phản hồi tự nhiên.


# 💻 CÀI ĐẶT VÀ CHẠY LOCAL

# 1️⃣ Clone repository từ GitHub
git clone https://github.com/VietHoangg03/LangChain_Hybrid_RAG.git
cd LangChain_Hybrid_RAG

# 2️⃣ Tạo môi trường ảo (virtual environment)
python3 -m venv venv

# 3️⃣ Kích hoạt môi trường ảo
# 👉 Dành cho macOS / Linux:
source venv/bin/activate

# 👉 Dành cho Windows:
venv\Scripts\activate

# 4️⃣ Cài đặt các thư viện cần thiết
pip install -r requirements.txt

# 5️⃣ Tạo file .env ở thư mục gốc (để lưu API key và config)
# ⚠️ Lưu ý: KHÔNG push file này lên GitHub vì có chứa key bí mật
# Mở file .env và thêm nội dung sau:

# ===========================================
# 🔐 ENVIRONMENT CONFIG (.env)
# ===========================================
OPENAI_API_KEY=sk-xxxx                     # 🔑 Key của bạn lấy từ https://platform.openai.com/api-keys
NEO4J_URI=neo4j+s://04c8805a.databases.neo4j.io   # 🌐 URI kết nối Neo4j Aura
NEO4J_USER=neo4j                           # 👤 Tên đăng nhập Neo4j
NEO4J_PASSWORD=your_password               # 🔒 Mật khẩu Neo4j
VECTOR_STORE_PATH=.vector_store/text_embeddings   # 📁 Thư mục chứa FAISS vector data
# ===========================================

# 6️⃣ Chạy thử ứng dụng trên local
streamlit run app/main.py

# 💡 Sau khi chạy, Streamlit sẽ mở giao diện tại:
# 👉 http://localhost:8501
# (Nếu không tự mở, copy link này dán vào trình duyệt)



# 🧩 CÁCH DÙNG GIT (CHUẨN VÀ AN TOÀN)

# 🔹 1️⃣ Kiểm tra trạng thái hiện tại của project
# Hiển thị các file đã thay đổi, thêm mới, hoặc chưa được commit
git status

# 🔹 2️⃣ Thêm file đã sửa hoặc thêm mới vào staging area
# Nếu muốn thêm tất cả thay đổi:
git add .

# (Hoặc thêm 1 file cụ thể, ví dụ:)
# git add app/retrievers/graph_tools.py

# 🔹 3️⃣ Tạo commit với nội dung mô tả thay đổi
git commit -m "fix: update config and add streamlit import"

# 🔹 4️⃣ Kéo bản mới nhất từ GitHub về trước khi push (tránh lỗi 'rejected')
git pull origin main --rebase

# 🔹 5️⃣ Push code từ máy local lên GitHub
git push origin main

# ⚠️ Nếu thấy lỗi:
# ! [rejected] main -> main (fetch first)
# (Tức là repo online có commit mới hơn repo local)
# => Hãy chạy lại lệnh dưới để đồng bộ:

git pull origin main --rebase
git push origin main

# 💣 Nếu bạn muốn ghi đè toàn bộ branch trên GitHub (chỉ dùng khi chắc chắn!)
# ❗ Lệnh này sẽ ghi đè lịch sử commit online bằng code hiện tại trên máy bạn
git push origin main --force

# ✅ Sau khi push thành công, bạn có thể kiểm tra lại bằng:
# - Vào GitHub → repo → tab “Commits”
# - Xem commit mới nhất hiển thị ở trên cùng



# 🧾 THÔNG TIN TÁC GIẢ

# 👤 Tác giả:
#   Viet Hoang

# 📂 Dự án:
#   LangChain Hybrid RAG – Bất Động Sản Hà Nội

# 🧠 Công nghệ sử dụng:
#   - Neo4j (Graph Database)
#   - FAISS (Vector Search)
#   - LangChain (Orchestration Framework)
#   - Streamlit (Frontend UI)
#   - OpenAI API (LLM và Embedding)
