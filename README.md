# 1. Build và start
docker-compose up -d --build

docker compose down
docker compose up -d --build


# 2. Kiểm tra docker đã build ổn chưa
docker ps


# 3. chạy embedding đồ thị
docker exec -it hybrid_rag_app bash
python scripts/embed_graph_neo4j.py

