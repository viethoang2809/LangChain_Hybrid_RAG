from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
database = os.getenv("NEO4J_DATABASE", "neo4j")

driver = GraphDatabase.driver(uri, auth=(user, password))
with driver.session(database=database) as session:
    res = session.run("MATCH (n) RETURN COUNT(n) AS total_nodes;")
    print("✅ Kết nối thành công!")
    print("Số lượng node trong DB:", res.single()["total_nodes"])
