# run from repo root: python - <<'PY'
import os
from neo4j import GraphDatabase, basic_auth

uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
user = os.getenv("NEO4J_USER", "neo4j")
pwd = os.getenv("NEO4J_PASSWORD", "password")
db = os.getenv("NEO4J_DATABASE", "neo4j")

with open("data/sample_import.cypher") as f:
    statements = f.read()

driver = GraphDatabase.driver(uri, auth=basic_auth(user, pwd))
with driver.session(database=db) as session:
    for stmt in statements.split(";"):
        s = stmt.strip()
        if s:
            session.run(s)
driver.close()
PY
