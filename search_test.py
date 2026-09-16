import os, psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3")
q = "how much can I expense for home office equipment?"
vec = model.encode([q], normalize_embeddings=True)[0]

conn = psycopg.connect(os.environ["DATABASE_URL"])
register_vector(conn)
rows = conn.execute("""
    SELECT c.heading_path, 1 - (e.embedding <=> %s) AS score
    FROM chunk_embeddings_bge_1024 e
    JOIN chunks c ON c.id = e.chunk_id AND c.status = 'active'
    WHERE e.department = 'finance'
    ORDER BY e.embedding <=> %s
    LIMIT 5
""", (vec, vec)).fetchall()

for path, score in rows:
    print(f"{score:.3f}  {path}")