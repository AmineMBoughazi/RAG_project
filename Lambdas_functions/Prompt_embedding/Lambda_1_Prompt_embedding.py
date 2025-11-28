import json
import boto3
import os
import psycopg2

DB_HOST = "db-embeddings.cjyesqq620p9.eu-west-3.rds.amazonaws.com"

bedrock = boto3.client('bedrock-runtime')

conn = psycopg2.connect(
    host=DB_HOST,
    user="postgres",
    password="DatabasePassword1234",
    database="postgres",
    sslmode="require"
)
cur =  conn.cursor()
print("connecté à la base")

def generate_embedding(text):
    """Generates a sentence embedding"""

    response = bedrock.invoke_model(
        modelId = "amazon.titan-embed-text-v2:0",
        contentType = "application/json",
        accept = "application/json",
        body = json.dumps({"inputText": text})
    )
    embedding = json.loads(response["body"].read())['embedding']
    return embedding

def lambda_handler(event = -1, context = -1):
    prompt = "- Qui est concerné par la réforme de la facturation electronique ?"

    prompt_embedding = generate_embedding(prompt)

    create_tmp_embeddings_table()

    query = f"""
    SELECT 
    name,
    id,
    metadata,
    vecteur <-> '{prompt_embedding}' as distance
    FROM embeddings
    order by distance
    LIMIT 4
    """
    cur.execute(query)
    responses = cur.fetchall()
    print(responses)
    return

def create_tmp_embeddings_table() :

    def generate_vector(dim=1024):
        """Génère un vecteur de 1024 floats pour l'exemple."""
        import random
        return [random.uniform(-1, 1) for _ in range(dim)]
    cur.execute("DROP TABLE IF EXISTS embeddings")
    query = """
    CREATE TABLE embeddings (
    id SERIAL PRIMARY KEY,
    name TEXT,
    metadata JSONB,
    vecteur VECTOR(1024)   -- colonne pgvector
    );
    """
    cur.execute(query)
    entries = [
        ("doc_1", 1, {"source": "pdf1", "page": 5}, generate_vector()),
        ("doc_2", 2, {"source": "pdf2", "page": 12}, generate_vector()),
        ("doc_3", 3, {"source": "pdf3", "page": 1}, generate_vector()),
        ("doc_4", 4, {"source": "pdf4", "page": 8}, generate_vector()),
    ]

    for name, id_val, metadata, vecteur in entries:
        cur.execute(
            """
            INSERT INTO embeddings (name, id, metadata, vecteur)
            VALUES (%s, %s, %s, %s)
            """,
            (name, id_val, json.dumps(metadata), vecteur)
        )


#lambda_handler()