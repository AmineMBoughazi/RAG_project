import json
import boto3
import os
import psycopg2
import pymupdf4llm
import pymupdf.layout
"""
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
"""
def extract_text(pdf_file):
    md_read = pymupdf4llm.LlamaMarkdownReader()
    data = md_read.load_data(pdf_file)
    print(data)
    return

def generate_embedding(text):
    """Generates a chunk embedding"""
    response = bedrock.invoke_model(
        modelId = "amazon.titan-embed-text-v2:0",
        contentType = "application/json",
        accept = "application/json",
        body = json.dumps({"inputText": text})
    )
    embedding = json.loads(response["body"].read())['embedding']
    return embedding

def lambda_handler(event = -1, context = -1):

    return



extract_text("Documents/paper_recherche_oceans.pdf")