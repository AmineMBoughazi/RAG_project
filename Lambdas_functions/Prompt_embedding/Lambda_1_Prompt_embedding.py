import json
import boto3
import os
import psycopg2

bedrock = boto3.client('bedrock-runtime')
token = boto3.client("rds").generate_db_auth_token(
    DBHostname=DB_HOST,
    Port=5432,
    DBUsername="postgres"
)

conn = psycopg.connect(
    host=DB_HOST,
    user="postgres",
    password=token,
    sslmode="require"
)

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
    print(prompt_embedding)
    return


lambda_handler()