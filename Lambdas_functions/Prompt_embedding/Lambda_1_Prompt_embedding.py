import json
import boto3
import os
import psycopg2
import logging
from dotenv import load_dotenv


# Logging Configuration
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

db_config = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "dbname": os.getenv("DB_NAME"),
    "port": os.getenv("DB_PORT"),
    "sslmode": os.getenv("DB_SSLMODE"),
}

bedrock = boto3.client('bedrock-runtime')

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
    prompt = "à quoi sert le bloc de donnée de paiement ? "


    try:
        connection = psycopg2.connect(**db_config)
        print("Connected to the database!")
        with connection.cursor() as cursor:
            prompt_embedding = generate_embedding(prompt)
            query = f"""
            SELECT 
            chunk_id,
            document_id,
            text,
            image_id,
            embeddings <-> '{prompt_embedding}' as distance
            FROM chunks_table
            order by distance
            LIMIT 4
            """
            cursor.execute(query)
            responses = cursor.fetchall()
            for each in responses:
                print(each)
                print('\n--------------------------------------------------------------------\n')
    except Exception as e :
        print(f"Error inserting data: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error inserting data: {str(e)}')
        }
    finally:
        # Fermer la connexion
        if 'connection' in locals():
            connection.close()
            print("Connexion à la base de données fermée.")
    return


lambda_handler()