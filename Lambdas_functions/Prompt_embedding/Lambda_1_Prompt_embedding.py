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

def get_k_best_chunks(cursor,user_query,k_best):
    prompt_embedding = generate_embedding(user_query)
    query = f"""
        SELECT 
        chunk_id,
        chunk_index,
        document_id,
        text,
        start_offset,
        end_offset,
        embeddings <-> '{prompt_embedding}' as distance
        FROM chunks_table
        order by distance
        LIMIT {k_best}
    """

    cursor.execute(query)
    responses = cursor.fetchall()
    return responses

def merge_overlapping_chunks(chunks):
    """
    Prend une liste de chunks avec `start_offset` et `end_offset`,
    et retourne une liste de segments non-chevauchants :

    - fusionne les chevauchements
    - combine aussi les textes associés
    """
    # 1) Trier par start_offset
    sorted_chunks = sorted(chunks, key=lambda c: c[4])

    merged_segments = []

    for chunk in sorted_chunks:
        if not merged_segments:
            # premier segment
            merged_segments.append({
                "start_offset": chunk[4],
                "end_offset": chunk[5],
                "texts": [chunk[3]]
            })
            continue

        # segment en construction
        last = merged_segments[-1]

        # si le chunk chevauche ou touche l'actuel
        if chunk[4] < last["end_offset"]:
            new_start = last["end_offset"]
            # extension si nécessaire
            last["end_offset"] = max(last["end_offset"], chunk[5])
            last["texts"].append(chunk[3][new_start:])
        else:
            # sinon on ajoute un nouveau segment
            merged_segments.append({
                "start_offset": chunk[4],
                "end_offset": chunk[5],
                "texts": [chunk[3]]
            })

    # 3) Concaténer les textes fusionnés
    aggregate_chunks = ""
    for seg in merged_segments:
        aggregate_chunks += "\n".join(seg["texts"])

    return aggregate_chunks

def call_claude_bedrock(context_prompt: str) -> str:
    bedrock = boto3.client("bedrock-runtime", region_name="eu-west-3")  # adapte si besoin

    body = {
        "modelId": "arn:aws:bedrock:eu-west-3:358309278066:inference-profile/eu.anthropic.claude-sonnet-4-5-20250929-v1:0",  # à ajuster selon ton compte/région
        "contentType": "application/json",
        "accept": "application/json",
        "body": json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": context_prompt}
                    ]
                }
            ],
            "max_tokens": 600,
            "temperature": 0.3
        })
    }

    response = bedrock.invoke_model(
        modelId=body["modelId"],
        body=body["body"],
        accept=body["accept"],
        contentType=body["contentType"],
    )

    response_body = json.loads(response["body"].read())

    # Claude via Bedrock renvoie un tableau de content blocks
    # On récupère tout le texte de la première réponse
    contents = response_body.get("output_text") or response_body.get("content", [])
    if isinstance(contents, list) and contents:
        # Claude style content: [{"role":"assistant","content":[{"type":"text","text":"..."}]}]
        first = contents[0]
        if isinstance(first, dict) and "text" in first:
            return first["text"]
        if isinstance(first, dict) and "content" in first:
            blocks = first["content"]
            return "\n".join(
                b.get("text", "") for b in blocks if b.get("type") == "text"
            )

    # fallback brut si le format change
    return json.dumps(response_body, indent=2)


def lambda_handler(event = -1, context = -1):
    user_query = "à quoi sert le bloc de donnée de paiement ? et quel est son rôle global au sein de la réforme de la facture électronique ?"
    #connection = psycopg2.connect(**db_config)
    #cursor = connection.cursor()

    try:
        connection = psycopg2.connect(**db_config)
        print("Connected to the database!")
        with connection.cursor() as cursor:
            best_chunks = get_k_best_chunks(cursor, user_query, 3)

            context_chunks = merge_overlapping_chunks(best_chunks)

            context_prompt = f"""
                You are a retrieval-augmented generative AI assistant.

                Before going to the core task that you need to achieve you must follow each of these guidelines before answering. 
                Guidelines:
                • Cite only the information contained in the context chunks.
                • The priority is to answer to the query. Don't over detail if you judge that it's not needed. 
                • If the answer is not found in the chunks, say “I don’t have enough information in the context to answer that exactly.“
                • Do not hallucinate or infer facts not present in the provided chunks.
                • Provide short relevant citations to the chunks if helpful.
                • For complex questions, include a brief step-by-step rationale before the final answer.
                • Answer the query as if the person doesn't know that you're a RAG AI. You must not use keywords like "context" or "chunk" or anything else.
                

                Below are the **most relevant context chunks** retrieved from the document corpus. These chunks have been selected by semantic similarity to the user’s query.
            
                ----

                CONTEXT CHUNKS:
                {context_chunks}

                ----

                TASK:
                Using the information in the provided context chunks *and nothing else*, answer the user’s question accurately and concisely.


                USER QUERY:
                {user_query}

                """
            llm_response = call_claude_bedrock(context_prompt)

            print(llm_response)
            return llm_response

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