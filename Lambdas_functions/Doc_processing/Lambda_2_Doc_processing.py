import json
import boto3
import os
import psycopg2
import hashlib
from psycopg2.extras import execute_values
import re
import fitz  # pymupdf
from pathlib import Path
from typing import List
from dotenv import load_dotenv
import logging
import urllib.parse

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
s3_client = boto3.client('s3')


def create_tables(cur) -> None:

    cur.execute("""
    CREATE TABLE IF NOT EXISTS documents_table (
        document_id UUID PRIMARY KEY NOT NULL,
        source_path TEXT NOT NULL,
        title TEXT NOT NULL,
        type TEXT NOT NULL,
        last_modified TIMESTAMP NOT NULL
        )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS images_table (
        image_id UUID PRIMARY KEY NOT NULL,
        document_id UUID NOT NULL REFERENCES documents_table(document_id) ON DELETE CASCADE,
        page_number int NOT NULL,
        image_url TEXT NOT NULL,
        caption TEXT NOT NULL
        )        
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS chunks_table (
        chunk_id UUID PRIMARY KEY NOT NULL,
        document_id UUID NOT NULL REFERENCES documents_table(document_id) ON DELETE CASCADE,
        text TEXT NOT NULL,
        embeddings VECTOR NOT NULL,
        chunk_index INTEGER NOT NULL,
        start_offset INTEGER NOT NULL,
        end_offset INTEGER NOT NULL,
        page_num INTEGER NOT NULL,
        image_id UUID REFERENCES images_table(image_id)
        )        
    """)




    return

def extract_text_with_images(file_name,pdf_bytes):
    pdf_path = f"/tmp/{file_name}"
    with open(pdf_path, "wb") as pdf_file:
        pdf_file.write(pdf_bytes)

    doc = fitz.open(pdf_path)
    pdf_path = Path(pdf_path).resolve()
    out_dir = pdf_path.parent / "extracted_images"
    out_dir.mkdir(parents=True, exist_ok=True)
    big_text = ""
    images_info = []

    for page_index, page in enumerate(doc):
        page_dict = page.get_text("dict")
        text_bbox = []
        image = False

        page_images = page.get_images(full=True)  # liste de tuples
        image_iter = iter(page_images)

        for block_index, block in enumerate(page_dict["blocks"]):
            if block["type"] == 0:  # texte
                block_text = f"###PAGE_{page_index}####"

                for line in block["lines"]:
                    line_text = ""

                    for span in line["spans"]:
                        text = span["text"]

                        # Ajouter un espace si nécessaire (sinon les mots collent)
                        if (
                                len(line_text) > 0
                                and not line_text.endswith(" ")
                                and not text.startswith(" ")
                        ):
                            line_text += " "

                        line_text += text

                    block_text += line_text + "\n"  # fin de ligne
                    text_bbox.append([block_text,block.get("bbox")])
                big_text += block_text + "\n"  # fin de paragraphe (block)
            elif block["type"] == 1:  # image
                image = True

                ################################################ TRAITER L'IMAGE ICI - SAUVEGARDE SUR LE DISQUE MAIS A TERME SAUVEGARDERA SUR S3
                try:
                    # On prend la prochaine image de la page
                    img = next(image_iter)
                    xref = img[0]  # premier élément = xref (int)
                    smask_xref = img[1]
                    pix = fitz.Pixmap(doc, xref)

                    if smask_xref is not None:
                        mask = fitz.Pixmap(doc, smask_xref)
                        if pix.alpha:
                            pix = fitz.Pixmap(pix, 0)

                        pix = fitz.Pixmap(pix, mask)

                    if pix.n > 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)

                    image_info = doc.extract_image(xref)
                    image_id = generate_id(str(xref))
                    img_bytes = image_info["image"]
                    img_ext = image_info.get("ext", "png")  # png par défaut
                    img_path = out_dir / f"page{page_index + 1}_img_{image_id}.{img_ext}"
                    pix.save(img_path)
                    pix = None

                except StopIteration:
                    print(f"[WARN] Plus d'images dans get_images() pour page {page_index + 1}, block {block_index}")
                ################################################

                placeholder = f"[IMAGE page={page_index+1} image_id={image_id}]"
                big_text += "\n" + placeholder + "\n"
                images_info.append({
                    "image_id": image_id,
                    "page": page_index + 1,
                    "block_index": block_index,
                    "bbox": block.get("bbox"),
                    "url" : "",
                    "caption":"",
                    "caption_treated" : False
                })
        candidates = []
        if image :
            for k,image_bbox in enumerate(images_info):
                if image_bbox["caption_treated"]:continue
                bboxi = image_bbox["bbox"]
                for bboxt in text_bbox :
                    score = distance_scoring(bboxt[1],bboxi)
                    (dy_below, dy_above, height_threshold) = score
                    if min(dy_below,dy_above) > height_threshold:
                        continue
                    else :
                        candidates.append([bboxt[0],min(abs(dy_below),abs(dy_above))])
                if len(candidates) > 0:
                    candidates = [c for c in candidates if c[0] != " \n"]
                    candidates = sorted(candidates, key=lambda c: c[1])
                    image_bbox["caption"] = "\n".join(c[0] for c in candidates[:5])
                    image_bbox["caption_treated"] = True
                    candidates = []
    doc.close()
    return big_text, images_info

def distance_scoring(bboxt : list, bboxi : list) -> tuple:
    # horizontal overlap
    dy_below = -1 * (bboxt[1] - bboxi[1])
    dy_above = -1 * (bboxt[3] - bboxi[3])
    height_image = bboxi[3] - bboxi[1]
    height_threshold = height_image / 2
    return (dy_below,dy_above,height_threshold)

def remove_summary(text,count_balises) :
    def search_word_first_apparition(lines):
        for line_index,text in enumerate(lines):
            for text_index,word  in enumerate(text):
                if "#SOMMAIRE" in word:
                    return text_index, line_index
    lines = text.splitlines()
    tokens = [text.strip().split() for text in lines]
    text_index,lines_index = search_word_first_apparition(tokens)
    n = 0
    lines_to_pop = list(range(0,lines_index))
    while lines_index < len(lines) :
        text_index = 0
        lines_to_pop.append(lines_index)
        while text_index < len(tokens[lines_index]) :
            if "#SOMMAIRE" in tokens[lines_index][text_index]:
                n += 1
            if n == count_balises:
                break
            text_index += 1
        if n == count_balises: break
        lines_index += 1
    for idx in sorted(lines_to_pop, reverse=True):
        lines.pop(idx)
    text = "\n".join(lines)

    return text

def text_cleaning(text: str) -> str:
    cleaned_text_dots, count_balises = re.subn(r".*\.{2,}.* ?\d+\n?", "#SOMMAIRE", text)
    cleaned_text_summary = remove_summary(cleaned_text_dots,count_balises)

    return cleaned_text_summary

def sliding_window_chunking(text : str,doc_id : str,chunk_window : int = 350,step : int = 100):
    chunks = []

    clean_text = text_cleaning(text)
    page_pattern = re.compile(r'###PAGE_(.*?)####', re.DOTALL) # ####PAGE_{page_index}###
    image_pattern = re.compile(r'\[IMAGE page=\d+ image_id=(.*?)\]', re.DOTALL) # [IMAGE page={page_index+1} path={img_path}]
    tokens = clean_text.strip().split()
    i = 0
    page_index = 1

    while i < len(tokens):
        start_offset = i
        end_offset = i + chunk_window
        window_text = tokens[i:i+chunk_window]

        chunk = " ".join(window_text)

        page_match = page_pattern.search(chunk)
        if page_match :
            page_index = int(page_match.group(1))+1

        doc_page = page_index

        image_match = image_pattern.search(chunk)
        image_id = None
        if image_match:
            image_id = image_match.group(1)

        chunk_id = generate_id(chunk)

        chunk = page_pattern.sub("", chunk)
        chunk = image_pattern.sub("", chunk)

        chunk = {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "doc_page": doc_page,
            "text": chunk.strip(),
            "embedding" :[],
            "chunk_index" : 0,
            "start_offset" : start_offset,
            "end_offset" : end_offset,
            "image_id" : image_id,
        }
        chunks.append(chunk)
        i += step
    return chunks

def generate_id(character):
    return hashlib.md5(character.encode()).hexdigest()

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

def embed_chunks(chunks):
    for chunk_index,chunk in enumerate(chunks):
        chunk["embedding"] = generate_embedding(chunk["text"])
        chunk["chunk_index"] = chunk_index
    return chunks

def download_file_from_s3(bucket_name, file_key):
    """
    Downloads a file from S3 and extracts its content as bytes.
    """
    try:
        logger.info(f"Now downloading file: {file_key}")

        # Extract the file name from the file_key
        file_name = os.path.basename(file_key)
        logger.info(f"File name: {file_name}")

        # Extract the directory path from the file_key
        directory_path = os.path.dirname(file_key)
        logger.info(f"Directory path: {directory_path}")

        local_pdf_path = f"/tmp/{file_name}"
        s3_client.download_file(bucket_name, file_key, local_pdf_path)

        with open(local_pdf_path, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()

        response = s3_client.head_object(Bucket=bucket_name, Key=file_key)
        last_modified = response["LastModified"]
        doc_type = response["ContentType"].split("/")[1]
        logger.info(f"Extracted file {file_key} from S3")

        return pdf_bytes, file_name, directory_path, last_modified, doc_type

    except Exception as e:
        logger.error(f"Error fetching {file_key} from {bucket_name}: {e}")
        raise

def documents_table_insertion(cursor,document_row):
    ######## Documents_table population
    insert_query = """
                   INSERT INTO documents_table (document_id, \
                                                source_path, \
                                                title, \
                                                type, \
                                                last_modified)
                   VALUES (%s, %s, %s, %s, %s) ON CONFLICT (document_id) DO NOTHING; \
                   """
    cursor.execute(insert_query, document_row)
    print("Document inserted to documents_table")

def images_table_insertion(cursor,images,doc_id):
    insert_query = """
                   INSERT INTO images_table (image_id, \
                                             document_id, \
                                             page_number, \
                                             image_url, \
                                             caption)
                   VALUES %s ON CONFLICT (image_id) DO NOTHING; \
                   """

    images_rows = [
        (
            i["image_id"],
            doc_id,
            i["page"],
            i["url"],
            i["caption"],
        )
        for i in images
    ]
    execute_values(cursor, insert_query, images_rows)
    print("Images inserted to images_table")

def chunks_table_insertion(cursor,chunks):
    chunk_rows = [
        (
            c["chunk_id"],
            c["doc_id"],
            c["text"],
            c["embedding"],
            c["chunk_index"],
            c["start_offset"],
            c["end_offset"],
            c["doc_page"],
            c["image_id"],
        )
        for c in chunks
    ]
    insert_query = """
                   INSERT INTO chunks_table (chunk_id, \
                                             document_id, \
                                             text, \
                                             embeddings, \
                                             chunk_index, \
                                             start_offset, \
                                             end_offset, \
                                             page_num, \
                                             image_id)
                   VALUES %s ON CONFLICT (chunk_id) DO NOTHING; \
                   """
    execute_values(cursor, insert_query, chunk_rows)
    print("Chunks inserted to chunks_table")

def lambda_handler(event, context = -1):

    file_key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
    bucket_name = event['Records'][0]['s3']['bucket']['name']
    logger.info(f"Processing file {file_key}")

    # Downloading PDF file from s3
    pdf_bytes, file_name, directory_path, last_modified, doc_type = download_file_from_s3(bucket_name, file_key)
    print("File name:", file_name)

    #doc_id creation : same name = same id
    doc_id = generate_id(file_name)

    #Text & Images Extraction
    text, images = extract_text_with_images(file_name,pdf_bytes)

    #Text Chunking
    chunks = sliding_window_chunking(text,doc_id)

    #Chunk Embedding
    embedded_chunks = embed_chunks(chunks)
    print("Embedding done")

    try:
        connection = psycopg2.connect(**db_config)
        print("Connected to the database!")
        with connection.cursor() as cursor:

            create_tables(cursor)

            ######## Documents_table population
            document_row = (doc_id, directory_path, file_name, doc_type, last_modified)
            documents_table_insertion(cursor, document_row)
            ######## Images_table population
            images_table_insertion(cursor, images, doc_id)
            ######## chunks_table population
            chunks_table_insertion(cursor, embedded_chunks)

        # Valider les changements
        connection.commit()
        print("Data inserted successfully!")
        return {
            'statusCode': 200,
            'body': json.dumps('Data inserted successfully!')
        }
    except Exception as e:
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

with open("event.txt", "r", encoding="utf-8") as f:
    event = json.load(f)

lambda_handler(event)