import json
import boto3
import os
import psycopg2
import pymupdf4llm
import pymupdf.layout
import hashlib
import re
import fitz  # pymupdf
from pathlib import Path
from PIL import Image
import io
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

def extract_images_from_pdf(pdf_path: str, output_folder: str = "extracted_images"):
    pdf_path = Path(pdf_path).resolve()
    out_dir = pdf_path.parent / "extracted_images"
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    for page_index in range(len(doc)):
        page = doc[page_index]
        images = page.get_images(full=True)
        text = page.get_text()

        print(f"Page {page_index + 1}: {len(images)} image(s) trouvée(s)")
        for img_index, img in enumerate(images):
            xref = img[0]
            smask_xref = img[1]
            width, height = img[2], img[3]

            pix = fitz.Pixmap(doc,xref)

            if smask_xref is not None:
                mask = fitz.Pixmap(doc,smask_xref)

                if pix.alpha :
                    pix = fitz.Pixmap(pix,0)

                pix = fitz.Pixmap(pix,mask)

            if pix.n > 4 :
                pix = fitz.Pixmap(fitz.csRGB, pix)

            img_path = out_dir / f"page{page_index + 1}_img{img_index}.png"
            pix.save(img_path)
            pix = None

    doc.close()
    print("Extraction terminée ✅")

def extract_text(pdf_file):
    md_read = pymupdf4llm.LlamaMarkdownReader()
    data = md_read.load_data(pdf_file)
    Big_text = ""
    for item in data:
        print(item.text)
        Big_text += item.text
        print("\n")
        print("-------------------------------------------------------------------------")
        print("\n")
    return

def generate_doc_id(pdf_path):
    return hashlib.md5(filename.encode()).hexdigest()

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
    pdf_path = "Documents/EXTRAIT_RAG.pdf"
    doc_id = generate_doc_id(pdf_path)

    text = extract_text(pdf_path)
    return


extract_images_from_pdf("Documents/EXTRAIT_RAG.pdf")
#extract_text("Documents/EXTRAIT_RAG.pdf")