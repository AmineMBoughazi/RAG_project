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

def extract_text_with_images(pdf_path):
    # ouvrir le PDF
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
                block_text = ""

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

                ################################################ TRAITER L'IMAGE DIRECTEMENT ICI ?
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

                placeholder = f"[IMAGE page={page_index+1} block={block_index}]"
                big_text += "\n" + placeholder + "\n"
                images_info.append({
                    "page": page_index + 1,
                    "block_index": block_index,
                    "bbox": block.get("bbox"),
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
    for ii in images_info:
        print(ii)
    print('------------------------------------')
    print(big_text)
    return big_text, images_info

def distance_scoring(bboxt : list, bboxi : list) -> tuple:
    # horizontal overlap
    dy_below = -1 * (bboxt[1] - bboxi[1])
    dy_above = -1 * (bboxt[3] - bboxi[3])
    height_image = bboxi[3] - bboxi[1]
    height_threshold = height_image / 2
    return (dy_below,dy_above,height_threshold)


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

def lambda_handler(event = -1, context = -1):
    pdf_path = "Documents/EXTRAIT_RAG.pdf"
    doc_id = generate_doc_id(pdf_path)

    text = extract_text_with_images(pdf_path)
    return


#extract_images_from_pdf("Documents/EXTRAIT_RAG.pdf")
extract_text_with_images("Documents/EXTRAIT_RAG.pdf")