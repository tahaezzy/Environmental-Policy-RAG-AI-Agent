## VERSION 4.3
import os
import json
import hashlib
import torch
from tqdm import tqdm
import re
import fitz
import pdfplumber
import pytesseract
import numpy as np
import latex2mathml.converter
import logging
from rank_bm25 import BM25Okapi
from pdf2image import convert_from_path
from PIL import Image
from chromadb import Client as ChromaClient
from sentence_transformers import SentenceTransformer
from ollama import Client as OllamaClient

# Validate Tesseract path
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust if different
if not os.path.exists(TESSERACT_PATH):
    logging.error(f"Tesseract not found at {TESSERACT_PATH}. Install or update path.")
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Temporary PATH fix for Windows-installed scripts
scripts_path = r'C:\Users\tahae\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts'
os.environ['PATH'] = scripts_path + os.pathsep + os.environ.get('PATH', '')

# Setup logging
logging.basicConfig(filename='greenpolicy.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
BATCH_SIZE = 32
MAX_TOKENS_PER_CHUNK = 500

# Model Initialization
llm = OllamaClient()
embed_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')
client = ChromaClient()
collection = client.get_or_create_collection("policies")
examplepolicies = "examplepolicy_docs.txt"

# Paths and Cache
doc_folder = "example_policy_docs"
cache_file = "embedding_cache.json"

if os.path.exists(cache_file):
    with open(cache_file, "r", encoding="utf-8") as f:
        embedding_cache = json.load(f)
else:
    embedding_cache = {}

# Helper Functions
def get_text_hash(text: str) -> str:
    """Compute SHA256 hash of text."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def get_embedding(text: str) -> list:
    """Retrieve or compute embedding for single text."""
    h = get_text_hash(text)
    if h in embedding_cache:
        return embedding_cache[h]
    emb = embed_model.encode(text).tolist()
    embedding_cache[h] = emb
    return emb

def save_cache() -> None:
    """Save embedding cache to JSON."""
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(embedding_cache, f)

def split_headers(text: str) -> list:
    """Split text into sections by headers (## or #)."""
    pattern = r'(?:^|\n)(#{1,6}\s+.+)'
    matches = list(re.finditer(pattern, text))
    sections = []
    if not matches:
        return [{"header": None, "text": text}]
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        header = m.group(1).strip()
        section_text = text[start:end].strip()
        sections.append({"header": header, "text": section_text})
    return sections

def chunk_text(text: str, max_tokens: int = MAX_TOKENS_PER_CHUNK) -> list:
    """Split text into token-limited chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_tokens):
        chunk = " ".join(words[i:i+max_tokens])
        if chunk.strip():  # Skip empty
            chunks.append(chunk)
    return chunks

def hybrid_chunking(text: str, max_tokens_per_chunk: int = MAX_TOKENS_PER_CHUNK) -> list:
    """Hybrid chunking: headers until max_tokens, then token-based."""
    sections = split_headers(text)
    chunks = []
    current_chunk_text = ""
    current_headers = []
    current_tokens = 0
    for sec in sections:
        sec_text = sec["text"].strip()
        if not sec_text:
            continue
        sec_tokens = len(sec_text.split())
        if current_tokens + sec_tokens <= max_tokens_per_chunk:
            current_chunk_text += sec_text + "\n\n"
            current_headers.append(sec["header"])
            current_tokens += sec_tokens
        else:
            if current_chunk_text:
                chunks.append({"text": current_chunk_text.strip(), "headers": current_headers})
            if sec_tokens > max_tokens_per_chunk:
                token_chunks = chunk_text(sec_text, max_tokens=max_tokens_per_chunk)
                for t_chunk in token_chunks:
                    chunks.append({"text": t_chunk, "headers": [sec["header"]]})
                current_chunk_text = ""
                current_headers = []
                current_tokens = 0
            else:
                current_chunk_text = sec_text
                current_headers = [sec["header"]]
                current_tokens = sec_tokens
    if current_chunk_text:
        chunks.append({"text": current_chunk_text.strip(), "headers": current_headers})
    return chunks

def get_batch_embeddings(texts: list[str], batch_size: int = BATCH_SIZE) -> list[list[float]]:
    """Embed texts in batches with caching."""
    embeddings = []
    try:
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_hashes = [get_text_hash(t) for t in batch_texts]
            batch_embs = [None] * len(batch_texts)
            uncached_indices = []
            for j, h in enumerate(batch_hashes):
                if h in embedding_cache:
                    batch_embs[j] = embedding_cache[h]
                else:
                    uncached_indices.append(j)
            if uncached_indices:
                uncached_texts = [batch_texts[j] for j in uncached_indices]
                new_embs = embed_model.encode(uncached_texts, batch_size=len(uncached_texts), show_progress_bar=True).tolist()
                for k, emb in enumerate(new_embs):
                    h = batch_hashes[uncached_indices[k]]
                    embedding_cache[h] = emb
                    batch_embs[uncached_indices[k]] = emb
            embeddings.extend(batch_embs)
        for emb in embeddings:
            if not emb or len(emb) == 0:
                logging.error(f"Invalid embedding detected")
        save_cache()
    except Exception as e:
        logging.error(f"Batch embedding error: {e}")
        return [[]] * len(texts)
    return embeddings

def clean_text(text: str) -> str:
    """Remove extra whitespace, headers/footers."""
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'^(Header|Footer|Page \d+).*', '', text, flags=re.M)
    return text

def handle_math(text: str) -> str:
    """Convert LaTeX math to plain text."""
    try:
        mathml = latex2mathml.converter.convert(text)
        text = re.sub(r'<[^>]+>', '', mathml)
        return text
    except Exception as e:
        logging.error(f"Math handling error: {e}")
        text = re.sub(r'\$\$(.*?)\$\$', r' [Math: \1] ', text)
        return text

def extract_pdf_text(file_path: str, password: str = None) -> str:
    """Extract text from PDF with tables and OCR."""
    text = ""
    try:
        logging.info(f"Processing PDF: {file_path}")
        doc = fitz.open(file_path)
        if doc.needs_pass:
            if password:
                doc.authenticate(password)
            else:
                logging.warning(f"Skipping password-protected PDF: {file_path}")
                return ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text("text").strip()
            with pdfplumber.open(file_path, password=password) as pdf:
                plumb_page = pdf.pages[page_num]
                tables = plumb_page.extract_tables()
                if tables:
                    for table in tables:
                        md_table = "\n".join(["| " + " | ".join(map(str, row)) + " |" for row in table])
                        page_text += "\n\nTable:\n" + md_table + "\n\n"
            page_text = handle_math(page_text)
            if len(page_text) < 50:
                logging.info(f"Running OCR on page {page_num + 1}")
                images = convert_from_path(file_path, first_page=page_num+1, last_page=page_num+1)
                for img in images:
                    ocr_text = pytesseract.image_to_string(img)
                    logging.info(f"OCR output: {ocr_text[:100]}...")
                    page_text += ocr_text + "\n\n"
            text += page_text + "\n\n--- Page Break ---\n\n"
        text = clean_text(text)
    except Exception as e:
        logging.error(f"Error processing PDF {file_path}: {e}")
    return text

all_tasks = []
chunk_texts = []
for file_name in os.listdir(doc_folder):
    file_path = os.path.join(doc_folder, file_name)
    if file_name.endswith((".txt", ".pdf")):
        if file_name.endswith(".txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception as e:
                logging.error(f"Error reading TXT {file_path}: {e}")
                continue
        elif file_name.endswith(".pdf"):
            text = extract_pdf_text(file_path)
        if text.strip():
            chunks = hybrid_chunking(text)
            for idx, chunk in enumerate(chunks):
                if chunk["text"].strip():
                    all_tasks.append((chunk["text"], chunk["headers"], file_name, idx))
                    chunk_texts.append(chunk["text"])
if not all_tasks:
    logging.warning("No valid documents found.")
    print("No valid documents found.")
else:
    print(f"Processing {len(chunk_texts)} chunks with batch embeddings...")
    batch_embs = get_batch_embeddings(chunk_texts, batch_size=BATCH_SIZE)
    results = []
    for i, (chunk_text, headers, file_name, idx) in enumerate(all_tasks):
        doc_id = f"{file_name}_{idx}"
        emb = batch_embs[i]
        if not emb or len(emb) == 0:
            logging.error(f"Skipping chunk {doc_id}: Invalid embedding")
            continue
        results.append((doc_id, chunk_text, emb, file_name))
    for doc_id, chunk_text, emb, file_name in results:
        try:
            collection.add(
                documents=[chunk_text],
                metadatas=[{"source": file_name, "headers": json.dumps(headers)}],
                ids=[doc_id],
                embeddings=[emb]
            )
        except Exception as e:
            logging.error(f"Error adding chunk {doc_id}: {e}")
    save_cache()
    print(f"Policy documents ingested successfully! Collection size: {collection.count()}")

def add_document(text: str, doc_id: str, source=examplepolicies) -> None:
    """Add a single document to Chroma."""
    try:
        emb = get_embedding(text)
        collection.add(
            documents=[text],
            metadatas=[{"source": source}],
            ids=[str(doc_id)],
            embeddings=[emb]
        )
        save_cache()
    except Exception as e:
        logging.error(f"Error adding document {doc_id}: {e}")

def ask_rag(query: str, model_name="qwen2.5:0.5b", top_k=4):
    """Query RAG with hybrid retrieval."""

    ##############################################################################################
    # Default model priority list (Keep Updating with latest pullable free models)
    ## Build cycle system to test different models. 
    #if model_name is None:
    #    models_to_try = ["llama3.2", "llama3.1", "llama3", "phi3", "mistral", "codellama"]
    #else:
    #    models_to_try = [model_name]
    ##############################################################################################

    try:
        if collection.count() == 0:
            logging.error("Chroma collection is empty")
            return None, None
        
        query_emb = embed_model.encode(query)
        
        # Fix: Include embeddings in the query results
        results = collection.query(
            query_embeddings=[query_emb], 
            n_results=top_k * 2,
            include=['documents', 'metadatas', 'embeddings', 'distances']  # Explicitly request embeddings
        )
        
        logging.info(f"Query results: {results}")
        
        if not results['documents'] or not results['documents'][0]:
            logging.error("No documents retrieved from Chroma")
            return None, None
        
        # Fix: Check if embeddings exist before using them
        if not results['embeddings'] or len(results['embeddings']) == 0 or len(results['embeddings'][0]) == 0:
            logging.error("No embeddings retrieved from Chroma")
            # Fallback to just using BM25 scoring
            tokenized_docs = [doc.split() for doc in results['documents'][0]]
            bm25 = BM25Okapi(tokenized_docs)
            bm25_scores = bm25.get_scores(query.split())
            top_indices = np.argsort(bm25_scores)[::-1][:top_k]
        else:
            # Use hybrid scoring as intended
            retrieved_embs = results['embeddings'][0]  # Already a numpy array
            if not isinstance(retrieved_embs, np.ndarray):
                retrieved_embs = np.array(retrieved_embs)
                
            dense_scores = np.dot(retrieved_embs, query_emb) / (np.linalg.norm(retrieved_embs, axis=1) * np.linalg.norm(query_emb))
            
            tokenized_docs = [doc.split() for doc in results['documents'][0]]
            bm25 = BM25Okapi(tokenized_docs)
            bm25_scores = bm25.get_scores(query.split())
            
            combined_scores = 0.7 * dense_scores + 0.3 * bm25_scores
            top_indices = np.argsort(combined_scores)[::-1][:top_k]
        
        context = " ".join([results['documents'][0][i] for i in top_indices])
        metadata = [results['metadatas'][0][i] for i in top_indices] if 'metadatas' in results and results['metadatas'] else None
        
        # Load model if not already loaded
        if not hasattr(ask_rag, "loaded"):
            llm.chat(model=model_name, messages=[{"role": "user", "content": "Test."}])
            ask_rag.loaded = True
        
        response = llm.chat(model=model_name, messages=[
            {"role": "system", "content": "You are an expert environmental policy assistant. Assist using 2025 data."},
            {"role": "user", "content": f"Answer based on context: {context}\nQuestion: {query}"}
        ])
        
        return response.message, metadata
        
    except Exception as e:
        logging.error(f"Error in RAG query: {e}")
        return None, None

def check_compliance(project_desc: str, query: str = "Does this project violate any rules?"):
    """Compliance checker with RAG."""
    full_query = f"Project: {project_desc}. {query}"
    return ask_rag(full_query)

# REPL
print("GreenPolicyAI REPL. Type 'exit' to quit.")
print("For compliance check, prefix with 'compliance:' (e.g., 'compliance: Build factory near wetland').")
while True:
    user_input = input("Question: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    try:
        if user_input.lower().startswith("compliance:"):
            project_desc = user_input[11:].strip()
            answer, meta = check_compliance(project_desc)
        else:
            answer, meta = ask_rag(user_input)
        if answer:
            print("Answer:", answer.content)
            print("Sources:", meta)
        else:
            print("Error occurred—check greenpolicy.log.")
    except Exception as e:
        logging.error(f"REPL error: {e}")
        print("Error—check log.")