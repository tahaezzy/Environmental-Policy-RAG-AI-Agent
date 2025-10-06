## BETA VERSION 5.0.1
# Changes:
'''
Addressed critical issues, efficency, quick fixes, and overall robustenss (Includes but not limited to adding atomic cache saving). 
'''

## DIRECTORY AND PATH TEST______________________
import os
print("CWD:", os.getcwd())
print("Script dir:", os.path.dirname(__file__))
print("Looks like these folders exist?:")
for name in ["Project Guidelines Folder", "Regulations Folder", "Knowledge Base Folder"]:
    print(name, "->", os.path.exists(name))
##______________________________________________

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
#from chromadb import Client as ChromaClient
from sentence_transformers import SentenceTransformer, CrossEncoder 
from ollama import Client as OllamaClient
import networkx as nx  # For graph KB
import spacy  # For rule-based extraction
from pathlib import Path  # For cache path
import psutil
import pickle
import asyncio
import gc
import time
import sqlite3
from typing import List, Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('GreenPolicyAI.log', mode='a'),
        logging.StreamHandler()
    ]
)

# Required @DEPENDENCIES. 
try:
    import faiss ## FAISS
    FAISS_AVAILABLE = True
    logging.info("FAISS loaded successfully")
except ImportError:
    logging.warning("FAISS not available - install with: pip install faiss-cpu")
    FAISS_AVAILABLE = False
    faiss = None

try:
    import redis  ## REDIS 
    REDIS_AVAILABLE = True
    logging.info("Redis client loaded successfully")
except ImportError:
    logging.warning("Redis not available - install with: pip install redis")
    REDIS_AVAILABLE = False
    redis = None

try:
    import networkx as nx ## NETWORKX
    NETWORKX_AVAILABLE = True
    logging.info("NetworkX loaded successfully")
except ImportError:
    logging.warning("NetworkX not available - graph features disabled")
    NETWORKX_AVAILABLE = False
    nx = None

try:
    import spacy  ## SPACY
    SPACY_AVAILABLE = True
    logging.info("spaCy loaded successfully")
except ImportError:
    logging.warning("spaCy not available - run: python -m spacy download en_core_web_sm")
    SPACY_AVAILABLE = False
    spacy = None

try:
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2') ## CROSS ENCODER
    logging.info("Cross-encoder loaded successfully")
except Exception as e:
    logging.error(f"Failed to load cross-encoder: {e}")
    cross_encoder = None

# Tesseract validation
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
if not os.path.exists(TESSERACT_PATH):
    logging.error(f"Tesseract not found at {TESSERACT_PATH}")
else:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    logging.info("Tesseract configured successfully")

# Model Initialization
try:
    llm = OllamaClient()
    logging.info("Ollama client initialized")
except Exception as e:
    logging.error(f"Failed to initialize Ollama client: {e}")
    llm = None

try:
    embed_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Embedding model loaded on {'CUDA' if torch.cuda.is_available() else 'CPU'}")
except Exception as e:
    logging.error(f"Failed to load embedding model: {e}")
    embed_model = None

# Memory-dependent constants
def get_system_constants():
    memory_gb = psutil.virtual_memory().total / (1024**3)
    if memory_gb < 8:
        return {'BATCH_SIZE': 16, 'MAX_TOKENS': 256, 'CONTEXT_LEN': 8192, 'TOP_K': 3}
    elif memory_gb < 16:
        return {'BATCH_SIZE': 32, 'MAX_TOKENS': 512, 'CONTEXT_LEN': 16384, 'TOP_K': 5}
    else:
        return {'BATCH_SIZE': 64, 'MAX_TOKENS': 1024, 'CONTEXT_LEN': 32768, 'TOP_K': 5}

# @ Constants
constants = get_system_constants()
LLM_MODEL_NAME = "qwen2.5:0.5b"
BATCH_SIZE = constants['BATCH_SIZE']
MAX_TOKENS_PER_CHUNK = constants['MAX_TOKENS']
CONTEXT_WINDOW_LENGTH = constants['CONTEXT_LEN']
TOP_K_DEFAULT = constants['TOP_K']
THINK_MODE = False

# User Folders
BASE_DIR = Path(__file__).parent.resolve()
project_guidelines_folder = str(BASE_DIR / "Project Guidelines Folder")
compliance_regulations_folder = str(BASE_DIR / "Regulations Folder")
knowledge_base_folder = str(BASE_DIR / "Knowledge Base Folder")
# Debug print/log to confirm paths exist
logging.info(f"project_guidelines_folder = {project_guidelines_folder} (exists={os.path.exists(project_guidelines_folder)})")
logging.info(f"compliance_regulations_folder = {compliance_regulations_folder} (exists={os.path.exists(compliance_regulations_folder)})")
logging.info(f"knowledge_base_folder = {knowledge_base_folder} (exists={os.path.exists(knowledge_base_folder)})")

# @ Paths and Cache
cache_file = "embedding_cache.json"
DB_PATH = "greenpolicy.db"
FAISS_INDEX_PATH = "faiss_index.bin"
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
#GRAPH_CACHE = "reg_graph.pkl"  # Graph persistence
GRAPH_CACHE_JSON = "reg_graph.json"
GRAPH_EMB_CACHE = "reg_graph_embeddings.npy"

# Check for exisitng embeddings cache file
if os.path.exists(cache_file):
    with open(cache_file, "r", encoding="utf-8") as f:
        embedding_cache = json.load(f)
else:
    embedding_cache = {}

# Load spaCy for rule-based extraction (en_core_web_sm for NER)
nlp = None
if SPACY_AVAILABLE and spacy is not None:
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        logging.warning("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm. Falling back to blank English pipeline.")
        nlp = spacy.blank("en")  # No NER, but keeps sentence segmentation if we add sentencizer
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
else:
    logging.warning("spaCy not available; NLP features will use simple regex fallback.")

# Globals for graph (loaded once)
reg_graph = None
faiss_index = None
node_texts = None
embedder = embed_model  # Reuse existing embedder

logging.info(f"System constants: {constants}")

class RedisCache:
    """Redis-based caching system for embeddings and responses"""
    
    def __init__(self):
        self.client = None
        self.enabled = False
        if REDIS_AVAILABLE:
            try:
                self.client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
                self.client.ping()  # Test connection
                self.enabled = True
                logging.info("Redis cache enabled")
            except Exception as e:
                logging.warning(f"Redis not available: {e}. Using fallback caching.")
                self.enabled = False
    
    def get_embedding(self, text_hash: str):
        """Get embedding from cache"""
        if not self.enabled:
            return None
        try:
            cached = self.client.get(f"emb:{text_hash}")
            return json.loads(cached) if cached else None
        except Exception as e:
            logging.warning(f"Redis get error: {e}")
            return None
    
    def set_embedding(self, text_hash: str, embedding, expire_hours: int = 48):
        """Cache embedding with expiration"""
        if not self.enabled:
            return
        try:
            self.client.setex(f"emb:{text_hash}", expire_hours * 3600, json.dumps(embedding))
        except Exception as e:
            logging.warning(f"Redis set error: {e}")
    
    def get_response(self, query_hash: str):
        """Get cached LLM response"""
        if not self.enabled:
            return None
        try:
            return self.client.get(f"resp:{query_hash}")
        except Exception as e:
            logging.warning(f"Redis response get error: {e}")
            return None
    
    def set_response(self, query_hash: str, response: str, expire_hours: int = 2):
        """Cache LLM response with shorter expiration"""
        if not self.enabled:
            return
        try:
            self.client.setex(f"resp:{query_hash}", expire_hours * 3600, response)
        except Exception as e:
            logging.warning(f"Redis response set error: {e}")
class DatabaseManager:
    """SQLite database manager for document storage and metadata"""
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.faiss_index = None
        self.faiss_index_path = FAISS_INDEX_PATH
        self._init_database()
        self._load_faiss_index()
    
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            # Execute each statement separately
            conn.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT UNIQUE NOT NULL,
                    content TEXT NOT NULL,
                    source_file TEXT NOT NULL,
                    headers TEXT,
                    chunk_index INTEGER,
                    token_count INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT NOT NULL,
                    embedding_vector BLOB NOT NULL,
                    faiss_index INTEGER,
                    FOREIGN KEY (doc_id) REFERENCES documents (doc_id)
                )
            ''')
            
            # Execute indexes separately
            conn.execute('CREATE INDEX IF NOT EXISTS idx_doc_id ON documents(doc_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_source_file ON documents(source_file)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_faiss_index ON embeddings(faiss_index)')
        
        logging.info("Database initialized successfully")
    
    def _load_faiss_index(self):
        """Load or create FAISS index"""
        if not FAISS_AVAILABLE:
            logging.warning("FAISS not available")
            return
        
        if os.path.exists(self.faiss_index_path):
            try:
                self.faiss_index = faiss.read_index(self.faiss_index_path)
                logging.info(f"Loaded FAISS index with {self.faiss_index.ntotal} vectors")
            except Exception as e:
                logging.error(f"Failed to load FAISS index: {e}")
                self._create_faiss_index()
        else:
            self._create_faiss_index()
    
    def _create_faiss_index(self):
        """Create new FAISS index"""
        if not FAISS_AVAILABLE or not embed_model:
            return
        
        try:
            # Create index for 384-dimensional vectors (all-MiniLM-L6-v2)
            dimension = 384
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            logging.info("Created new FAISS index")
        except Exception as e:
            logging.error(f"Failed to create FAISS index: {e}")
    
    def add_document(self, doc_id: str, content: str, source_file: str, 
                    headers, chunk_index: int, embedding):
        """Add document and embedding to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Add document
                conn.execute('''
                    INSERT OR REPLACE INTO documents 
                    (doc_id, content, source_file, headers, chunk_index, token_count)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (doc_id, content, source_file, json.dumps(headers), 
                     chunk_index, len(content.split())))
                
                # Add embedding to FAISS
                if self.faiss_index is not None and embedding:
                    embedding_array = np.array([embedding]).astype('float32')
                    faiss.normalize_L2(embedding_array)  # Normalize for cosine similarity
                    faiss_index = self.faiss_index.ntotal
                    self.faiss_index.add(embedding_array)
                    
                    # Store embedding metadata in SQLite
                    embedding_blob = pickle.dumps(embedding)
                    conn.execute('''
                        INSERT OR REPLACE INTO embeddings 
                        (doc_id, embedding_vector, faiss_index)
                        VALUES (?, ?, ?)
                    ''', (doc_id, embedding_blob, faiss_index))
                
                logging.debug(f"Added document: {doc_id}")
                
        except Exception as e:
            logging.error(f"Failed to add document {doc_id}: {e}")
    
    def search_similar(self, query_embedding, top_k = TOP_K_DEFAULT):
        """Search for similar documents using FAISS"""
        if not self.faiss_index or not query_embedding:
            return []
        
        try:
            query_array = np.array([query_embedding]).astype('float32')
            faiss.normalize_L2(query_array)
            
            scores, indices = self.faiss_index.search(query_array, top_k)
            
            results = []
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                    if idx == -1:  # Invalid index
                        continue
                    
                    cursor = conn.execute('''
                        SELECT d.*, e.faiss_index 
                        FROM documents d 
                        JOIN embeddings e ON d.doc_id = e.doc_id 
                        WHERE e.faiss_index = ?
                    ''', (int(idx),))
                    
                    row = cursor.fetchone()
                    if row:
                        results.append({
                            'doc_id': row['doc_id'],
                            'content': row['content'],
                            'source_file': row['source_file'],
                            'headers': json.loads(row['headers']) if row['headers'] else [],
                            'similarity_score': float(score),
                            'metadata': {
                                'chunk_index': row['chunk_index'],
                                'token_count': row['token_count'],
                                'created_at': row['created_at']
                            }
                        })
            
            return results
            
        except Exception as e:
            logging.error(f"FAISS search failed: {e}")
            return []
    
    def get_document_count(self) -> int:
        """Get total number of documents"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('SELECT COUNT(*) FROM documents')
                return cursor.fetchone()[0]
        except Exception as e:
            logging.error(f"Failed to get document count: {e}")
            return 0
    
    def save_faiss_index(self):
        """Save FAISS index to disk"""
        if self.faiss_index:
            try:
                faiss.write_index(self.faiss_index, self.faiss_index_path)
                logging.info("FAISS index saved successfully")
            except Exception as e:
                logging.error(f"Failed to save FAISS index: {e}")

class StreamingResponse:
    """Handles streaming responses from Ollama"""
    
    @staticmethod
    def stream_ollama_response(model_name: str, messages, options):
        """Stream response from Ollama word by word"""
        if not llm:
            yield "Error: Ollama client not available"
            return
        
        try:
            # Note: This is a simplified streaming implementation
            # Real streaming would require Ollama's streaming API
            response = llm.chat(model=model_name, messages=messages, options=options or {})
            
            if hasattr(response, 'message') and hasattr(response.message, 'content'):
                content = response.message.content
            elif isinstance(response, dict) and 'message' in response:
                content = response['message'].get('content', str(response['message']))
            else:
                content = str(response)
            
            # Simulate streaming by yielding words
            words = content.split()
            for i, word in enumerate(words):
                if i == 0:
                    yield word
                else:
                    yield " " + word
                time.sleep(0.05)  # Small delay for streaming effect
                
        except Exception as e:
            yield f"Error in streaming response: {e}"
    
    @staticmethod
    
    def print_streaming_response(generator) -> str:
        """Print streaming response and return complete text"""
        complete_response = ""
        print("Answer: ", end="", flush=True)
        
        for chunk in generator:
            print(chunk, end="", flush=True)
            complete_response += chunk
        
        print()  # New line after complete response
        return complete_response
    
# Global instances
cache = RedisCache()
db = DatabaseManager()

# Utility Functions
def get_text_hash(text: str) -> str:
    """Compute SHA256 hash of text."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def check_memory_usage() -> bool:
    """Check if memory usage is acceptable"""
    memory = psutil.virtual_memory()
    if memory.percent > 85:
        logging.warning(f"High memory usage: {memory.percent}%")
        return False
    return True

def memory_cleanup():
    """Perform memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logging.info("Memory cleanup performed")

def get_embedding_cached(text: str):
    """Get embedding with Redis caching"""
    if not embed_model:
        return []
    
    text_hash = get_text_hash(text)
    
    # Try Redis cache first
    embedding = cache.get_embedding(text_hash)
    if embedding:
        return embedding
    
    # Compute new embedding
    try:
        embedding = embed_model.encode(text).tolist()
        cache.set_embedding(text_hash, embedding)
        return embedding
    except Exception as e:
        logging.error(f"Failed to compute embedding: {e}")
        return []
    
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

# Text processing functions
def clean_text(text: str) -> str:
    """Remove extra whitespace, headers/footers."""
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'^(Header|Footer|Page \d+).*', '', text, flags=re.M)
    return text

def chunk_text(text: str, max_tokens: int = MAX_TOKENS_PER_CHUNK) -> list:
    """Split text into token-limited chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_tokens):
        chunk = " ".join(words[i:i+max_tokens])
        if chunk.strip():  # Skip empty
            chunks.append(chunk)
    return chunks

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


def hybrid_chunking(text: str, max_tokens_per_chunk: int = MAX_TOKENS_PER_CHUNK) -> list:
    """
    Perform hybrid chunking: header-based merge until max_tokens, then token-based splitting, resetting to header mode after.
    
    Args:
        text: Full document text.
        max_tokens_per_chunk: Maximum tokens allowed per chunk.
    
    Returns:
        List[dict]: List of chunks with 'text' and 'headers' keys.
    """
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

        # Try to merge with current
        if current_tokens + sec_tokens <= max_tokens_per_chunk:
            current_chunk_text += sec_text + "\n\n"
            current_headers.append(sec["header"])
            current_tokens += sec_tokens
        else:
            # Commit current chunk if any
            if current_chunk_text:
                chunks.append({"text": current_chunk_text.strip(), "headers": current_headers})

            # Handle oversized section with token splitting
            if sec_tokens > max_tokens_per_chunk:
                token_chunks = chunk_text(sec_text, max_tokens=max_tokens_per_chunk)
                for t_chunk in token_chunks:
                    chunks.append({"text": t_chunk, "headers": [sec["header"]]})
            # Start new merge cycle
            else:
                current_chunk_text = sec_text + "\n\n"
                current_headers = [sec["header"]]
                current_tokens = sec_tokens

    # Commit final chunk            
    if current_chunk_text:
        chunks.append({"text": current_chunk_text.strip(), "headers": current_headers})

    return chunks

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

        #Handle Locked PDFs
        if getattr(doc, "is_encrypted", False) or getattr(doc, "needs_pass", False):
            try:
                auth_ok = False
                try:
                    auth_ok = doc.authenticate("")  # may not exist on all versions
                except Exception:
                    auth_ok = False
                if not auth_ok:
                    logging.warning(f"Skipping encrypted or password-protected PDF: {file_path}")
                return ""
            except Exception as e:
                logging.error(f"Error handling encrypted PDF {file_path}: {e}")
                return ""
            
        # Process PDF    
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

def extract_facts_to_json(text: str, source: str = "guideline") -> List[Dict[str, Any]]:
    """
    Robust regex-based fact extraction. Extracts structured facts (quantities, obligations, limits) from free text using regex.
    Prioritizes longer obligation phrases (e.g., 'must not') before shorter ones (e.g., 'must') to avoid conflicting matches.

    Args:
        text (str): Raw guideline/regulation chunk.
        source (str): 'guideline' or 'regulation' (used for tagging).
    
    Returns:
        List[Dict[str, Any]]: Structured facts as JSON-like dicts.
    """
    facts = []

    if not text or not text.strip():
        return facts

    # Pattern for numeric constraints like "85 tomatoes", "200 liters", "5 km"
    num_pattern = re.compile(r'\b(\d+(?:\.\d+)?)\s*([A-Za-z%°μµ/.\-]+)\b')

    # Obligation words (Order Layer Matters) @IMPROVE - Add more obligations
    obligations = { 
        "must not": "prohibited",
        "shall not": "prohibited",
        "may not": "prohibited",
        "must": "mandatory",
        "shall": "mandatory",
        "should": "recommended",
        "may": "permitted"
    }

    # avoid false positives for page/figure markers
    false_units = {"page", "pages", "fig", "figure", "table", "pp"}

    # Scan for numeric constraints
    for match in num_pattern.finditer(text):
        quantity, unit = match.groups()
        unit_norm = unit.lower().strip()
        if unit_norm in false_units:
            continue
        try:
            qty = float(quantity)
        except Exception:
            continue
        facts.append({
            "type": "constraint",
            "source": source,
            "quantity": qty,
            "unit": unit_norm,
            "context": text[max(0, match.start()-40):match.end()+40].strip()
        })

    # Scan for obligations:iterate in order of decreasing phrase length to catch negatives first
    for word, label in sorted(obligations.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE):
            facts.append({
                "type": "obligation",
                "source": source,
                "obligation": label,
                "matched_word": word,
                "context": text[:200]  # snippet
            })

    # dedupe simple (headless) facts by JSON string
    seen = set()
    deduped = []
    for f in facts:
        key = json.dumps(f, sort_keys=True)
        if key not in seen:
            seen.add(key)
            deduped.append(f)
    return deduped

# @COMPLIANCE FUNCTIONS

# Add a new global for stable node ordering / text map
node_id_order = []        # list[str] -> maps FAISS index -> node id
node_id_to_text = {}      # dict[node_id] -> text (for safe lookup)

def extract_entities_relations(markdown: str) -> List[Dict[str, Any]]:
    """
    Robust rule-based extraction of entities (rule IDs, section ids, object mentions)
    and relations from regulation markdown/text.

    Input:
        markdown: full text for a regulation doc (str)

    Output:
        List of triples:
        [
          {
            "head": "Reg101",
            "relation": "requires" | "depends_on" | "overrides" | "applies_to" | "prohibits",
            "tail": "Buffer50m" or "Reg102",
            "evidence": "sentence text containing the relation"
          },
          ...
        ]
    """
    triples = []
    if not markdown or not markdown.strip():
        return triples

    # canonical rule-id regex: supports "Reg 101", "Section 3.4", "Rule12A", etc.
    rule_id_pattern = re.compile(r'\b(?:Reg|Section|Rule)\s*\d+(?:\.\d+)?[A-Z]?\b', flags=re.IGNORECASE)
    rule_ids_found = set([m.group(0).replace(" ", "") for m in rule_id_pattern.finditer(markdown)])

    # Relation keyword groups
    relation_patterns = {
        'requires': ['requires', 'require', 'must', 'shall', 'must comply', 'shall comply', 'required'],
        'depends_on': ['as defined in', 'per', 'see', 'refer to', 'according to'],
        'overrides': ['except', 'notwithstanding', 'supersedes', 'override'],
        'prohibits': ['prohibit', 'forbid', 'must not', 'may not', 'not permitted']
    }

    # lower-case text for quick keyword checks
    lower_text = markdown.lower()

    # Use spaCy for sentence splitting and noun-chunk extraction
    doc = nlp(markdown)
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue

        # find rule ids in the sentence (normalized without spaces)
        sent_rule_ids = [m.group(0).replace(" ", "") for m in rule_id_pattern.finditer(sent_text)]

        # find simple named entities (ORG/GPE etc) and noun_chunks as fallback
        ent_texts = [ent.text for ent in sent.ents if ent.text.strip()]
        noun_chunks = [nc.text for nc in sent.noun_chunks if nc.text.strip()]

        # determine relation type by keywords (first match)
        rel_type = 'applies_to'
        st_low = sent_text.lower()
        for name, keywords in relation_patterns.items():
            if any(k in st_low for k in keywords):
                rel_type = name
                break

        # if we have 2+ rule ids, make a direct triple between them
        if len(sent_rule_ids) >= 2:
            head, tail = sent_rule_ids[0], sent_rule_ids[1]
            triples.append({'head': head, 'relation': rel_type, 'tail': tail, 'evidence': sent_text})
            continue

        # if one rule id + other phrase, attach rule -> phrase
        if len(sent_rule_ids) == 1:
            head = sent_rule_ids[0]
            # look for other rule-like ids in sentence (already none), else fallback to a noun_chunk/entity
            tail = None
            if len(ent_texts) >= 1:
                tail = ent_texts[0]
            elif len(noun_chunks) >= 1:
                # pick a short noun chunk that isn't the rule id itself
                for nc in noun_chunks:
                    if nc.replace(" ", "") != head and len(nc) < 120:
                        tail = nc
                        break
            else:
                # fallback: attempt to capture "Buffer 50m" or numeric+unit patterns
                m = re.search(r'\b(buffer|setback|zone)\b[^.]{0,80}', sent_text, flags=re.IGNORECASE)
                tail = m.group(0).strip() if m else sent_text[:120]

            if tail:
                triples.append({'head': head, 'relation': rel_type, 'tail': tail, 'evidence': sent_text})
            continue

        # no explicit rule ids: attempt to pair two entities/noun-chunks
        if len(ent_texts) >= 2:
            head, tail = ent_texts[0], ent_texts[1]
            triples.append({'head': head, 'relation': rel_type, 'tail': tail, 'evidence': sent_text})
            continue

        # fallback: use first two noun chunks
        if len(noun_chunks) >= 2:
            head, tail = noun_chunks[0], noun_chunks[1]
            triples.append({'head': head, 'relation': rel_type, 'tail': tail, 'evidence': sent_text})
            continue

    # deduplicate by head-relation-tail
    unique = {}
    for t in triples:
        key = f"{t['head']}||{t['relation']}||{t['tail']}"
        if key not in unique:
            unique[key] = t

    return list(unique.values())


def build_reg_graph(regs_folder: str = compliance_regulations_folder, rebuild: bool = False) -> None:
    """
    Build or load a regulations graph (NetworkX DiGraph) and FAISS index.

    Inputs:
        regs_folder: path to PDFs folder
        rebuild: force rebuild ignoring cache

    Side effects (globals set):
        reg_graph (nx.DiGraph), faiss_index (faiss index), node_texts (list[str]),
        node_id_order (list[str]), node_id_to_text (dict)
    """
    global reg_graph, faiss_index, node_texts, node_id_order, node_id_to_text

    cache_path = Path(GRAPH_CACHE_JSON)
    emb_path = Path(GRAPH_EMB_CACHE)
    # Try load cache (must include explicit node_id ordering)
    if cache_path.exists() and not rebuild:
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            reg_graph = nx.node_link_graph(data['graph'])
            node_id_order = data.get('node_ids', list(reg_graph.nodes()))
            # node_texts array should correspond to node_id_order
            node_texts = data.get('node_texts', [reg_graph.nodes[n].get('text', '') for n in node_id_order])
           # Load embeddings (prefer .npy, fallback to JSON-embedded list)
            embeddings = None
            if emb_path.exists():
                try:
                    embeddings = np.load(emb_path).astype("float32")
                except Exception as e:
                    logging.warning(f"Failed to load .npy embeddings: {e}")
            else:
                json_emb = data.get("embeddings")
                if json_emb:
                    embeddings = np.asarray(json_emb, dtype="float32")
                    logging.info("Loaded embeddings from JSON fallback.")

            # Verify embedding integrity
            if embeddings is not None:
                embeddings = np.atleast_2d(np.ascontiguousarray(embeddings)).astype("float32")

                if len(node_id_order) != embeddings.shape[0]:
                    logging.warning(
                        f"Node/embedding count mismatch: {len(node_id_order)} nodes vs {embeddings.shape[0]} embeddings. Rebuilding recommended."
                    )

                # Build FAISS index if available
                if "faiss" in globals() and faiss is not None:
                    dim = embeddings.shape[1]
                    faiss_index = faiss.IndexFlatIP(dim)
                    faiss.normalize_L2(embeddings)
                    faiss_index.add(embeddings)
                    logging.info(f"FAISS index loaded (dim={dim}, nodes={len(node_id_order)}).")
                else:
                    faiss_index = None
                    logging.warning("FAISS not available; skipping index creation.")

                node_id_to_text = {
                    nid: node_texts[i] if i < len(node_texts)
                    else reg_graph.nodes[nid].get("text", "")
                    for i, nid in enumerate(node_id_order)
                }

                logging.info("Loaded regulation graph and embeddings from cache.")
                return
            else:
                logging.warning("Embeddings missing in cache, rebuilding...")
        except Exception as e:
            logging.warning(f"Failed to load graph cache ({e}), rebuilding...")

    # Rebuild graph from scratch
    if not os.path.exists(regs_folder):
        logging.error(f"Regulations folder not found: {regs_folder}")
        reg_graph = nx.DiGraph()
        faiss_index = None
        node_texts, node_id_order, node_id_to_text = [], [], {}
        return

    pdf_files = [f for f in os.listdir(regs_folder) if f.lower().endswith(".pdf")]
    all_triples = []

    for file_name in pdf_files:
        pdf_path = os.path.join(regs_folder, file_name)
        try:
            md = extract_pdf_text(pdf_path)
            triples = extract_entities_relations(md)
            for t in triples:
                t.setdefault("source", file_name)
            all_triples.extend(triples)
        except Exception as e:
            logging.warning(f"Failed to parse {file_name}: {e}")

    reg_graph = nx.DiGraph()
    for t in all_triples:
        h, ta = t["head"], t["tail"]
        if "text" not in reg_graph.nodes.get(h, {}):
            reg_graph.add_node(h, text=f"{h}: {t.get('evidence','')[:300]}")
        if "text" not in reg_graph.nodes.get(ta, {}):
            reg_graph.add_node(ta, text=f"{ta}: {t.get('evidence','')[:300]}")
        reg_graph.add_edge(
            h, ta,
            relation=t.get("relation", "applies_to"),
            evidence=t.get("evidence", ""),
            source=t.get("source")
        )

    if reg_graph.number_of_nodes() == 0:
        logging.warning("No nodes found after parsing regulations.")
        reg_graph = nx.DiGraph()
        faiss_index = None
        node_texts, node_id_order, node_id_to_text = [], [], {}
        return

    node_id_order = list(reg_graph.nodes())
    node_texts = [reg_graph.nodes[n].get("text", "") for n in node_id_order]
    node_id_to_text = {nid: node_texts[i] for i, nid in enumerate(node_id_order)}

    try:
        embeddings = np.asarray(embed_model.encode(node_texts, convert_to_numpy=True)).astype("float32")
    except TypeError:
        embeddings = np.asarray(embed_model.encode(node_texts)).astype("float32")

    embeddings = np.atleast_2d(np.ascontiguousarray(embeddings))
    faiss_index = None
    if "faiss" in globals() and faiss is not None:
        dim = embeddings.shape[1]
        faiss_index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)
        faiss_index.add(embeddings)
        logging.info(f"FAISS index built (dim={dim}, nodes={len(node_id_order)}).")
    else:
        logging.warning("FAISS not available; skipping index creation.")

    # Atomic cache saving in case of crashes
    graph_data = nx.node_link_data(reg_graph)
    cache_data = {
        "graph": graph_data,
        "node_ids": node_id_order,
        "node_texts": node_texts
    }

    tmp_emb = str(emb_path) + ".tmp"
    tmp_json = str(cache_path) + ".tmp"

    try:
        np.save(tmp_emb, embeddings)
        os.replace(tmp_emb, emb_path)
        with open(tmp_json, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2)
        os.replace(tmp_json, cache_path)
        logging.info(f"Built and cached regulation graph ({len(node_id_order)} nodes, {reg_graph.number_of_edges()} edges).")
    except Exception as e:
        logging.warning(f"Failed to write cache files: {e}")

def graph_enhanced_retrieval(section_text: str, top_k = TOP_K_DEFAULT, traversal_depth: int = 1) -> str:
    """
    Hybrid retrieval: embed the query, search FAISS, map results to node IDs using node_id_order,
    and return a short linearized subgraph that includes relations/evidence.
    Returns a concise string (trimmed) suitable to pass to the LLM prompt.
    """
    global reg_graph, faiss_index, node_texts, node_id_order, node_id_to_text

    if reg_graph is None or faiss_index is None or not node_id_order:
        build_reg_graph()

    if reg_graph is None or faiss_index is None or not node_id_order:
        return "No regulations available."

    # embed query robustly
    try:
        q_emb = np.asarray(embed_model.encode([section_text], convert_to_numpy=True)).astype('float32')
    except TypeError:
        q_emb = np.asarray(embed_model.encode([section_text])).astype('float32')

    q_emb = np.ascontiguousarray(q_emb)
    faiss.normalize_L2(q_emb)
    try:
        scores, indices = faiss_index.search(q_emb, top_k)
    except Exception as e:
        logging.error(f"FAISS search failed: {e}")
        return "Regulations retrieval failed."

    retrieved_parts = []
    for i, idx in enumerate(indices[0]):
        if idx == -1 or idx >= len(node_id_order):
            continue
        node_id = node_id_order[idx]
        node_text = node_id_to_text.get(node_id, reg_graph.nodes[node_id].get('text', ''))
        # collect neighbors (ego graph) due to traversal_depth
        subgraph = nx.ego_graph(reg_graph, node_id, radius=traversal_depth)
        # linearize: node + immediate edges (limit lengths)
        part = f"[{node_id}] {node_text[:300]}"
        for u, v, data in subgraph.edges(data=True):
            rel = data.get('relation', '')
            evidence = data.get('evidence', '')[:180]
            part += f" -> {u} [{rel}] {v}: {evidence}..."
        retrieved_parts.append(part[:1200])  # trim each retrieved part to avoid huge prompts

    if not retrieved_parts:
        return "No relevant regulations found."

    return "Relevant Regulations with Relations: " + " || ".join(retrieved_parts)


def _extract_json_array_from_text(text: str) -> Optional[str]:
    """
    Helper: extract the first JSON array substring (a [...] block) from text (dotall).
    Returns the substring or None if not found.
    """
    m = re.search(r'(\[[\s\S]*\])', text)
    if m:
        return m.group(1)
    return None

def split_pdf_into_sections(pdf_path):
    """Split project PDF into semantic sections using hybrid_chunking.

    Args:
        pdf_path (str): Path to project PDF.

    Returns:
        List[Dict[str, Any]]: Sections as [{'title': str, 'content': str, 'metadata': dict}].
    """
    text = extract_pdf_text(pdf_path)
    chunks = hybrid_chunking(text)
    sections = []
    for i, chunk in enumerate(chunks):
        title = chunk['headers'][0] if chunk['headers'] else f"Section {i+1}"
        metadata = {"estimated_tokens": len(chunk['text'].split()), "source": pdf_path}
        sections.append({"title": title, "content": chunk['text'], "metadata": metadata})
    return sections

def check_compliance_for_section_with_regex(section: Dict[str, Any], rag_regs: str, model_name: str = str(LLM_MODEL_NAME)):
    """
    Enhanced compliance check:
    - Extracts structured JSON facts from section and regs using regex.
    - Passes both raw text and extracted JSON to AI.
    - Saves JSON, query, and AI answer to a log file for transparency.
    - Extracts structured JSON facts from section and regs using regex.
    - If rag_regs is not provided, compute it via graph_enhanced_retrieval.
    - Calls the LLM and attempts robust JSON extraction.

    Returns:
        dict with 'flags' (AI output), 'facts' (structured facts used), 'answer' (LLM raw answer)
    """
    section_text = section.get("content", "") or ""
    if rag_regs is None:
        try:
            rag_regs = graph_enhanced_retrieval(section_text)
        except Exception as e:
            rag_regs = ""

    # Extract structured facts
    guideline_facts = extract_facts_to_json(section_text, source="guideline")
    reg_facts = extract_facts_to_json(rag_regs or "", source="regulation")
    structured_json = {"guideline_facts": guideline_facts, "regulation_facts": reg_facts}

    # Compliance prompt for AI
    prompt = f"""
You are a regulatory compliance expert.
Here is a project section and relevant regulations.

Section Title: {section.get('title')}
Section Content: {section_text}

Relevant Regulations: {rag_regs}

Here are structured facts extracted with regex:
{json.dumps(structured_json, indent=2)}

Use BOTH the raw text and structured facts to check for violations.
Output PURE JSON list of flags with keys:
["issue","evidence","reg_ref","severity","confidence"].
"""

    output = ""   # always defined
    flags: List[Dict[str, Any]] = []

    # Call LLM
    try:
        response = llm.chat(model=model_name, messages=[{'role': 'user', 'content': prompt}],
                           options={"temperature": 0.4, "top_p": 0.9})
        # extract text robustly
        if isinstance(response, dict) and 'message' in response and isinstance(response['message'], dict):
            output = response['message'].get('content', '')
        elif hasattr(response, 'message') and hasattr(response.message, 'content'):
            output = response.message.content
        else:
            output = str(response)
        output = output.strip()

        # Try direct parse, then fallback to JSON substring extraction
        try:
            parsed = json.loads(output)
            if isinstance(parsed, list):
                flags = parsed
        except Exception:
            json_sub = _extract_json_array_from_text(output)
            if json_sub:
                try:
                    parsed = json.loads(json_sub)
                    if isinstance(parsed, list):
                        flags = parsed
                except Exception:
                    flags = []
            else:
                flags = []
    except Exception as e:
        logging.error(f"Regex+AI compliance check failed: {e}")
        flags = []

    # normalize flags: ensure list of dicts, normalize confidence values
    safe_flags = []
    for f in flags:
        if not isinstance(f, dict):
            continue
        f.setdefault('issue', '')
        f.setdefault('evidence', '')
        f.setdefault('reg_ref', '')
        f.setdefault('severity', 'medium')
        # normalize confidence
        try:
            conf = float(f.get('confidence', 0.5))
            conf = max(0.0, min(1.0, conf))
        except Exception:
            conf = 0.5
        f['confidence'] = conf
        safe_flags.append(f)

    # Save transparency log
    log_entry = {
        "section_title": section.get("title"),
        "section_content": section_text[:400],
        "relevant_regs": rag_regs[:400],
        "structured_facts": structured_json,
        "ai_answer": output,
        "flags": flags
    }
    try:
        with open("compliance_logs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        logging.warning(f"Failed to write compliance log: {e}")

    return {"flags": flags, "facts": structured_json, "answer": output}

def check_file_compliance(project_desc: str, project_pdf_path: str):
    """
    Compliance checker: Splits project PDF, checks sections aginst regulations graph RAG, aggregates flags.

    Returns a report dict that preserves the original keys and includes:
      - section_reports: List[ { index, title, flags, error (or None), duration_s, skipped } ]
      - processing_time_s: total time spent for this file
    """
    start_total = time.time()
    project_pdf_path = os.path.abspath(project_pdf_path)

    if not os.path.exists(project_pdf_path):
        raise ValueError(f"Project PDF not found: {project_pdf_path}")

    # Attempt to split into sections; propagate error (caller typically handles it)
    try:
        sections = split_pdf_into_sections(project_pdf_path)
    except Exception as e:
        logging.exception(f"Failed to split PDF into sections: {project_pdf_path}")
        # Keep the raising behavior for a missing/invalid PDF as before.
        raise

    all_flags: List[Dict[str, Any]] = []
    section_reports: List[Dict[str, Any]] = []

    if not sections:
        # No content extracted — return an explicit report (keeps original keys)
        report = {
            "project_desc": project_desc,
            "project_pdf": project_pdf_path,
            "sections_checked": 0,
            "section_reports": [],
            "flags": [],
            "overall_summary": {
                "total_flags": 0,
                "compliance_status": "No content extracted"
            },
            "processing_time_s": time.time() - start_total
        }
        return report

    # Process each section independently; capture exceptions per-section
    for idx, section in enumerate(sections):
        t0 = time.time()
        sec_report: Dict[str, Any] = {
            "index": idx,
            "title": section.get("title"),
            "metadata": section.get("metadata", {}),
            "flags": [],
            "error": None,
            "duration_s": None,
            "skipped": False
        }

        # Lightweight skip: skip sections with essentially no content
        content = (section.get("content") or "").strip()
        if len(content) < 20:
            sec_report["skipped"] = True
            sec_report["duration_s"] = 0.0
            section_reports.append(sec_report)
            continue

        try:
            # compute RAG regs for this section and call the checker
            rag_regs = graph_enhanced_retrieval(content)
            result = check_compliance_for_section_with_regex(section, rag_regs=rag_regs)
            # result should be dict; get flags safely
            if isinstance(result, dict):
                flags = result.get("flags", [])
            elif isinstance(result, list):
                flags = result
            else:
                flags = []

            if not isinstance(flags, list):
                logging.warning(f"Unexpected flags type for section {idx} in {project_pdf_path}")
                flags = []

            # annotate each flag with section index/title if not already present
            for f in flags:
                if isinstance(f, dict):
                    f.setdefault("section_index", idx)
                    f.setdefault("section_title", section.get("title"))
            sec_report["flags"] = flags
            all_flags.extend(flags)
        except Exception as e:
            logging.exception(f"Error checking compliance for section {idx} in {project_pdf_path}: {e}")
            sec_report["error"] = str(e)
            # Continue processing other sections (do not re-raise)
        finally:
            sec_report["duration_s"] = time.time() - t0
            section_reports.append(sec_report)

        # Optional: memory check after each section (uncomment if desired)
        if not check_memory_usage():
            memory_cleanup()

    total_flags = len(all_flags)
    processing_time = time.time() - start_total

    report = {
        "project_desc": project_desc,
        "project_pdf": project_pdf_path,
        "sections_checked": len(sections),
        "section_reports": section_reports,
        "flags": all_flags,
        "overall_summary": {
            "total_flags": total_flags,
            "compliance_status": "Compliant" if total_flags == 0 else "Non-Compliant (Review Required)"
        },
        "processing_time_s": processing_time
    }
    return report


def check_project_compliance(compliance_folder = project_guidelines_folder, project_desc: str = None):
    """
    Batch compliance checker: Processes all PDFs in the project guidelines folder folder, then generates individual compliance reports .
    """
    start_batch = time.time()
    if not os.path.exists(compliance_folder):
        logging.error(f"Compliance folder not found: {compliance_folder}")
        return {"error": f"Folder not found: {compliance_folder}"}

    # deterministic order and case-insensitive pdf detection
    pdf_files = sorted([f for f in os.listdir(compliance_folder) if f.lower().endswith(".pdf")])

    if not pdf_files:
        logging.warning(f"No PDF files found in {compliance_folder}")
        return {"error": "No PDF files found in folder"}

    logging.info(f"Processing {len(pdf_files)} PDFs for compliance checking...")
    print(f"\nProcessing {len(pdf_files)} project PDFs for compliance...\n")

    all_reports: List[Dict[str, Any]] = []
    total_flags = 0
    files_checked = 0
    files_failed = 0

    for pdf_file in pdf_files:
        pdf_path = os.path.join(compliance_folder, pdf_file)
        desc = project_desc or f"Compliance check for {pdf_file}" ##IMPROVE##

        print(f"Checking: {pdf_file}...")
        file_start = time.time()
        try:
            report = check_file_compliance(desc, pdf_path)
            # Ensure a dict
            if not isinstance(report, dict):
                raise RuntimeError("Child report is not a dict")

            # attach filename and timing
            report["filename"] = pdf_file
            report["processing_time_s"] = report.get("processing_time_s", time.time() - file_start)

            all_reports.append(report)
            # safe read of flags from child report
            file_flags = report.get("overall_summary", {}).get("total_flags", 0)
            try:
                file_flags = int(file_flags)
            except Exception:
                file_flags = 0
            total_flags += file_flags
            files_checked += 1

            status = "✓ Compliant" if file_flags == 0 else f"✗ {file_flags} violations"
            print(f"  {status}\n")

        except Exception as e:
            logging.exception(f"Failed to process {pdf_file}: {e}")
            files_failed += 1
            print(f"  ✗ Error processing file: {e}\n")

    batch_time = time.time() - start_batch
    aggregated_report = {
        "batch_description": project_desc or "Batch compliance check",
        "compliance_folder": compliance_folder,
        "files_checked": files_checked,
        "files_failed": files_failed,
        "total_pdfs": len(pdf_files),
        "total_violations": total_flags,
        "individual_reports": all_reports,
        "overall_summary": {
            "compliant_files": sum(1 for r in all_reports if r.get("overall_summary", {}).get("total_flags", 0) == 0),
            "non_compliant_files": sum(1 for r in all_reports if r.get("overall_summary", {}).get("total_flags", 0) > 0),
            "total_flags_across_all_files": total_flags,
            "batch_status": "All Compliant" if total_flags == 0 else f"{total_flags} Total Violations Across {files_checked} Files"
        },
        "processing_time_s": batch_time
    }
    return aggregated_report

# End of @Compliance Functions

def visualize_reg_graph(output_path: str = "reg_graph.png"):
    """
    Visualize the regulations graph for the user for transparency.
    
    Args:
        output_path (str): File path to save the visualization.
    """
    global reg_graph
    if reg_graph is None or reg_graph.number_of_nodes() == 0:
        print("No regulation graph available to visualize.")
        return

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(reg_graph, k=0.5, iterations=50)
    nx.draw(reg_graph, pos, with_labels=True, node_size=1500, node_color="lightblue", font_size=8, font_weight="bold", arrowsize=15)
    edge_labels = nx.get_edge_attributes(reg_graph, 'relation')
    nx.draw_networkx_edge_labels(reg_graph, pos, edge_labels=edge_labels, font_size=7)

    plt.title("Regulation Graph")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Graph visualization saved at {output_path}")

def ingest_documents():
    """Ingest documents into SQLite + FAISS"""
    if not os.path.exists(knowledge_base_folder):
        logging.error(f"Document folder not found: {knowledge_base_folder}")
        return
    
    files = [f for f in os.listdir(knowledge_base_folder) if f.lower().endswith(('.txt', '.pdf', '.md'))]
    if not files:
        logging.warning("No documents found to ingest")
        return
    
    logging.info(f"Ingesting {len(files)} files...")
    
    for file_name in files:
        file_path = os.path.join(knowledge_base_folder, file_name)
        
        # Extract text
        if file_name.endswith('.txt'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except Exception as e:
                logging.error(f"Failed to read {file_name}: {e}")
                continue
        else:  # PDF
            text = extract_pdf_text(file_path)
        
        if not text.strip():
            logging.warning(f"No text extracted from {file_name}")
            continue
        
        # Chunk text
        chunks = hybrid_chunking(text)
        
        # Get embeddings for all chunks
        chunk_texts = [chunk["text"] for chunk in chunks]
        embeddings = get_batch_embeddings(chunk_texts)
        
        # Add to database
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            if embedding:  # Only add if embedding is valid
                doc_id = f"{file_name}_{i}"
                db.add_document(
                    doc_id=doc_id,
                    content=chunk["text"],
                    source_file=file_name,
                    headers=chunk["headers"],
                    chunk_index=i,
                    embedding=embedding
                )
        
        logging.info(f"Ingested {file_name} with {len(chunks)} chunks")
        
        # Periodic memory check
        if not check_memory_usage():
            memory_cleanup()
    
    # Save FAISS index
    db.save_faiss_index()
    print(f"Document ingestion complete! Total documents: {db.get_document_count()}")

def hybrid_search(query: str, top_k: int = TOP_K_DEFAULT):
    """Perform hybrid search using FAISS + BM25"""
    if not query.strip():
        return "", []
    
    # Get query embedding
    query_embedding = get_embedding_cached(query)
    if not query_embedding:
        logging.error("Failed to get query embedding")
        return "", []
    
    # FAISS semantic search
    semantic_results = db.search_similar(query_embedding, top_k * 2)
    
    if not semantic_results:
        logging.warning("No semantic search results")
        return "", []
    
    # BM25 re-ranking
    documents = [result['content'] for result in semantic_results]
    tokenized_docs = [doc.split() for doc in documents]
    
    try:
        bm25 = BM25Okapi(tokenized_docs)
        bm25_scores = bm25.get_scores(query.split())
        
        # Combine scores (70% semantic, 30% BM25)
        final_results = []
        for i, result in enumerate(semantic_results):
            combined_score = 0.7 * result['similarity_score'] + 0.3 * bm25_scores[i]
            result['combined_score'] = combined_score
            final_results.append(result)
        
        # Sort by combined score and take top_k
        final_results.sort(key=lambda x: x['combined_score'], reverse=True)
        top_results = final_results[:top_k]
        
        # Create context
        context = " ".join([result['content'] for result in top_results])
        
        return context, top_results
        
    except Exception as e:
        logging.error(f"Hybrid search failed: {e}")
        # Fallback to semantic only
        context = " ".join([result['content'] for result in semantic_results[:top_k]])
        return context, semantic_results[:top_k]

def ask_rag_streaming(query: str, model_name: str = LLM_MODEL_NAME):
    """RAG query with streaming response"""
    # Check for cached response
    query_hash = get_text_hash(f"{query}_{model_name}")
    cached_response = cache.get_response(query_hash)
    
    if cached_response:
        print("Answer:", cached_response)
        logging.info("Returned cached response")
        return cached_response, []
    
    # Perform hybrid search
    context, search_results = hybrid_search(query)
    
    if not context:
        error_msg = "No relevant information found in the knowledge base."
        print("Answer:", error_msg)
        return error_msg, []
    
    # Prepare messages
    messages = [
        {
            "role": "system",
            "content": "You are an expert environmental policy assistant. Provide clear, accurate answers based on the given context. Think step by step. Cite sources when possible."
        },
        {
            "role": "user",
            "content": f"Context: {context[:8000]}\n\nQuestion: {query}"
        }
    ]
    
    # Stream response
    response_generator = StreamingResponse.stream_ollama_response(
        model_name=model_name,
        messages=messages,
        options={
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 2048
        }
    )
    
    # Print streaming response and get complete text
    complete_response = StreamingResponse.print_streaming_response(response_generator)
    
    # Cache the response
    cache.set_response(query_hash, complete_response)
    
    return complete_response, search_results

def main():
    """Main application"""
    logging.info("Starting GreenPolicyAI with advanced optimizations...")
    
    # System info
    memory_gb = psutil.virtual_memory().total / (1024**3)
    logging.info(f"System memory: {memory_gb:.1f}GB")
    logging.info(f"Redis enabled: {cache.enabled}")
    logging.info(f"FAISS enabled: {FAISS_AVAILABLE}")
    
    # Ingest documents
    if db.get_document_count() == 0:
        print("No documents found in database. Starting ingestion...")
        ingest_documents()
    else:
        print(f"Found {db.get_document_count()} documents in database.")
    
    # Main REPL with streaming
    print("\nWelcome to GreenPolicyAI.")
    print("To perform a compliance check, please type '!COMPLIANCE!'")
    print("Type 'exit' or 'quit' to end the program.")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\nQuestion: ").strip()
            
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            
            # Check memory before processing
            if not check_memory_usage():
                print("Warning: High memory usage detected. Consider restarting.")
                memory_cleanup()

            # Use compliance checker if requested 
            if user_input in ["!COMPLIANCE!"]:
                start_time = time.time()
                print("Beginning Compliance Check...")
                check_project_compliance()
                end_time = time.time()

            # Else process query as usual
            else:
                start_time = time.time()
                response, sources = ask_rag_streaming(user_input)
                end_time = time.time()
            
            # Show sources
            if sources:
                unique_sources = list(set([s['source_file'] for s in sources]))
                print(f"\nSources: {', '.join(unique_sources)}")
                print(f"Response time: {end_time - start_time:.2f}s") # Comment out during publishing stage.
            else:
                print("No sources available")
            
        except KeyboardInterrupt:
            print("\nProgram interrupted by user.")
            break
        except MemoryError:
            logging.error("System memory error")
            print("System out of memory. Restarting recommended.")
            memory_cleanup()

        except Exception as e:
            logging.error(f"Main loop error: {e}")
            print(f"An error occurred: {e}")
            print("Check logs for details.")

if __name__ == "__main__":
    main()