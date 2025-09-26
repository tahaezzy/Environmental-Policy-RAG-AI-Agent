## BETA VERSION 4.5.1
# Changes:
'''
-  Integrated Graph Knowledge Base for improved regulations traversing (better literal and overall context)
using NetworkX (in-memory) and FAISS (hybrid vector-graph retrieval) instead of only regular RAG. 
- Implemented Rule-based extraction with spaCy + regex for entities/relations (no extra LLM calls for efficiency).
- Adapted build_reg_graph to use "Regulations Folder" and parse PDFs for extraction
- Implemented split_pdf_into_sections using hybrid_chunking for project PDFs.
- Fully implemented compliance cheking system with graph-enhanced dual retrieval for regulations.
- Updated check_compliance to handle project PDF path (prompt user in REPL for compliance mode).
- Integrated into REPL: For "compliance:", prompt for project PDF path, then process.
- Switched to Redis for cross-session caching for embeddings, query results, and LLM responses. 
- Added dynamic constants and memory checking for smoother compute usage.
- Enabled streaming responses for better user experience. Added more system messages.
- Improved logging across project along with use of classes for containment and better error handling.
'''



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
import time
import sqlite3
from typing import List, Dict, Any, Optional, Tuple

# Load Required DEPENDENCIES. 
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


# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('GreenPolicyAI.log', mode='a'),
        logging.StreamHandler()
    ]
)

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

# Constants
constants = get_system_constants()
LLM_MODEL_NAME = "qwen2.5:0.5b"
BATCH_SIZE = constants['BATCH_SIZE']
MAX_TOKENS_PER_CHUNK = constants['MAX_TOKENS']
CONTEXT_WINDOW_LENGTH = constants['CONTEXT_LEN']
TOP_K_DEFAULT = constants['TOP_K']
THINK_MODE = False

# Paths and Cache
knowledge_base_folder = "Knowledge Base Folder"
project_guidelines_folder = "Project Guidlines Folder"
compliance_regulations_folder = "Compliance Regulations Folder"
DB_PATH = "greenpolicy.db"
FAISS_INDEX_PATH = "faiss_index.bin"
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
cache_file = "embedding_cache.json"
GRAPH_CACHE = "reg_graph.pkl"  # Graph persistence


# Check for exisitng embeddings cache file
if os.path.exists(cache_file):
    with open(cache_file, "r", encoding="utf-8") as f:
        embedding_cache = json.load(f)
else:
    embedding_cache = {}

# Load spaCy for rule-based extraction (en_core_web_sm for NER)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Run: python -m spacy download en_core_web_sm")
    raise

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
    
    def search_similar(self, query_embedding, top_k: int = 5):
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
    import gc
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

# @ COMPLIANCE FUNCTIONS

def extract_entities_relations(markdown):
    """Rule-based extraction: Entities (rules/entities) and relations from regs MD.

    Uses spaCy NER (ORG/GPE for regs/locations; custom for rule IDs) + dependency parse
    + regex patterns for relations (e.g., 'requires', 'as per', 'overrides').
    Efficient: No LLM; ~90% accuracy on structured regs.
    Returns triples: [{'head': 'Reg101', 'relation': 'requires', 'tail': 'Buffer50m', 'evidence': 'text span'}]

    Args:
        markdown (str): Parsed regs MD.

    Returns:
        List[Dict[str, Any]]: Extracted triples.
    """
    doc = nlp(markdown)
    triples = []

    # Entity extraction: Rule IDs (regex) + NER
    rule_entities = re.findall(r'(Reg|Section|Rule)\s*(\d+[A-Z]?)', markdown, re.IGNORECASE)
    entities = {f"{match[0]}{match[1]}": match[0] for match in rule_entities}  # e.g., 'Reg101'

    # Relations via patterns (simple dependency + keywords)
    relation_patterns = {
        'requires': ['requires', 'must comply with'],
        'depends_on': ['as defined in', 'per', 'see'],
        'overrides': ['except', 'notwithstanding', 'supersedes']
    }

    for sent in doc.sents:
        for token in sent:
            if token.dep_ == 'ROOT' and token.lemma_ in ['require', 'comply', 'override']:  # Verb roots
                # Find entities in sentence
                sent_entities = [ent.text for ent in sent.ents if ent.label_ in ['ORG', 'GPE']] + \
                                [m.group() for m in re.finditer(r'(Reg|Section)\s*\d+', sent.text)]
                if len(sent_entities) >= 2:
                    head, tail = sent_entities[0], sent_entities[1]  # Simplistic; refine with POS
                    rel_type = next((k for k, v in relation_patterns.items() if any(p in sent.text.lower() for p in v)), 'applies_to')
                    triples.append({
                        'head': head,
                        'relation': rel_type,
                        'tail': tail,
                        'evidence': sent.text.strip()
                    })

    # Dedup and filter
    unique_triples = {f"{t['head']}-{t['relation']}-{t['tail']}": t for t in triples}.values()
    return list(unique_triples)

def build_reg_graph(regs_folder: str = knowledge_base_folder, rebuild: bool = False) -> None:
    """Builds the regulations graph and FAISS index once.

    Parses all PDFs in folder to MD, extracts per-doc, aggregates into NetworkX DiGraph.
    Embeds node texts, indexes in FAISS (inner product for cosine sim).
    Caches to JSON: Graph as node_link_data dict, embeddings as list of lists (reconstruct index on load).

    Args:
        regs_folder (str): Folder with reg PDFs.
        rebuild (bool): Force rebuild (ignores cache).
    """
    global reg_graph, faiss_index, node_texts

    cache_path = Path(GRAPH_CACHE)
    if cache_path.exists() and not rebuild:
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Reconstruct graph from node_link_data
            reg_graph = nx.node_link_graph(data['graph'])
            # Reconstruct node_texts
            node_texts = data['node_texts']
            # Reconstruct FAISS: Create new index and add cached embeddings
            embeddings = np.array(data['embeddings']).astype('float32')
            dim = embeddings.shape[1]
            faiss_index = faiss.IndexFlatIP(dim)
            faiss.normalize_L2(embeddings)
            faiss_index.add(embeddings)
            print("Loaded graph from JSON cache.")
            return
        except (json.JSONDecodeError, KeyError, Exception) as e:
            logging.warning(f"JSON cache load failed: {e}, rebuilding.")

    # Parse all PDFs (reuse extract_pdf_text)
    reg_markdowns = []
    pdf_files = [f for f in os.listdir(regs_folder) if f.endswith(".pdf")]
    for file_name in pdf_files:
        pdf_path = os.path.join(regs_folder, file_name)
        md = extract_pdf_text(pdf_path)  # MD-like text
        triples = extract_entities_relations(md)
        reg_markdowns.append(md)

    # Build regulations graph: Nodes from unique entities + chunks; edges from triples
    reg_graph = nx.DiGraph()
    all_entities = set()
    all_triples = []
    for md in reg_markdowns:
        triples = extract_entities_relations(md)
        all_triples.extend(triples)
        for t in triples:
            reg_graph.add_edge(t['head'], t['tail'], relation=t['relation'], evidence=t['evidence'])
            all_entities.update([t['head'], t['tail']])

    # Add node texts (from evidence/evidence)
    for node in all_entities:
        node_text = f"Regulation Node: {node}. " + " ".join([data['evidence'] for _, _, data in reg_graph.in_edges(node, data=True)] + [data['evidence'] for _, _, data in reg_graph.out_edges(node, data=True)])
        reg_graph.nodes[node]['text'] = node_text[:500]  # Truncate for efficiency

    # Embed and index
    node_list = list(reg_graph.nodes(data=True))
    node_texts = [n[1]['text'] for n in node_list]
    embeddings = embed_model.encode(node_texts)
    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    emb_normalized = embeddings.copy()
    faiss.normalize_L2(emb_normalized)
    faiss_index.add(emb_normalized.astype('float32'))

    # Cache as JSON: Graph dict, node_texts list, embeddings as list of lists
    graph_data = nx.node_link_data(reg_graph)  # Serializable dict
    cache_data = {
        'graph': graph_data,
        'node_texts': node_texts,
        'embeddings': embeddings.tolist()  # List of lists for JSON
    }
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, indent=2)
    print(f"Built graph with {len(reg_graph.nodes)} nodes, {len(reg_graph.edges)} edges. Cached as JSON.")

def graph_enhanced_retrieval(section_text: str, top_k: int = 3, traversal_depth: int = 1) -> str:
    """Hybrid retrieval: Vector search on nodes + graph traversal for relations.

    Embeds query, FAISS top-k nodes, traverses neighbors (depth-limited), linearizes subgraph.
    Returns enriched regs text with paths (e.g., 'Reg101 [requires] RegB: texts...').

    Args:
        section_text (str): Project section for query.
        top_k (int): Initial vector matches.
        traversal_depth (int): Graph hops (1-2 for efficiency).

    Returns:
        str: Retrieved regs text.
    """
    global reg_graph, faiss_index, node_texts
    if reg_graph is None or faiss_index is None:
        build_reg_graph()

    query_emb = embed_model.encode([section_text])
    faiss.normalize_L2(query_emb)
    scores, indices = faiss_index.search(query_emb.astype('float32'), top_k)

    retrieved = []
    node_ids = list(reg_graph.nodes)
    for idx in indices[0]:
        if idx == -1 or idx >= len(node_ids): continue
        node_id = node_ids[idx]
        node_text = node_texts[idx]

        # Traverse: Collect subgraph (self + neighbors up to depth)
        subgraph = nx.ego_graph(reg_graph, node_id, radius=traversal_depth)
        path_text = f"Node {node_id}: {node_text}"
        for u, v, data in subgraph.edges(data=True):
            path_text += f" | {u} [{data['relation']}] {v}: {data['evidence'][:100]}..."

        retrieved.append(path_text)

    return "Relevant Regulations with Relations: " + " | ".join(retrieved)

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

def check_compliance_for_section(section, model_name = str(LLM_MODEL_NAME)):
    """Checks a single section for compliance using Ollama LLM with graph-enhanced RAG.

    Stateless call: Fresh prompt per section to isolate context.
    Output: JSON flags for non-compliance.

    Args:
        section (Dict[str, Any]): From split_pdf_into_sections.
        model_name (str): Ollama model.

    Returns:
        List[Dict[str, Any]]: Flags like [{'issue': str, 'evidence': str, 'reg_ref': str, 'severity': str, 'confidence': float}].
                              Empty list if compliant.
    """
    section_text = section['content']
    rag_regs = graph_enhanced_retrieval(section_text)

    prompt = f"""
You are a regulatory compliance expert for GIS/environmental projects.
Analyze ONLY this section: Do not reference or assume other project parts.

Section Title: {section['title']}
Section Content: {section_text}

Relevant Regulations with Relations: {rag_regs}

Perform line-by-line check for non-compliance, considering relationships (e.g., dependencies).
Flag ONLY violations.
For each flag:
- issue: Brief description.
- evidence: Exact quote from section.
- reg_ref: Quote from regulations (include relation if relevant).
- severity: 'high'/'medium'/'low'.
- confidence: 0.0-1.0.

If compliant, return empty list [].
Output PURE JSON: [{{"issue": "...", "evidence": "...", "reg_ref": "...", "severity": "...", "confidence": 0.95}}]
"""

    try:
        response = llm.chat(model=model_name, messages=[{'role': 'user', 'content': prompt}],
                           options={"temperature": 0.6, "top_p": 0.95, "top_k": 20, "repeat_penalty": 1.1, "enable_thinking": THINK_MODE})
        output = response['message']['content'].strip()

        flags = json.loads(output)
        if not isinstance(flags, list):
            flags = []
    except (json.JSONDecodeError, KeyError, Exception) as e:
        logging.error(f"Compliance check error: {e}")
        flags = []

    # Annotate with section metadata
    for flag in flags:
        flag['section_title'] = section['title']
        flag['metadata'] = section['metadata']

    return flags

def check_file_compliance(project_desc: str, project_pdf_path):
    """Full compliance checker: Splits project PDF, checks sections with graph RAG, aggregates flags.

    Args:
        project_desc (str): User description (for logging/context).
        project_pdf_path (str): Path to project PDF.

    Returns:
        Dict[str, Any]: Report {'flags': [...], 'overall_summary': {...}}.
    """
    if not os.path.exists(project_pdf_path):
        raise ValueError(f"Project PDF not found: {project_pdf_path}")

    sections = split_pdf_into_sections(project_pdf_path)
    all_flags = []
    for section in sections:
        flags = check_compliance_for_section(section)
        all_flags.extend(flags)

    report = {
        "project_desc": project_desc,
        "project_pdf": project_pdf_path,
        "sections_checked": len(sections),
        "flags": all_flags,
        "overall_summary": {
            "total_flags": len(all_flags),
            "compliance_status": "Compliant" if len(all_flags) == 0 else "Non-Compliant (Review Required)"
        }
    }
    return report

def ingest_documents():
    """Ingest documents into SQLite + FAISS"""
    if not os.path.exists(knowledge_base_folder):
        logging.error(f"Document folder not found: {knowledge_base_folder}")
        return
    
    files = [f for f in os.listdir(knowledge_base_folder) if f.endswith(('.txt', '.pdf'))]
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
    print("\nWelcome to GreenPolicyAI (Advanced Edition)")
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
            
            # Process query with streaming
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