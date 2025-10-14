import os
import json
import torch
import pytesseract
import logging
from sentence_transformers import SentenceTransformer, CrossEncoder 
from ollama import Client as OllamaClient
from pathlib import Path  # For cache path
import psutil
from rank_bm25 import BM25Okapi
import numpy as np
import time
import datetime
import re
import fitz
import pdfplumber
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import gc

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('GreenPolicyAI.log', mode='a'),
        logging.StreamHandler()
    ]
)

try:
    import spacy  ## SPACY
    SPACY_AVAILABLE = True
    logging.info("spaCy loaded successfully")
except ImportError:
    logging.warning("spaCy not available - run: python -m spacy download en_core_web_sm")
    SPACY_AVAILABLE = False
    spacy = None

try:
    import networkx as nx ## NETWORKX
    logging.info("NetworkX loaded successfully")
except ImportError:
    logging.warning("NetworkX not available - graph features disabled")
    nx = None

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
        return {'BATCH_SIZE': 32, 'MAX_TOKENS': 512, 'CONTEXT_LEN': 16384, 'TOP_K': 4}
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

