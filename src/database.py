# src/database.py
from src.config import os, json, np, List, Dict, Any, Optional, Tuple 
import pickle
import sqlite3

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

from src.config import embed_model, TOP_K_DEFAULT, logging

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
FAISS_INDEX_PATH = "faiss_index.bin"
DB_PATH = "greenpolicy.db"

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
    
    def add_document(self, doc_id: str, content: str, source_file: str, headers: List[str], chunk_index: int, embedding: np.ndarray) -> None:
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
    
    def search_similar(self, query_embedding: np.ndarray, top_k: int = TOP_K_DEFAULT) -> List[Tuple[int, float]]:
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
    
    def save_faiss_index(self) -> None:
        """Save FAISS index to disk"""
        if self.faiss_index:
            try:
                faiss.write_index(self.faiss_index, self.faiss_index_path)
                logging.info("FAISS index saved successfully")
            except Exception as e:
                logging.error(f"Failed to save FAISS index: {e}")

cache = RedisCache()
db = DatabaseManager()