# src/utils.py
from src.database import cache
from src.config import *

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('GreenPolicyAI.log', mode='a'),
        logging.StreamHandler()
    ]
)

def get_text_hash(text: str) -> str:
    """Compute SHA256 hash of text."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def check_memory_usage() -> bool:
    """Check if memory usage is acceptable."""
    memory = psutil.virtual_memory()
    if memory.percent > 85:
        logging.warning(f"High memory usage: {memory.percent}%")
        return False
    return True

def get_embedding_cached(text: str) -> Optional[np.ndarray]:
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

def memory_cleanup() -> None:
    """Perform memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logging.info("Memory cleanup performed")

def get_embedding(text: str, embed_model: SentenceTransformer, embedding_cache: dict, cache_file: str) -> list:
    """Retrieve or compute embedding for single text."""
    h = get_text_hash(text)
    if h in embedding_cache:
        return embedding_cache[h]
    emb = embed_model.encode(text).tolist()
    embedding_cache[h] = emb
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(embedding_cache, f)
    return emb

def get_batch_embeddings(texts: list[str], embed_model: SentenceTransformer, embedding_cache: dict, cache_file: str, batch_size: int) -> list[list[float]]:
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
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(embedding_cache, f)
    except Exception as e:
        logging.error(f"Batch embedding error: {e}")
        return [[]] * len(texts)
    return embeddings

