# src/rag.py
from src.config import *
from src.utils import *
from src.pdf_processing import hybrid_chunking, extract_pdf_text
from src.database import *

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
            combined_score = 0.65 * result['similarity_score'] + 0.35 * bm25_scores[i]
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