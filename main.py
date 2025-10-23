'''
Overall Program Structure
=================================================================
greenpolicyai/
│
├── main.py
│
├── src/
│   ├── __init__.py
│   ├── config.py
|   |       └── IMPORTS: os, json, torch, pytesseract, logging, SentenceTransformer, CrossEncoder, Client as OllamaClient, Path, 
|   |       |            psutil, BM25Okapi, numpy as np, time, datetime, re, fitz, pdfplumber, hashlib, List, Dict, Any, Optional, 
|   |       |            Tuple, tqdm, gc
|   |       └── get_system_constants()
│   ├── utils.py        
|   |       └── IMPORTS: config.py/*, database.py/cache
|   |       └── get_text_hash()
|   |       └── check_memory_usage()
|   |       └── memory_cleanup()
|   |       └── get_embedding()
|   |       └── get_batch_embedding()
|   |       └── clean_text()
│   ├── database.py
|   |       └── Class: RedisCache
|   |       └── Class: DatabaseManager 
│   ├── pdf_processing.py     
|   |       └── extract_pdf_text()
|   |       └── split_pdf_into_sections()
|   |       └── split_headers()
|   |       └── hybrid_chunking()
|   |       └── handle_math()
|   |       └── extract_pdf_text() 
│   ├── compliance_checker.py 
|   |       └── extract_facts_to_json(
|   |       └── extract_entities_relations()
|   |       └── build_reg_graph()
|   |       └── graph_enhanced_retrieval()
|   |       └── _extract_json_array_from_text()
|   |       └── split_pdf_into_sections()
|   |       └── check_compliance_for_section_with_regex()
|   |       └── check_file_compliance()
|   |       └── check_project_compliance()
|   |       └── assign_confidence()
│   ├── visualization.py      
|   |       └── _to_display_text()
|   |       └── flatten_compliance_logs()
|   |       └── print_compliance_summary_table()
|   |       └── show_violation_detail()
|   |       └── visualize_reg_graph()
│   └── rag.py                
|           └── ingest_documents(), 
|           └── hybrid_search(),
|           └──  ask_rag_streaming()
│
└── logs/
    └── compliance_logs.jsonl (created dynamically)
=================================================================
'''

from src.config import *
from src.utils import *
from src.database import *
from src.compliance_checker import check_file_compliance, check_compliance_for_section_with_regex, check_project_compliance, extract_pdf_text, split_pdf_into_sections
from src.visualization import visualize_reg_graph, print_compliance_summary_table
from src.rag import *

'''

# === DEBUGGING: automatic freeze detection ===
import faulthandler
import sys

# Enable faulthandler globally so crashes and timeouts show a stack trace.
faulthandler.enable(all_threads=True)

# Dump stack traces every 10 seconds while running — helps diagnose hangs.
faulthandler.dump_traceback_later(timeout=10, repeat=True, file=sys.stderr)
print("Debugging mode active: The program will dump stack traces every 10s if it freezes.")

'''

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
    
    print("\nWelcome to GreenPolicyAI.")
    print("To perform a compliance check, please type '!COMPLIANCE!'")
    print("Type 'exit' or 'quit' to end the program.")
    print("-" * 50)
    
    # Main REPL with streaming
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

            # Initialize common variables to avoid UnboundLocalError later
            response = None
            sources: List[Dict[str, Any]] = []
            start_time = time.time()

            # Use compliance checker if requested 
            if user_input in ["!COMPLIANCE!"]:
                logging.info("Entered compliance check command.")
                print("Beginning Compliance Check...")
                # capture the batch report in case you want to inspect it
                try:
                    aggregated_report = check_project_compliance()
                    logging.info("Exited compliance check normally.")
                except Exception as e:
                    logging.exception(f"Compliance batch failed: {e}")
                    aggregated_report = {"files_checked": 0, "total_violations": 0}
                end_time = time.time()
                # minimal summary for the user
                print(f"Compliance batch completed: files_checked={aggregated_report.get('files_checked', 0)}, total_violations={aggregated_report.get('total_violations', 0)}")
                visualize_reg_graph() ## Creat regulations graph visualization
                print_compliance_summary_table() ## Create compliance table

            else:
                # Regular RAG query path
                response, sources = ask_rag_streaming(user_input)
                end_time = time.time()

            # Show sources (safe because 'sources' is always defined)
            if sources:
                unique_sources = list({s.get('source_file') for s in sources if isinstance(s, dict) and s.get('source_file')})
                print(f"\nSources: {', '.join(unique_sources)}")
                print(f"Response time: {end_time - start_time:.2f}s")
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
            logging.error(f"Main loop error: ")
            print(f"An error occurred: {e}")
            print("Check logs for details.")

if __name__ == "__main__":
    main()