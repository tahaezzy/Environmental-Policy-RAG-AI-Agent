# src/pdf_processing.py
from pdf2image import convert_from_path
from src.config import *
import latex2mathml.converter

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