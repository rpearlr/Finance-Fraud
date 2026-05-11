import os
import sys
import time
import logging
import hashlib
import pickle
from pathlib import Path
from typing import Optional
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rag")

# ── Config ─────────────────────────────────────────────────────────────────────
DOCS_DIR         = Path("data/docs")
INDEX_CACHE_DIR  = Path("data/rag_cache")    # cached PDF text extractions
FAISS_INDEX_PATH = Path("data/faiss_index")  # persisted FAISS index + chunks

for d in [DOCS_DIR, INDEX_CACHE_DIR, FAISS_INDEX_PATH]:
    d.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE    = 500   # characters per chunk
CHUNK_OVERLAP = 50    # overlap between chunks to preserve context
TOP_K         = 4     # chunks to retrieve per query

# Azure OpenAI — only used for answer generation (gpt-4o-mini)
AZURE_OPENAI_ENDPOINT    = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY         = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
CHAT_DEPLOYMENT          = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini")



# ── Module-level state ─────────────────────────────────────────────────────────
_embed_model  = None   # HuggingFace SentenceTransformer, loaded once per session
_faiss_index  = None   # faiss.IndexFlatIP, loaded from disk on first query
_faiss_chunks = []     # list of chunk dicts, parallel to FAISS vectors

# ── Source type mapping ────────────────────────────────────────────────────────
SOURCE_TYPE_MAP = {
    "sebi"  : "SEBI",
    "aif"   : "SEBI",
    "rbi"   : "RBI",
    "basel" : "Basel",
    "fema"  : "FEMA",
    "irdai" : "IRDAI",
}


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — PDF TEXT EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract text from a PDF file.
    Extract text from a PDF file using pypdf.
    Results are cached by MD5 hash — unchanged PDFs are never re-extracted.
    """
    file_hash  = hashlib.md5(pdf_path.read_bytes()).hexdigest()
    cache_path = INDEX_CACHE_DIR / f"{pdf_path.stem}_{file_hash}.txt"

    if cache_path.exists():
        log.info(f"  Using cached extraction for {pdf_path.name}")
        return cache_path.read_text(encoding="utf-8")

    text = ""
    try:
        from pypdf import PdfReader
        log.info(f"  Extracting {pdf_path.name} via pypdf...")
        reader = PdfReader(str(pdf_path))
        pages  = [page.extract_text() or "" for page in reader.pages]
        text   = "\n\n".join(pages)
        log.info(f"  Extracted {len(text):,} chars from {len(pages)} pages")
    except Exception as e:
        log.error(f"  pypdf extraction failed: {e}")
        raise

    if not text.strip():
        log.warning(f"  No text found in {pdf_path.name}")
        return ""

    cache_path.write_text(text, encoding="utf-8")
    return text


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — CHUNKING
# ══════════════════════════════════════════════════════════════════════════════

def chunk_text(text: str, source_name: str, source_type: str) -> list[dict]:
    """
    Split text into overlapping chunks with metadata.
    Each chunk dict contains content + source info used for retrieval and citation.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    raw_chunks = splitter.split_text(text)
    chunks     = []

    for i, chunk_content in enumerate(raw_chunks):
        chunk_content = chunk_content.strip()
        if len(chunk_content) < 50:   # skip tiny fragments
            continue

        chunk_id = hashlib.md5(
            f"{source_name}_{i}_{chunk_content[:50]}".encode()
        ).hexdigest()

        chunks.append({
            "id"         : chunk_id,
            "content"    : chunk_content,
            "source"     : source_name,
            "source_type": source_type,
            "chunk_index": i,
            "chunk_total": len(raw_chunks),
        })

    avg_len = sum(len(c["content"]) for c in chunks) // max(len(chunks), 1)
    log.info(f"  Chunked into {len(chunks)} pieces (avg {avg_len} chars each)")
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — EMBEDDING (HuggingFace, free, local)
# ══════════════════════════════════════════════════════════════════════════════

def warmup_embed_model() -> None:
    """
    Pre-load the SentenceTransformer into memory.
    Call this once at application startup so the first user query is fast.
    """
    from sentence_transformers import SentenceTransformer

    global _embed_model
    if _embed_model is None:
        log.info("  Loading HuggingFace embedding model (all-MiniLM-L6-v2)...")
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        log.info(f"  Embedding model ready — dim={_embed_model.get_sentence_embedding_dimension()}")
    else:
        log.info("  Embedding model already in memory — skipping load")


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Embed texts using HuggingFace all-MiniLM-L6-v2.
    - Completely free, runs locally, no API key needed
    - Downloads ~90MB on first run, cached by sentence-transformers after that
    - Output: 384-dimensional L2-normalised vectors
    - Batch size 64 — no rate limits since it's local
    """
    global _embed_model
    if _embed_model is None:
        warmup_embed_model()  # fallback if called before startup hook

    all_embeddings = []
    batch_size     = 64

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        log.info(
            f"  Embedding batch {i//batch_size + 1}/"
            f"{(len(texts) - 1)//batch_size + 1} ({len(batch)} chunks)..."
        )
        # normalize_embeddings=True fuses L2 normalisation into the encode pass
        # so we don't need a separate faiss.normalize_L2() call on the result
        embeddings = _embed_model.encode(
            batch,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        all_embeddings.extend(embeddings.tolist())

    log.info(f"  Generated {len(all_embeddings)} embeddings (dim={len(all_embeddings[0])})")
    return all_embeddings


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — FAISS LOCAL VECTOR STORE
# ══════════════════════════════════════════════════════════════════════════════

def init_faiss_index(dim: int = 384) -> None:
    """Create a fresh in-memory FAISS index. Called at start of indexing."""
    import faiss
    global _faiss_index, _faiss_chunks
    _faiss_index  = faiss.IndexFlatIP(dim)   # inner product on L2-normalized = cosine similarity
    _faiss_chunks = []
    log.info(f"  Created new FAISS index (dim={dim})")


def upload_chunks_to_search(chunks: list[dict], embeddings: list[list[float]]) -> None:
    """
    Add chunks + their embeddings to the FAISS index, then persist to disk.
    Safe to call multiple times per indexing run — appends to the current index.
    Embeddings are expected to already be L2-normalised (done by get_embeddings).
    """
    import faiss
    import numpy as np

    global _faiss_index, _faiss_chunks

    vectors = np.array(embeddings, dtype="float32")
    # Vectors are already L2-normalised by get_embeddings(normalize_embeddings=True)
    # so inner product == cosine similarity without an extra normalisation step.

    if _faiss_index is None:
        init_faiss_index(dim=vectors.shape[1])

    _faiss_index.add(vectors)
    _faiss_chunks.extend(chunks)

    log.info(f"  Added {len(chunks)} chunks — index total: {_faiss_index.ntotal}")

    # Persist to disk so the index survives between Python sessions
    faiss.write_index(_faiss_index, str(FAISS_INDEX_PATH / "index.faiss"))
    with open(FAISS_INDEX_PATH / "chunks.pkl", "wb") as f:
        pickle.dump(_faiss_chunks, f)

    log.info(f"  Saved FAISS index -> {FAISS_INDEX_PATH}/")


def load_faiss_index() -> bool:
    """
    Load the FAISS index from disk into memory.
    Returns True if successful, False if index file doesn't exist yet.
    Skips loading if already in memory.
    """
    import faiss

    global _faiss_index, _faiss_chunks

    index_file  = FAISS_INDEX_PATH / "index.faiss"
    chunks_file = FAISS_INDEX_PATH / "chunks.pkl"

    if not index_file.exists() or not chunks_file.exists():
        return False

    if _faiss_index is not None:
        return True   # already loaded this session

    log.info("Loading FAISS index from disk...")
    _faiss_index = faiss.read_index(str(index_file))
    with open(chunks_file, "rb") as f:
        _faiss_chunks = pickle.load(f)
    log.info(f"Loaded FAISS index — {_faiss_index.ntotal} chunks")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_chunks(query: str, source_filter: Optional[str] = None, top_k: int = TOP_K) -> list[dict]:
    """
    Embed the query, search FAISS for nearest neighbours,
    optionally filter by source type (SEBI / RBI / Basel / FEMA),
    return top-k most relevant chunks.
    """
    import faiss
    import numpy as np

    if not load_faiss_index():
        raise FileNotFoundError(
            "FAISS index not found. Run: python agents/rag_pipeline.py index"
        )

    # Embed query with the same model used during indexing
    # Vectors come out L2-normalised from get_embeddings(), no extra step needed
    query_embedding = get_embeddings([query])[0]
    vector = np.array([query_embedding], dtype="float32")

    # Fetch extra results when filtering so we still get top_k after filter
    k = min(top_k * 3 if source_filter else top_k, _faiss_index.ntotal)
    scores, indices = _faiss_index.search(vector, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        chunk = _faiss_chunks[idx]

        if source_filter and chunk.get("source_type") != source_filter:
            continue

        results.append({
            "content"        : chunk["content"],
            "source"         : chunk["source"],
            "source_type"    : chunk["source_type"],
            "relevance_score": round(float(score), 4),
        })

        if len(results) == top_k:
            break

    log.info(f"Retrieved {len(results)} chunks for: '{query[:60]}'")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 6 — ANSWER GENERATION (Azure gpt-4o-mini)
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a regulatory compliance expert for Indian financial markets.
You answer questions strictly based on the provided regulatory document excerpts.
Always cite which document your answer comes from.
If the provided excerpts do not contain enough information to answer confidently, say so clearly.
Never make up regulatory details — accuracy is critical in financial compliance.

Format your answers as:
1. A direct answer to the question (2-4 sentences)
2. Key points as bullet points if applicable
3. Source citation at the end: [Source: document name]
"""


def generate_answer(query: str, chunks: list[dict]) -> dict:
    """
    Pass retrieved chunks as context to gpt-4o-mini and generate a grounded answer.
    Only this function uses Azure — everything before it is free and local.
    Degrades gracefully if Azure OpenAI is not configured.
    """
    if not chunks:
        return {
            "answer"     : "No relevant regulatory documents found for this query. "
                           "Try rephrasing or check that documents have been indexed.",
            "sources"    : [],
            "chunks_used": 0,
        }

    # Graceful degradation — return raw chunks if Azure not configured
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_KEY:
        return {
            "answer"     : "Azure OpenAI not configured. Raw retrieved chunks:\n\n" +
                           "\n---\n".join(c["content"] for c in chunks),
            "sources"    : list({f"{c['source']} ({c['source_type']})" for c in chunks}),
            "chunks_used": len(chunks),
            "tokens_used": 0,
        }

    # Build context from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Excerpt {i} — {chunk['source']} ({chunk['source_type']}, "
            f"relevance: {chunk['relevance_score']})]\n{chunk['content']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    user_message = (
        f"Answer the following question using ONLY the regulatory excerpts below.\n\n"
        f"Question: {query}\n\n"
        f"Regulatory excerpts:\n{context}"
    )

    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_deployment=CHAT_DEPLOYMENT,
        temperature=0.1,
        max_tokens=600,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", user_message),
    ])

    prompt_value = prompt.invoke({})
    response = llm.invoke(prompt_value)
    
    answer = response.content
    tokens = 0
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        tokens = response.usage_metadata.get("total_tokens", 0)
    
    sources = list({f"{c['source']} ({c['source_type']})" for c in chunks})

    return {
        "answer"     : answer,
        "sources"    : sources,
        "chunks_used": len(chunks),
        "model"      : "gpt-4o-mini",
        "tokens_used": tokens,
    }


# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENT ASSISTANT AGENT — imported by FastAPI /agent/query
# ══════════════════════════════════════════════════════════════════════════════

class DocumentAssistantAgent:
    """
    Drop-in agent for FastAPI. Import and call .run(question).

    Usage in backend/main.py:
        from agents.rag_pipeline import DocumentAssistantAgent
        rag_agent = DocumentAssistantAgent()

        result = rag_agent.run(
            "What are SEBI AIF disclosure requirements?",
            source_filter="SEBI"
        )
        print(result["answer"])
    """

    def __init__(self):
        self._ready = self._check_ready()

    def _check_ready(self) -> bool:
        index_file = FAISS_INDEX_PATH / "index.faiss"
        if not index_file.exists():
            log.warning("FAISS index not found — run: python agents/rag_pipeline.py index")
            return False
        if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_KEY:
            log.warning("Azure OpenAI not configured — answers will be raw chunk text")
        return True

    def run(self, question: str, source_filter: Optional[str] = None, top_k: int = TOP_K) -> dict:
        """
        Full RAG pipeline: retrieve -> generate -> return structured response.

        Returns:
            answer       : str  — grounded answer from gpt-4o-mini
            sources      : list — document names used
            chunks_used  : int  — number of chunks retrieved
            tokens_used  : int  — Azure OpenAI tokens consumed (0 if not configured)
            latency_ms   : int  — total pipeline time in milliseconds
            agent        : str  — always "rag_agent"
        """
        if not self._ready:
            self._ready = self._check_ready()

        if not self._ready:
            return {
                "answer"     : "RAG Agent not ready. Run: python agents/rag_pipeline.py index",
                "sources"    : [],
                "chunks_used": 0,
                "agent"      : "rag_agent",
            }

        try:
            start  = time.time()
            chunks = retrieve_chunks(question, source_filter, top_k=top_k)
            result = generate_answer(question, chunks)
            result["latency_ms"] = int((time.time() - start) * 1000)
            result["agent"]      = "rag_agent"
            return result

        except Exception as e:
            log.error(f"RAG Agent error: {e}")
            return {
                "answer"     : f"RAG Agent error: {str(e)}",
                "sources"    : [],
                "chunks_used": 0,
                "agent"      : "rag_agent",
            }


# ══════════════════════════════════════════════════════════════════════════════
# INDEXING PIPELINE — run once, or whenever you add new PDFs
# ══════════════════════════════════════════════════════════════════════════════

def infer_source_type(filename: str) -> str:
    name_lower = filename.lower()
    for keyword, label in SOURCE_TYPE_MAP.items():
        if keyword in name_lower:
            return label
    return "General"


def run_indexing_pipeline() -> None:
    """
    Full pipeline: scan data/docs/ -> extract -> chunk -> embed -> index.
    - Safe to re-run: unchanged PDFs skip extraction via MD5 cache
    - Rebuilds FAISS index from scratch each run to avoid duplicate chunks
    """
    pdf_files = list(DOCS_DIR.glob("*.pdf"))

    if not pdf_files:
        log.error(f"No PDFs found in {DOCS_DIR}/")
        log.error("Place regulatory PDFs in data/docs/ and re-run")
        sys.exit(1)

    log.info(f"Found {len(pdf_files)} PDFs: {[f.name for f in pdf_files]}")
    log.info("Rebuilding FAISS index from scratch...")

    # Reset so re-runs don't duplicate chunks
    init_faiss_index(dim=384)

    total_chunks = 0

    for pdf_path in pdf_files:
        log.info(f"\nProcessing: {pdf_path.name}")
        source_type = infer_source_type(pdf_path.name)
        log.info(f"  Source type: {source_type}")

        # Stage 1 — Extract
        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            log.warning(f"  Skipping — no text extracted")
            continue

        # Stage 2 — Chunk
        chunks = chunk_text(text, source_name=pdf_path.stem, source_type=source_type)
        if not chunks:
            log.warning(f"  Skipping — no chunks produced")
            continue

        # Stage 3 — Embed (local, free)
        embeddings = get_embeddings([c["content"] for c in chunks])

        # Stage 4 — Store in FAISS + persist to disk
        upload_chunks_to_search(chunks, embeddings)

        total_chunks += len(chunks)
        log.info(f"  Done — {len(chunks)} chunks indexed")

    log.info(f"\nIndexing complete")
    log.info(f"  Total chunks : {total_chunks}")
    log.info(f"  Documents    : {len(pdf_files)}")
    log.info(f"  Index saved  : {FAISS_INDEX_PATH}/")
    log.info(f"\nTest with:")
    log.info(f"  python agents/rag_pipeline.py query \"What are SEBI AIF disclosure requirements?\"")


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python agents/rag_pipeline.py index")
        print("  python agents/rag_pipeline.py query \"What are SEBI AIF disclosure requirements?\"")
        print("  python agents/rag_pipeline.py query \"Basel III capital ratios\" --source Basel")
        sys.exit(1)

    command = sys.argv[1]

    if command == "index":
        run_indexing_pipeline()

    elif command == "query":
        if len(sys.argv) < 3:
            print("Provide a question: python agents/rag_pipeline.py query \"your question\"")
            sys.exit(1)

        question      = sys.argv[2]
        source_filter = None

        if "--source" in sys.argv:
            idx = sys.argv.index("--source")
            if idx + 1 < len(sys.argv):
                source_filter = sys.argv[idx + 1]

        log.info(f"Query: '{question}'")
        if source_filter:
            log.info(f"Source filter: {source_filter}")

        agent  = DocumentAssistantAgent()
        result = agent.run(question, source_filter=source_filter)

        print("\n" + "=" * 60)
        print("ANSWER")
        print("=" * 60)
        print(result["answer"])
        print("\nSources    :", ", ".join(result["sources"]) or "none")
        print(f"Chunks used: {result['chunks_used']}")
        print(f"Tokens used: {result.get('tokens_used', 'N/A')}")
        print(f"Latency    : {result.get('latency_ms', 'N/A')}ms")
        print("=" * 60)

    else:
        print(f"Unknown command '{command}'. Use 'index' or 'query'.")
        sys.exit(1)