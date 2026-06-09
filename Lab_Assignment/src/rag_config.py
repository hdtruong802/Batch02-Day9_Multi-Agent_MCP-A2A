"""Shared configuration for RAG pipeline (Tasks 4–10)."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
STANDARDIZED_DIR = PROJECT_ROOT / "data" / "standardized"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# RecursiveCharacterTextSplitter — 800 chars phù hợp tiếng Việt (1 Điều luật),
# overlap 120 (~15%) giữ ngữ cảnh ở biên chunk cho citation.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
CHUNKING_METHOD = "recursive"

# Ưu tiên tách theo Điều/Chương (luật VN dùng **bold** thay vì # heading).
CHUNK_SEPARATORS = [
    "\n\n**Điều",
    "\n\n**Chương",
    "\n\n# ",
    "\n\n## ",
    "\n\n",
    "\n",
    ". ",
    " ",
    "",
]

# fastembed + ONNX — ổn định trên Windows (tránh PyTorch access violation với bge-m3).
# Multilingual MiniLM, phù hợp tiếng Việt; nhẹ hơn bge-m3.
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384
EMBEDDING_BACKEND = "fastembed"

VECTOR_STORE = "weaviate"
WEAVIATE_COLLECTION = "DrugLawDocs"
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "")

CHUNKS_CACHE_PATH = PROCESSED_DIR / "chunks.json"
EMBEDDINGS_CACHE_PATH = PROCESSED_DIR / "embeddings.npy"
PAGEINDEX_DOC_IDS_PATH = PROCESSED_DIR / "pageindex_doc_ids.json"
EMBED_BATCH_SIZE = 32

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
JINA_API_KEY = os.getenv("JINA_API_KEY", "")
PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
