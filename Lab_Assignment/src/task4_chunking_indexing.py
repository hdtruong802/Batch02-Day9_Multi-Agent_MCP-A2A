"""

Task 4 — Chunking & Indexing vào Vector Store.



Chunking: RecursiveCharacterTextSplitter (800/120) với separators ưu tiên

**Điều/**Chương vì luật VN dùng bold thay vì markdown heading.



Embedding: paraphrase-multilingual-MiniLM-L12-v2 (384 dim) via fastembed/ONNX.



Vector store: Weaviate (hybrid-ready); fallback local numpy nếu Weaviate chưa chạy.

"""



from langchain_text_splitters import RecursiveCharacterTextSplitter



from .rag_config import (

    CHUNK_OVERLAP,

    CHUNK_SEPARATORS,

    CHUNK_SIZE,

    CHUNKING_METHOD,

    EMBEDDING_DIM,

    EMBEDDING_MODEL,

    STANDARDIZED_DIR,

    VECTOR_STORE,

)

from .rag_store import embed_and_cache_chunks, extract_article, index_chunks



__all__ = [

    "CHUNK_SIZE",

    "CHUNK_OVERLAP",

    "CHUNKING_METHOD",

    "EMBEDDING_MODEL",

    "EMBEDDING_DIM",

    "VECTOR_STORE",

    "load_documents",

    "chunk_documents",

    "embed_chunks",

    "index_to_vectorstore",

    "run_pipeline",

]





def load_documents() -> list[dict]:

    """Đọc toàn bộ markdown files từ data/standardized/."""

    documents = []

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):

        content = md_file.read_text(encoding="utf-8")

        rel = md_file.relative_to(STANDARDIZED_DIR)

        parent = rel.parts[0] if len(rel.parts) > 1 else ""

        if parent == "legal" or "luat" in md_file.name.lower() or "nghi-dinh" in md_file.name.lower():

            doc_type = "legal"

        elif parent == "news" or md_file.name.startswith("news_"):

            doc_type = "news"

        else:

            doc_type = "legal" if any(

                kw in md_file.name.lower() for kw in ("luat", "nghi-dinh", "bo-luat")

            ) else "news"



        documents.append(

            {

                "content": content,

                "metadata": {

                    "source": md_file.name,

                    "type": doc_type,

                    "path": str(rel),

                },

            }

        )

    return documents





def chunk_documents(documents: list[dict]) -> list[dict]:

    """Chunk documents theo RecursiveCharacterTextSplitter."""

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=CHUNK_SIZE,

        chunk_overlap=CHUNK_OVERLAP,

        separators=CHUNK_SEPARATORS,

        length_function=len,

    )



    chunks: list[dict] = []

    for doc in documents:

        splits = splitter.split_text(doc["content"])

        for i, chunk_text in enumerate(splits):

            article = extract_article(chunk_text)

            metadata = {**doc["metadata"], "chunk_index": i}

            if article:

                metadata["article"] = article

            chunks.append({"content": chunk_text, "metadata": metadata})

    return chunks





def embed_chunks(chunks: list[dict]) -> tuple[list[dict], "np.ndarray"]:

    """Embed toàn bộ chunks bằng BAAI/bge-m3, lưu embeddings.npy."""

    embeddings = embed_and_cache_chunks(chunks, show_progress=True)

    return chunks, embeddings





def index_to_vectorstore(

    chunks: list[dict], embeddings, *, recreate: bool = True

) -> str:

    """Lưu chunks vào Weaviate (hoặc local fallback)."""

    return index_chunks(chunks, embeddings, recreate=recreate)





def run_pipeline():

    """Chạy toàn bộ pipeline: load → chunk → embed → index."""

    print("=" * 50)

    print("Task 4: Chunking & Indexing")

    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")

    print(f"  Vector Store: {VECTOR_STORE}")

    print("=" * 50)



    docs = load_documents()

    print(f"\n[OK] Loaded {len(docs)} documents")



    chunks = chunk_documents(docs)

    print(f"[OK] Created {len(chunks)} chunks")



    chunks, embeddings = embed_chunks(chunks)

    print(f"[OK] Embedded {len(chunks)} chunks")



    backend = index_to_vectorstore(chunks, embeddings)

    print(f"[OK] Indexed to {backend}")





if __name__ == "__main__":

    run_pipeline()


