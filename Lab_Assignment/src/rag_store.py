"""Vector store + chunk cache helpers shared across retrieval tasks."""



from __future__ import annotations



import json

import os

import re

from typing import Any



import numpy as np



from .rag_config import (

    CHUNKS_CACHE_PATH,

    EMBEDDING_BACKEND,

    EMBEDDING_DIM,

    EMBEDDING_MODEL,

    EMBEDDINGS_CACHE_PATH,

    EMBED_BATCH_SIZE,

    PROCESSED_DIR,

    WEAVIATE_API_KEY,

    WEAVIATE_COLLECTION,

    WEAVIATE_URL,

)



# Tránh crash threading khi load model trên Windows.

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

os.environ.setdefault("OMP_NUM_THREADS", "1")



_embedding_model = None

_weaviate_client = None

_local_index: dict[str, Any] | None = None

EMBEDDINGS_META_PATH = PROCESSED_DIR / "embeddings_meta.json"





def ensure_processed_dir() -> None:

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)





def _use_weaviate() -> bool:

    url = (WEAVIATE_URL or "").lower()

    key = (WEAVIATE_API_KEY or "").lower()

    if not url or not key or "xxx" in url or "xxx" in key:

        return False

    return True





def _normalize_rows(vectors: np.ndarray) -> np.ndarray:

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)

    return vectors / np.maximum(norms, 1e-9)





def _get_fastembed_model():

    from fastembed import TextEmbedding



    return TextEmbedding(EMBEDDING_MODEL)





class _EmbedWrapper:

    """API tương thích sentence-transformers cho task7 MMR."""



    def __init__(self, model):

        self._model = model



    def encode(self, texts, normalize_embeddings: bool = True, **kwargs):

        if isinstance(texts, str):

            texts = [texts]

            single = True

        else:

            single = False

        vectors = _normalize_rows(

            np.asarray(list(self._model.embed(texts)), dtype=np.float32)

        )

        if not normalize_embeddings:

            pass  # already normalized

        if single:

            return vectors[0]

        return vectors





def get_embedding_model():

    global _embedding_model

    if _embedding_model is None:

        if EMBEDDING_BACKEND == "fastembed":

            _embedding_model = _EmbedWrapper(_get_fastembed_model())

        else:

            from sentence_transformers import SentenceTransformer



            _embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    return _embedding_model





def embed_texts(texts: list[str], show_progress: bool = False) -> np.ndarray:

    if not texts:

        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    model = get_embedding_model()

    if isinstance(model, _EmbedWrapper):

        vectors = np.asarray(list(model._model.embed(texts)), dtype=np.float32)

        return _normalize_rows(vectors)

    vectors = model.encode(

        texts,

        batch_size=EMBED_BATCH_SIZE,

        show_progress_bar=show_progress,

        normalize_embeddings=True,

    )

    return np.asarray(vectors, dtype=np.float32)





def embed_query(query: str) -> list[float]:

    return embed_texts([query])[0].tolist()





def _save_embeddings_meta() -> None:

    ensure_processed_dir()

    EMBEDDINGS_META_PATH.write_text(

        json.dumps(

            {

                "model": EMBEDDING_MODEL,

                "dim": EMBEDDING_DIM,

                "backend": EMBEDDING_BACKEND,

            },

            ensure_ascii=False,

            indent=2,

        ),

        encoding="utf-8",

    )





def _embeddings_cache_valid() -> bool:

    if not EMBEDDINGS_CACHE_PATH.exists() or not CHUNKS_CACHE_PATH.exists():

        return False

    try:

        emb = np.load(EMBEDDINGS_CACHE_PATH, mmap_mode="r")

        if emb.shape[1] != EMBEDDING_DIM:

            return False

        if EMBEDDINGS_META_PATH.exists():

            meta = json.loads(EMBEDDINGS_META_PATH.read_text(encoding="utf-8"))

            return meta.get("model") == EMBEDDING_MODEL and meta.get("dim") == EMBEDDING_DIM

        return False

    except Exception:

        return False





def save_chunks_cache(chunks: list[dict]) -> None:

    """Lưu content + metadata (không nhúng embedding vào JSON)."""

    ensure_processed_dir()

    serializable = [

        {"content": c["content"], "metadata": c.get("metadata", {})} for c in chunks

    ]

    CHUNKS_CACHE_PATH.write_text(

        json.dumps(serializable, ensure_ascii=False, indent=2),

        encoding="utf-8",

    )





def save_embeddings_array(embeddings: np.ndarray) -> None:

    ensure_processed_dir()

    np.save(EMBEDDINGS_CACHE_PATH, embeddings.astype(np.float32))

    _save_embeddings_meta()





def load_chunks_cache() -> list[dict]:

    if not CHUNKS_CACHE_PATH.exists():

        return []

    return json.loads(CHUNKS_CACHE_PATH.read_text(encoding="utf-8"))





def load_embeddings_array() -> np.ndarray | None:

    if not _embeddings_cache_valid():

        return None

    return np.load(EMBEDDINGS_CACHE_PATH, mmap_mode="r")





def embed_and_cache_chunks(chunks: list[dict], *, show_progress: bool = True) -> np.ndarray:

    """Embed theo batch, ghi embeddings.npy để tiết kiệm RAM."""

    ensure_processed_dir()

    n = len(chunks)

    all_vectors: list[np.ndarray] = []



    for start in range(0, n, EMBED_BATCH_SIZE):

        end = min(start + EMBED_BATCH_SIZE, n)

        batch_texts = [chunks[i]["content"] for i in range(start, end)]

        all_vectors.append(embed_texts(batch_texts, show_progress=False))

        if show_progress:

            print(f"  embedded {end}/{n} chunks", flush=True)



    embeddings = np.vstack(all_vectors).astype(np.float32)

    np.save(EMBEDDINGS_CACHE_PATH, embeddings)

    save_chunks_cache(chunks)

    _save_embeddings_meta()

    return embeddings





def extract_article(content: str) -> str | None:

    match = re.search(r"\*\*Điều\s+(\d+)[^*]*\*\*", content)

    if match:

        return f"Điều {match.group(1)}"

    return None





def _connect_weaviate():

    global _weaviate_client

    if _weaviate_client is not None:

        return _weaviate_client



    import weaviate



    if _use_weaviate():

        _weaviate_client = weaviate.connect_to_weaviate_cloud(

            cluster_url=WEAVIATE_URL,

            auth_credentials=weaviate.auth.AuthApiKey(WEAVIATE_API_KEY),

        )

    else:

        _weaviate_client = weaviate.connect_to_local()



    return _weaviate_client





def _get_weaviate_collection():

    from weaviate.classes.config import Configure, DataType, Property



    client = _connect_weaviate()

    if client.collections.exists(WEAVIATE_COLLECTION):

        return client.collections.get(WEAVIATE_COLLECTION)



    return client.collections.create(

        name=WEAVIATE_COLLECTION,

        vectorizer_config=Configure.Vectorizer.none(),

        properties=[

            Property(name="content", data_type=DataType.TEXT),

            Property(name="source", data_type=DataType.TEXT),

            Property(name="doc_type", data_type=DataType.TEXT),

            Property(name="chunk_index", data_type=DataType.INT),

            Property(name="article", data_type=DataType.TEXT),

        ],

    )





def _build_local_index(chunks: list[dict], embeddings: np.ndarray) -> None:

    global _local_index

    _local_index = {

        "chunks": [{"content": c["content"], "metadata": c.get("metadata", {})} for c in chunks],

        "embeddings": np.asarray(embeddings, dtype=np.float32),

    }





def _load_local_index() -> dict[str, Any]:

    global _local_index

    if _local_index is not None:

        return _local_index



    chunks = load_chunks_cache()

    embeddings = load_embeddings_array()

    if not chunks or embeddings is None:

        return {"chunks": [], "embeddings": np.empty((0, EMBEDDING_DIM), dtype=np.float32)}



    _local_index = {"chunks": chunks, "embeddings": np.asarray(embeddings, dtype=np.float32)}

    return _local_index





def index_chunks(chunks: list[dict], embeddings: np.ndarray, *, recreate: bool = False) -> str:

    if not _use_weaviate():

        _build_local_index(chunks, embeddings)

        return "local"



    try:

        collection = _get_weaviate_collection()

        if recreate:

            client = _connect_weaviate()

            if client.collections.exists(WEAVIATE_COLLECTION):

                client.collections.delete(WEAVIATE_COLLECTION)

            collection = _get_weaviate_collection()



        with collection.batch.dynamic() as batch:

            for i, chunk in enumerate(chunks):

                meta = chunk.get("metadata", {})

                article = meta.get("article") or extract_article(chunk["content"]) or ""

                batch.add_object(

                    properties={

                        "content": chunk["content"],

                        "source": meta.get("source", ""),

                        "doc_type": meta.get("type", meta.get("doc_type", "")),

                        "chunk_index": int(meta.get("chunk_index", 0)),

                        "article": article,

                    },

                    vector=embeddings[i].tolist(),

                )

        return "weaviate"

    except Exception:

        _build_local_index(chunks, embeddings)

        return "local"





def _format_result(content: str, score: float, props: dict) -> dict:

    metadata = {

        "source": props.get("source", ""),

        "type": props.get("doc_type", props.get("type", "")),

        "chunk_index": props.get("chunk_index", 0),

    }

    if props.get("article"):

        metadata["article"] = props["article"]

    return {"content": content, "score": float(score), "metadata": metadata}





def vector_search(query: str, top_k: int = 10) -> list[dict]:

    query_vec = embed_query(query)



    if _use_weaviate():

        try:

            from weaviate.classes.query import MetadataQuery



            collection = _get_weaviate_collection()

            response = collection.query.near_vector(

                near_vector=query_vec,

                limit=top_k,

                return_metadata=MetadataQuery(distance=True),

            )



            results = []

            for obj in response.objects:

                distance = obj.metadata.distance if obj.metadata else 0.0

                score = 1.0 - float(distance)

                results.append(

                    _format_result(obj.properties["content"], score, obj.properties)

                )

            return sorted(results, key=lambda x: x["score"], reverse=True)

        except Exception:

            pass



    index = _load_local_index()

    if len(index["chunks"]) == 0:

        return []



    query_arr = np.asarray(query_vec, dtype=np.float32)

    scores = index["embeddings"] @ query_arr

    top_indices = np.argsort(scores)[::-1][:top_k]



    results = []

    for idx in top_indices:

        chunk = index["chunks"][int(idx)]

        meta = chunk.get("metadata", {})

        results.append(

            _format_result(

                chunk["content"],

                float(scores[idx]),

                {

                    "source": meta.get("source", ""),

                    "doc_type": meta.get("type", ""),

                    "chunk_index": meta.get("chunk_index", 0),

                    "article": meta.get("article", ""),

                },

            )

        )

    return results





def close_weaviate() -> None:

    global _weaviate_client

    if _weaviate_client is not None:

        _weaviate_client.close()

        _weaviate_client = None


