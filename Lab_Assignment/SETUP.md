# RAG Pipeline Day08 → Multi-Agent (copied)

## Cài đặt nhanh

```bash
cd "Improve Agent Day08"
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env          # điền API keys
```

## Chạy

```bash
# Chat UI (Supervisor-Workers)
streamlit run app.py

# Re-index nếu cần (đã có data/processed/)
python -m src.task4_chunking_indexing

# Test
pytest tests/test_individual.py -v
```

## Cấu trúc chính

| Thư mục / file | Mô tả |
|----------------|--------|
| `src/agents/` | Supervisor + Legal / News / Answer workers |
| `src/task4-10` | RAG pipeline (chunking, retrieval, generation) |
| `src/rag_store.py` | Embedding + vector index |
| `data/standardized/` | Knowledge base markdown |
| `data/processed/` | chunks.json + embeddings.npy |
| `app.py` | Streamlit chat |

## Lưu ý

- **Không commit** file `.env` (chứa API keys).
- `data/processed/` đã copy sẵn — không cần chạy lại Task 4 trừ khi đổi embedding model.
