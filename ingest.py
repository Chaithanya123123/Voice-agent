"""Document ingestion pipeline with incremental upserts + rich payloads.

Fixes vs original:
- HierarchicalChunker(max_characters=...) was silently ignored (pydantic
  extra='ignore') so settings.chunk_size had no effect. Now we chunk
  hierarchically, then post-split long chunks with character overlap.
- Previously deleted the whole collection on every run. Now upserts are
  per-document (delete-by-source filter + deterministic UUIDs).
- Payload now carries source/page_no/char_start for the chunk-viewer UI.
"""
from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker import HierarchicalChunker
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

from config import settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def split_with_overlap(text: str, size: int, overlap: int) -> list[tuple[str, int]]:
    """Split text into overlapping windows. Returns (window, char_start)."""
    text = text.strip()
    if not text or size <= 0:
        return [(text, 0)] if text else []
    if len(text) <= size:
        return [(text, 0)]
    step = max(1, size - max(0, overlap))
    out: list[tuple[str, int]] = []
    for start in range(0, len(text), step):
        window = text[start : start + size]
        if not window.strip():
            continue
        out.append((window, start))
        if start + size >= len(text):
            break
    return out


def load_document(source: str):
    """Parse a PDF with Docling. OCR toggle comes from settings.do_ocr."""
    print(f"  Parsing: {source}")
    pipeline = PdfPipelineOptions(do_ocr=settings.do_ocr)
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline)}
    )
    result = converter.convert(source)
    return result.document


def _page_of(doc_item) -> int | None:
    try:
        prov = getattr(doc_item, "prov", None)
        if prov:
            for p in prov:
                page = getattr(p, "page_no", None)
                if page is not None:
                    return int(page)
    except Exception:
        pass
    return None


def convert_chunk(doc_chunk, source: str) -> dict:
    """Flatten a Docling DocChunk -> dict with page + headings."""
    headings = doc_chunk.meta.headings or []
    content = (doc_chunk.text or "").strip()
    breadcrumb = " > ".join(headings)
    chunk_text = f"{breadcrumb}\n\n{content}" if breadcrumb else content
    pages: list[int] = []
    try:
        for item in (doc_chunk.meta.doc_items or []):
            pg = _page_of(item)
            if pg is not None and pg not in pages:
                pages.append(pg)
    except Exception:
        pass
    return {
        "source": source,
        "headings": headings,
        "content": content,
        "chunk_text": chunk_text,
        "page_no": pages[0] if pages else None,
        "pages": pages,
    }


def parse_and_chunk(
    pdf_path: Path, chunk_size: int | None = None, chunk_overlap: int | None = None
) -> list[dict]:
    """Parse one PDF -> list of overlapping chunk dicts."""
    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
    doc = load_document(str(pdf_path))
    chunker = HierarchicalChunker()
    base = [convert_chunk(c, source=pdf_path.name) for c in chunker.chunk(doc)]
    out: list[dict] = []
    for ch in base:
        if not ch["content"]:
            continue
        for window, start in split_with_overlap(ch["chunk_text"], size, overlap):
            item = dict(ch)
            item["chunk_text"] = window
            item["char_start"] = start
            # keep raw content window for preview
            item["content"] = window[:size]
            out.append(item)
    return out


def ensure_collection(client: QdrantClient, dim: int) -> None:
    if not client.collection_exists(collection_name=settings.collection_name):
        client.create_collection(
            collection_name=settings.collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )


def _doc_filter(source: str) -> Filter:
    return Filter(
        must=[FieldCondition(key="source", match=MatchValue(value=source))]
    )


def delete_document(client: QdrantClient, source: str) -> None:
    """Remove all points for one source file (idempotent)."""
    try:
        client.delete(
            collection_name=settings.collection_name,
            points_selector=_doc_filter(source),
            wait=True,
        )
    except Exception:
        pass


def ingest_single_file(
    pdf_path: Path,
    client: QdrantClient | None = None,
    embedder: SentenceTransformer | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> dict:
    """Ingest/replace one PDF. Returns summary dict for the API/UI."""
    pdf_path = Path(pdf_path)
    close_client = False
    if embedder is None:
        embedder = SentenceTransformer(settings.embedding_model)
    if client is None:
        client = QdrantClient(path=settings.qdrant_path)
        close_client = True
    try:
        chunks = parse_and_chunk(pdf_path, chunk_size, chunk_overlap)
        if not chunks:
            return {"source": pdf_path.name, "chunks": 0, "status": "empty"}
        texts = [c["chunk_text"] for c in chunks]
        embeddings = embedder.encode(texts, show_progress_bar=False)
        ensure_collection(client, embedder.get_embedding_dimension())
        delete_document(client, pdf_path.name)
        digest = file_sha256(pdf_path)[:8]
        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{pdf_path.name}:{digest}:{i}")),
                vector=emb.tolist(),
                payload={
                    "source": c["source"],
                    "headings": c["headings"],
                    "content": c["content"],
                    "chunk_text": c["chunk_text"],
                    "page_no": c["page_no"],
                    "char_start": c.get("char_start", 0),
                },
            )
            for i, (c, emb) in enumerate(zip(chunks, embeddings))
        ]
        client.upsert(collection_name=settings.collection_name, points=points, wait=True)
        preview = [
            {
                "index": i,
                "headings": c["headings"],
                "page_no": c["page_no"],
                "char_start": c.get("char_start", 0),
                "start": c["chunk_text"][:200],
            }
            for i, c in enumerate(chunks[:50])
        ]
        return {
            "source": pdf_path.name,
            "chunks": len(chunks),
            "status": "ok",
            "preview": preview,
        }
    finally:
        if close_client:
            client.close()


def list_chunks(
    client: QdrantClient, source: str | None = None, limit: int = 100
) -> list[dict]:
    """Return stored chunks (for the chunk-viewer UI)."""
    query_filter = _doc_filter(source) if source else None
    points, _ = client.scroll(
        collection_name=settings.collection_name,
        scroll_filter=query_filter,
        limit=limit,
        with_payload=True,
    )
    out = []
    for p in points:
        payload = p.payload or {}
        out.append(
            {
                "id": str(p.id),
                "source": payload.get("source"),
                "headings": payload.get("headings", []),
                "page_no": payload.get("page_no"),
                "char_start": payload.get("char_start", 0),
                "start": (payload.get("chunk_text") or "")[:200],
                "chunk_text": payload.get("chunk_text", ""),
            }
        )
    return out


def list_documents(client: QdrantClient) -> list[dict]:
    """Aggregate distinct sources with counts."""
    points, _ = client.scroll(
        collection_name=settings.collection_name,
        limit=10000,
        with_payload=True,
    )
    agg: dict[str, int] = {}
    for p in points:
        src = (p.payload or {}).get("source", "unknown")
        agg[src] = agg.get(src, 0) + 1
    return [{"source": k, "chunks": v} for k, v in sorted(agg.items())]


def ingest_documents(docs_dir: str) -> None:
    """CLI entry: incremental ingest of every PDF in docs_dir."""
    docs_path = Path(docs_dir)
    pdf_files = sorted(docs_path.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in '{docs_dir}'. Add PDFs and try again.")
        return
    print(f"\nFound {len(pdf_files)} PDF(s) in '{docs_dir}':")
    for f in pdf_files:
        print(f"  - {f.name}")
    embedder = SentenceTransformer(settings.embedding_model)
    client = QdrantClient(path=settings.qdrant_path)
    try:
        ensure_collection(client, embedder.get_embedding_dimension())
        total = 0
        for pdf_path in pdf_files:
            try:
                res = ingest_single_file(pdf_path, client=client, embedder=embedder)
                print(f"  [{res['status'].upper()}] {pdf_path.name} -> {res['chunks']} chunks")
                total += res["chunks"]
            except Exception as e:
                print(f"  [FAIL] {pdf_path.name}: {e}")
        print(f"\nTotal chunks (this run): {total}")
        info = client.get_collection(settings.collection_name)
        print(f"[OK] Collection now has {info.points_count} points")
    finally:
        client.close()
    print("\n[DONE] Ingestion complete. You can now start the voice agent / API.")


if __name__ == "__main__":
    print("Nano Voice Agents — Document Ingestion Pipeline")
    print(f"  Docs folder:    {settings.docs_dir}")
    print(f"  Embedding:      {settings.embedding_model}")
    print(f"  Qdrant path:    {settings.qdrant_path}")
    print(f"  Collection:     {settings.collection_name}")
    print(f"  Chunk size:     {settings.chunk_size} overlap {settings.chunk_overlap}")
    ingest_documents(settings.docs_dir)
