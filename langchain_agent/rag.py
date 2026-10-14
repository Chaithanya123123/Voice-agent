import sys

from langchain_mistralai import ChatMistralAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from config import settings


def retrieve(
    client: QdrantClient,
    embedder: SentenceTransformer,
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.0,
) -> list[dict]:
    """Embed the query and return the top-k most similar chunks from Qdrant."""
    query_vector = embedder.encode(query, normalize_embeddings=True).tolist()

    hits = client.query_points(
        collection_name=settings.collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True,
        score_threshold=score_threshold or None,
    )

    return [{**hit.payload, "score": round(hit.score, 4)} for hit in hits.points]


def build_context(retrieved_chunks: list[dict], max_chars: int | None = None) -> str:
    """Build a context string from retrieved chunks, capped by max_chars."""
    budget = max_chars or settings.max_context_chars
    parts = []
    used = 0
    for i, chunk in enumerate(retrieved_chunks, 1):
        piece = f"[Source {i}]\n{chunk['chunk_text']}"
        if used + len(piece) > budget:
            break
        parts.append(piece)
        used += len(piece)
    return "\n\n---\n\n".join(parts)


def rag(query: str, top_k: int = 5, client=None, embedder=None, style: str = "voice"):
    """
    End-to-end RAG pipeline:
      1. Retrieve relevant chunks from Qdrant
      2. Format as context
      3. Send context + query to Mistral LLM

    style: 'voice' (1-2 sentences) or 'text' (fuller answer with bullets).

    Returns:
        tuple: (answer_text, context_used, sources)
    """
    if embedder is None:
        embedder = SentenceTransformer(settings.embedding_model)
    if client is None:
        client = QdrantClient(path=settings.qdrant_path)

    # Step 1 — Retrieve
    chunks = retrieve(
        client, embedder, query, top_k=top_k,
        score_threshold=settings.score_threshold,
    )
    if not chunks:
        return (
            "I couldn't find specific information about that in our documentation. "
            "Could you rephrase your question or ask about plans, pricing, or coverage?",
            "",
            [],
        )

    # Step 2 — Build context
    context = build_context(chunks)

    # Step 3 — Generate answer
    if style == "text":
        brevity = (
            "You may use up to a short paragraph plus bullets for plans/prices. "
            "Include ₹ prices and validity."
        )
    else:
        brevity = "Keep responses brief and conversational for voice calls (single sentence preferred, max 2 sentences)."
    system_prompt = (
        "You are a helpful Nano Voice Agents customer support assistant.\n"
        "Answer the user's question using ONLY the context provided below.\n"
        "If the context does not contain enough information, say so — do not make things up.\n"
        "Always cite the plan name or section when referencing specific information.\n"
        f"{brevity}\n"
        "Format any prices with ₹ symbol and mention validity periods when relevant."
    )

    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    llm = ChatMistralAI(
        model=settings.mistral_model,
        temperature=0.2,
        api_key=settings.mistral_api_key,
    )
    messages = [
        ("system", system_prompt),
        ("user", user_message),
    ]

    response = llm.invoke(messages)
    return response.content, context, chunks


def main():
    """CLI interface for testing RAG independently."""
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "What are the available Nano Voice Agents plans?"

    print(f"Question: {query}\n")
    print("Generating answer...\n")

    answer, context = rag(query)[:2]

    print(f"ANSWER:\n{answer}")
    print(f"\n{'='*60}")
    print(f"\nSOURCES:\n{context}")


if __name__ == "__main__":
    main()
