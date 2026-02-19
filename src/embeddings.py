"""Local embeddings client for generating text embeddings using sentence-transformers."""

import asyncio
from functools import lru_cache

from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=4)
def _load_model(model_name: str) -> SentenceTransformer:
    """Load and cache a SentenceTransformer model (avoids reloading on repeated calls)."""
    return SentenceTransformer(model_name)


class LocalEmbeddings:
    """Local sentence-transformers client for generating text embeddings.

    Recommended models (all free, no API key required):
        - "BAAI/bge-small-en-v1.5"  (~130MB, best quality/speed tradeoff) [default]
        - "all-MiniLM-L6-v2"        (~80MB,  fastest)
        - "BAAI/bge-large-en-v1.5"  (~1.3GB, best quality, slower)
        - "nomic-ai/nomic-embed-text-v1" (~270MB, great for long docs)

    Install: pip install sentence-transformers
    """

    def __init__(
        self,
        model: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 64,
        device: str | None = None,  # None = auto-detect (cuda if available, else cpu)
    ):
        self.model_name = model
        self.batch_size = batch_size
        # Load model (cached globally so re-instantiation is cheap)
        self._model = _load_model(model)
        if device:
            self._model = self._model.to(device)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts. Runs in a thread pool to avoid blocking the event loop."""
        if not texts:
            return []

        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self._model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=len(texts) > 100,
                normalize_embeddings=True,  # cosine similarity works better when normalized
                convert_to_numpy=True,
            ).tolist(),
        )
        return embeddings

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        results = await self.embed_texts([query])
        return results[0]