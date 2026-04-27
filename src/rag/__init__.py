"""Clean RAG pipeline: ingest -> index -> retrieve -> rerank -> answer."""
from .config import Settings, settings
from .types import Chunk, Hit, Answer, Citation
from .pipeline import answer_query

__all__ = ["Settings", "settings", "Chunk", "Hit", "Answer", "Citation", "answer_query"]
