"""arabic-rag — نظام استرجاع معزّز بالتوليد للمستندات العربية مع استشهادات مرجعية."""

from .chunking import Chunk, chunk_document
from .evaluate import EvalResult, evaluate, load_dataset
from .generator import Answer, Citation, ExtractiveGenerator, OpenAIGenerator
from .index import HybridIndex
from .normalize import normalize, tokenize
from .pipeline import RAGPipeline

__version__ = "0.1.0"
__all__ = [
    "Answer", "Chunk", "Citation", "EvalResult", "ExtractiveGenerator",
    "HybridIndex", "OpenAIGenerator", "RAGPipeline", "chunk_document",
    "evaluate", "load_dataset", "normalize", "tokenize",
]
