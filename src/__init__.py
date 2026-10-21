"""
지방의회 회의록 ChromaDB 적재 및 메타데이터 파이프라인 패키지
"""

from src.chunker import CouncilMinuteChunker
from src.indexer import ChromaIndexer
from src.parser import parse_tsv_row, parse_speaker, clean_text
from src.pg_indexer import PGVectorIndexer
from src.pipeline import CouncilRAGPipeline
from src.trend_analyzer import PolicyTrendAnalyzer

__version__ = "1.3.0"

__all__ = [
    "parse_tsv_row",
    "parse_speaker",
    "clean_text",
    "CouncilMinuteChunker",
    "ChromaIndexer",
    "PGVectorIndexer",
    "CouncilRAGPipeline",
    "PolicyTrendAnalyzer",
]
