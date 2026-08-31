"""
gdd_userstory_mas.preprocessing
================================
Preprocessing pipeline: ingestion → PDF conversion → text cleaning →
segmentation → chunking → metadata tagging.

Public entry point: ``PreprocessingPipeline`` (in ``pipeline.py``).
Individual steps are importable independently for unit testing.
"""
from gdd_userstory_mas.preprocessing.ingestion import IngestResult, ingest_file
from gdd_userstory_mas.preprocessing.pdf_converter import convert_pdf_to_text
from gdd_userstory_mas.preprocessing.text_cleaner import clean_text
from gdd_userstory_mas.preprocessing.segmenter import Segment, segment_text
from gdd_userstory_mas.preprocessing.chunker import ChunkSpec, chunk_segments
from gdd_userstory_mas.preprocessing.metadata_tagger import tag_metadata
from gdd_userstory_mas.preprocessing.pipeline import (
    PreprocessingConfig,
    PreprocessingResult,
    PreprocessingPipeline,
)

__all__ = [
    "ingest_file",
    "IngestResult",
    "convert_pdf_to_text",
    "clean_text",
    "segment_text",
    "Segment",
    "chunk_segments",
    "ChunkSpec",
    "tag_metadata",
    "PreprocessingConfig",
    "PreprocessingResult",
    "PreprocessingPipeline",
]
