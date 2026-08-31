"""
pipeline.py — Preprocessing Pipeline Orchestrator
===================================================
Wires together all preprocessing steps into a single, callable pipeline:

  ingest → (pdf_convert) → clean → segment → chunk → tag_metadata

This module is the only entry point that downstream consumers (the MAS
orchestration module and the single-agent baseline) should call.  All
individual step modules remain independently importable and testable.

Error handling strategy (Technical Architecture §10):
  * Fail loudly and log everything.
  * PDF conversion failures halt the run for the affected document.
  * Text cleaning / segmentation / chunking errors halt the run.
  * Each error is captured as an ``ErrorFailureLog`` entry and written
    to the run logger.
  * The pipeline never silently skips or degrades content.

Output persistence:
  * Chunks are serialised to
    ``outputs/{run_id}/01_preprocessing/chunks.json`` if
    ``output_dir`` is provided.
  * ``GDDDocument`` and ``GDDMetadata`` are serialised alongside.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import yaml

from gdd_userstory_mas.logging.run_logger import RunLogger
from gdd_userstory_mas.preprocessing.chunker import ChunkerConfig, chunk_segments
from gdd_userstory_mas.preprocessing.ingestion import IngestResult, ingest_file
from gdd_userstory_mas.preprocessing.metadata_tagger import tag_metadata
from gdd_userstory_mas.preprocessing.pdf_converter import (
    PDFConversionError,
    convert_pdf_to_text,
)
from gdd_userstory_mas.preprocessing.segmenter import (
    Segment,
    SegmenterConfig,
    segment_text,
)
from gdd_userstory_mas.preprocessing.text_cleaner import clean_text
from gdd_userstory_mas.schemas.error_failure_log import ErrorFailureLog
from gdd_userstory_mas.schemas.gdd_chunk import GDDChunk
from gdd_userstory_mas.schemas.gdd_document import GDDDocument
from gdd_userstory_mas.schemas.gdd_metadata import GDDMetadata, TOCEntry


# ── Configuration dataclass ────────────────────────────────────────────────────

@dataclass
class PreprocessingConfig:
    """
    Merged configuration for the full preprocessing pipeline.

    Values map to ``config/pipeline.yaml`` keys.  The class can be
    constructed from a YAML file via ``PreprocessingConfig.from_yaml()``.
    """

    max_chunk_tokens: int = 1500
    overlap_tokens: int = 0
    tiktoken_encoding: str = "cl100k_base"
    sentence_splitter: str = "nltk"

    heading_patterns: List[str] = field(default_factory=list)
    min_heading_length: int = 2
    max_heading_length: int = 120

    reject_non_english: bool = False

    config_ref: Optional[str] = None
    """Opaque reference to the versioned config snapshot (Technical Architecture §8)."""

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "PreprocessingConfig":
        """Load configuration from a YAML file."""
        with open(yaml_path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        chunking = data.get("chunking", {})
        segmentation = data.get("segmentation", {})
        language = data.get("language_detection", {})

        return cls(
            max_chunk_tokens=chunking.get("max_chunk_tokens", 1500),
            overlap_tokens=chunking.get("overlap_tokens", 0),
            sentence_splitter=chunking.get("sentence_splitter", "nltk"),
            heading_patterns=segmentation.get("heading_patterns", []),
            min_heading_length=segmentation.get("min_heading_length", 2),
            max_heading_length=segmentation.get("max_heading_length", 120),
            reject_non_english=language.get("reject_non_english", False),
            config_ref=str(yaml_path),
        )

    @classmethod
    def defaults(cls) -> "PreprocessingConfig":
        """Return a config with sensible defaults (no file required)."""
        from gdd_userstory_mas.preprocessing.segmenter import DEFAULT_HEADING_PATTERNS
        return cls(heading_patterns=list(DEFAULT_HEADING_PATTERNS))


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class PreprocessingResult:
    """
    Full output of the preprocessing pipeline for one GDD document.

    Attributes
    ----------
    document:
        The ``GDDDocument`` schema object (root artifact).
    metadata:
        The ``GDDMetadata`` schema object (structural map).
    chunks:
        Ordered list of ``GDDChunk`` objects ready for agent consumption.
    segments:
        Internal ``Segment`` objects (for debugging / downstream use).
    errors:
        Any ``ErrorFailureLog`` entries generated during preprocessing.
    """

    document: GDDDocument
    metadata: GDDMetadata
    chunks: List[GDDChunk]
    segments: List[Segment]
    errors: List[ErrorFailureLog] = field(default_factory=list)


# ── Pipeline class ─────────────────────────────────────────────────────────────

class PreprocessingPipeline:
    """
    Orchestrates all preprocessing steps for a single GDD document.

    Parameters
    ----------
    config:
        Pipeline configuration.  ``None`` loads defaults.
    logger:
        ``RunLogger`` instance.  If omitted, a no-op logger is created.
    output_dir:
        If provided, chunks and metadata are written to
        ``<output_dir>/01_preprocessing/``.
    """

    def __init__(
        self,
        config: Optional[PreprocessingConfig] = None,
        logger: Optional[RunLogger] = None,
        output_dir: Optional[Path] = None,
    ) -> None:
        self._config = config or PreprocessingConfig.defaults()
        self._logger = logger
        self._output_dir = output_dir

    # ── Public entry point ─────────────────────────────────────────────────────

    def run(
        self,
        file_path: str | Path,
        *,
        document_id: Optional[str] = None,
        run_id: Optional[str] = None,
        genre: Optional[str] = None,
        source_provenance: Optional[str] = None,
    ) -> PreprocessingResult:
        """
        Run the full preprocessing pipeline on ``file_path``.

        Parameters
        ----------
        file_path:
            Path to the GDD file (.txt or .pdf).
        document_id:
            Pre-assigned document ID (from manifest), or auto-generated.
        run_id:
            Pre-assigned run ID, or auto-generated.
        genre:
            Optional game genre metadata.
        source_provenance:
            Optional provenance string.

        Returns
        -------
        PreprocessingResult

        Raises
        ------
        RuntimeError
            If any non-recoverable error occurs (PDF conversion failure,
            empty text, etc.).  All errors are also captured in the
            result's ``errors`` list.
        """
        errors: List[ErrorFailureLog] = []
        rid = run_id or str(uuid.uuid4())

        # ── Step 1: Ingestion ──────────────────────────────────────────────
        self._log("info", "ingestion", f"Starting ingestion: {file_path}")
        try:
            ingest_result: IngestResult = ingest_file(
                file_path,
                document_id=document_id,
                run_id=rid,
                reject_non_english=self._config.reject_non_english,
            )
        except (FileNotFoundError, ValueError) as exc:
            err = self._make_error(rid, "ingestion", "other", str(exc), "run_halted")
            errors.append(err)
            if self._logger:
                self._logger.log_error(err)
            raise RuntimeError(f"Ingestion failed: {exc}") from exc

        doc_id = ingest_result.document_id
        rid = ingest_result.run_id

        if ingest_result.language_warning:
            self._log("warning", "ingestion", ingest_result.language_warning)

        self._log(
            "info",
            "ingestion",
            f"Accepted '{ingest_result.file_name}' "
            f"(document_id={doc_id}, format={ingest_result.format})",
        )

        # ── Step 2: Extract raw text ───────────────────────────────────────
        raw_text: str
        if ingest_result.format == "pdf":
            self._log("info", "pdf_conversion", "Starting PDF-to-text conversion.")
            try:
                pdf_result = convert_pdf_to_text(ingest_result.file_path)
                raw_text = pdf_result.text
                for warning in pdf_result.warnings:
                    self._log("warning", "pdf_conversion", warning)
                    err = self._make_error(
                        rid, "pdf_conversion", "pdf_conversion_failure",
                        warning, "excluded_and_logged",
                        related_entity_id=doc_id,
                    )
                    errors.append(err)
                    if self._logger:
                        self._logger.log_error(err)
                self._log(
                    "info", "pdf_conversion",
                    f"Converted {pdf_result.page_count} pages "
                    f"({len(pdf_result.failed_pages)} failed).",
                )
            except PDFConversionError as exc:
                err = self._make_error(
                    rid, "pdf_conversion", "pdf_conversion_failure",
                    str(exc), "run_halted", related_entity_id=doc_id,
                )
                errors.append(err)
                if self._logger:
                    self._logger.log_error(err)
                raise RuntimeError(f"PDF conversion failed: {exc}") from exc
        else:
            # .txt: read directly
            try:
                raw_text = ingest_result.file_path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                err = self._make_error(
                    rid, "ingestion", "missing_source_content",
                    str(exc), "run_halted", related_entity_id=doc_id,
                )
                errors.append(err)
                if self._logger:
                    self._logger.log_error(err)
                raise RuntimeError(f"Could not read text file: {exc}") from exc

        # ── Step 3: Text Cleaning ──────────────────────────────────────────
        self._log("info", "text_cleaning", "Cleaning text.")
        try:
            cleaned = clean_text(raw_text)
        except Exception as exc:  # noqa: BLE001
            err = self._make_error(
                rid, "text_cleaning", "other", str(exc), "run_halted",
                related_entity_id=doc_id,
            )
            errors.append(err)
            if self._logger:
                self._logger.log_error(err)
            raise RuntimeError(f"Text cleaning failed: {exc}") from exc

        if not cleaned.strip():
            msg = "Text cleaning produced empty output."
            err = self._make_error(
                rid, "text_cleaning", "missing_source_content", msg, "run_halted",
                related_entity_id=doc_id,
            )
            errors.append(err)
            if self._logger:
                self._logger.log_error(err)
            raise RuntimeError(msg)

        self._log("info", "text_cleaning", f"Cleaned text length: {len(cleaned)} chars.")

        # ── Step 4: Segmentation ───────────────────────────────────────────
        self._log("info", "segmentation", "Detecting document structure.")
        seg_config = SegmenterConfig(
            heading_patterns=self._config.heading_patterns or [],
            min_heading_length=self._config.min_heading_length,
            max_heading_length=self._config.max_heading_length,
        )
        try:
            segments = segment_text(cleaned, config=seg_config)
        except Exception as exc:  # noqa: BLE001
            err = self._make_error(
                rid, "segmentation", "other", str(exc), "run_halted",
                related_entity_id=doc_id,
            )
            errors.append(err)
            if self._logger:
                self._logger.log_error(err)
            raise RuntimeError(f"Segmentation failed: {exc}") from exc

        structure_detected = any(s.title is not None for s in segments)
        self._log(
            "info", "segmentation",
            f"{len(segments)} segment(s) detected. "
            f"structure_detected={structure_detected}",
        )
        if not structure_detected:
            self._log(
                "warning", "segmentation",
                "No headings detected; applying single-segment fallback (VAL-010).",
            )

        # ── Step 5: Chunking ───────────────────────────────────────────────
        self._log("info", "chunking", "Chunking segments.")
        chunk_config = ChunkerConfig(
            max_chunk_tokens=self._config.max_chunk_tokens,
            overlap_tokens=self._config.overlap_tokens,
            tiktoken_encoding=self._config.tiktoken_encoding,
            sentence_splitter=self._config.sentence_splitter,
        )
        try:
            chunk_specs = chunk_segments(
                segments, config=chunk_config, cleaned_text=cleaned
            )
        except Exception as exc:  # noqa: BLE001
            err = self._make_error(
                rid, "chunking", "other", str(exc), "run_halted",
                related_entity_id=doc_id,
            )
            errors.append(err)
            if self._logger:
                self._logger.log_error(err)
            raise RuntimeError(f"Chunking failed: {exc}") from exc

        self._log("info", "chunking", f"{len(chunk_specs)} chunk(s) produced.")

        # ── Step 6: Metadata Tagging ───────────────────────────────────────
        self._log("info", "metadata_tagging", "Attaching metadata to chunks.")
        try:
            gdd_chunks = tag_metadata(
                chunk_specs, doc_id, segments,
                max_chunk_tokens=self._config.max_chunk_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            err = self._make_error(
                rid, "metadata_tagging", "schema_violation", str(exc), "run_halted",
                related_entity_id=doc_id,
            )
            errors.append(err)
            if self._logger:
                self._logger.log_error(err)
            raise RuntimeError(f"Metadata tagging failed: {exc}") from exc

        self._log(
            "info", "metadata_tagging",
            f"{len(gdd_chunks)} GDDChunk(s) ready for agent consumption.",
        )

        # ── Build schema objects ───────────────────────────────────────────
        gdd_document = GDDDocument(
            document_id=doc_id,
            file_name=ingest_result.file_name,
            format=ingest_result.format,  # type: ignore[arg-type]
            language="en",
            genre=genre,
            raw_text=cleaned,  # store the cleaned text for downstream access
            ingestion_timestamp=ingest_result.ingestion_timestamp,
            source_provenance=source_provenance,
        )

        toc_entries = [
            TOCEntry(
                title=seg.title,
                level=seg.level,
                position=seg.position,
            )
            for seg in segments
            if seg.title is not None
        ]

        gdd_metadata = GDDMetadata(
            document_id=doc_id,
            structure_detected=structure_detected,
            segment_count=len(segments),
            table_of_contents=toc_entries,
            preprocessing_config_ref=self._config.config_ref,
        )

        result = PreprocessingResult(
            document=gdd_document,
            metadata=gdd_metadata,
            chunks=gdd_chunks,
            segments=segments,
            errors=errors,
        )

        # ── Persist output if requested ────────────────────────────────────
        if self._output_dir:
            self._persist(result, rid)

        return result

    # ── Persistence helpers ────────────────────────────────────────────────────

    def _persist(self, result: PreprocessingResult, run_id: str) -> None:
        """Write preprocessing outputs to ``output_dir/01_preprocessing/``."""
        out_dir = self._output_dir / "01_preprocessing"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Chunks
        chunks_path = out_dir / "chunks.json"
        chunks_data = [c.model_dump(mode="json") for c in result.chunks]
        chunks_path.write_text(
            json.dumps(chunks_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Document metadata
        doc_path = out_dir / "gdd_document.json"
        doc_path.write_text(
            result.document.model_dump_json(indent=2), encoding="utf-8"
        )

        # GDD metadata
        meta_path = out_dir / "gdd_metadata.json"
        meta_path.write_text(
            result.metadata.model_dump_json(indent=2), encoding="utf-8"
        )

        self._log(
            "info", "metadata_tagging",
            f"Preprocessing output written to {out_dir}",
        )

    # ── Logging / error helpers ────────────────────────────────────────────────

    def _log(self, level: str, stage: str, message: str) -> None:
        if self._logger:
            getattr(self._logger, level)(stage, message)

    @staticmethod
    def _make_error(
        run_id: str,
        stage: str,
        error_type: str,
        message: str,
        resolution: str,
        *,
        related_entity_id: Optional[str] = None,
    ) -> ErrorFailureLog:
        return ErrorFailureLog(
            error_id=str(uuid.uuid4()),
            run_id=run_id,
            stage=stage,  # type: ignore[arg-type]
            error_type=error_type,  # type: ignore[arg-type]
            message=message,
            resolution=resolution,  # type: ignore[arg-type]
            related_entity_id=related_entity_id,
            timestamp=datetime.now(tz=timezone.utc),
        )
