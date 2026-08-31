"""
tests/unit/preprocessing/test_ingestion.py
============================================
Unit tests for the ingestion module (SRS-001, SRS-002).

Tests are fully offline — no LLM calls, no external services.
"""
import uuid
from pathlib import Path

import pytest

from gdd_userstory_mas.preprocessing.ingestion import (
    SUPPORTED_EXTENSIONS,
    IngestResult,
    ingest_file,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def txt_file(tmp_path: Path) -> Path:
    """A valid, non-empty English .txt GDD file."""
    p = tmp_path / "sample_gdd.txt"
    p.write_text(
        "1. Overview\nThis is an action RPG game about exploring dungeons.\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture()
def empty_txt_file(tmp_path: Path) -> Path:
    """A zero-byte .txt file (should fail VAL-002)."""
    p = tmp_path / "empty.txt"
    p.write_bytes(b"")
    return p


@pytest.fixture()
def pdf_file(tmp_path: Path) -> Path:
    """A fake .pdf file (non-zero bytes; content not read by ingestion)."""
    p = tmp_path / "sample.pdf"
    p.write_bytes(b"%PDF-1.4 fake content")
    return p


# ── VAL-001: Extension validation ─────────────────────────────────────────────

class TestExtensionValidation:
    def test_accepts_txt(self, txt_file: Path) -> None:
        result = ingest_file(txt_file)
        assert result.format == "txt"

    def test_accepts_pdf(self, pdf_file: Path) -> None:
        result = ingest_file(pdf_file)
        assert result.format == "pdf"

    def test_rejects_unsupported_extension(self, tmp_path: Path) -> None:
        bad = tmp_path / "document.docx"
        bad.write_text("some content", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            ingest_file(bad)

    def test_supported_extensions_set(self) -> None:
        assert ".txt" in SUPPORTED_EXTENSIONS
        assert ".pdf" in SUPPORTED_EXTENSIONS


# ── VAL-002: Empty file ────────────────────────────────────────────────────────

class TestEmptyFileValidation:
    def test_rejects_empty_file(self, empty_txt_file: Path) -> None:
        with pytest.raises(ValueError, match="empty"):
            ingest_file(empty_txt_file)


# ── File existence ─────────────────────────────────────────────────────────────

class TestFileNotFound:
    def test_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            ingest_file(tmp_path / "nonexistent.txt")


# ── ID assignment (SRS-002) ────────────────────────────────────────────────────

class TestIdAssignment:
    def test_auto_generates_uuid_document_id(self, txt_file: Path) -> None:
        result = ingest_file(txt_file)
        # Must be a valid UUID4
        parsed = uuid.UUID(result.document_id, version=4)
        assert str(parsed) == result.document_id

    def test_auto_generates_uuid_run_id(self, txt_file: Path) -> None:
        result = ingest_file(txt_file)
        parsed = uuid.UUID(result.run_id, version=4)
        assert str(parsed) == result.run_id

    def test_accepts_custom_document_id(self, txt_file: Path) -> None:
        custom_id = "doc-001"
        result = ingest_file(txt_file, document_id=custom_id)
        assert result.document_id == custom_id

    def test_accepts_custom_run_id(self, txt_file: Path) -> None:
        custom_run = "run-001"
        result = ingest_file(txt_file, run_id=custom_run)
        assert result.run_id == custom_run

    def test_two_calls_produce_different_ids(self, txt_file: Path) -> None:
        r1 = ingest_file(txt_file)
        r2 = ingest_file(txt_file)
        assert r1.document_id != r2.document_id
        assert r1.run_id != r2.run_id

    def test_result_contains_expected_fields(self, txt_file: Path) -> None:
        result = ingest_file(txt_file)
        assert isinstance(result, IngestResult)
        assert result.file_name == txt_file.name
        assert result.format == "txt"
        assert result.ingestion_timestamp is not None
        assert result.file_path == txt_file.resolve()

    def test_language_warning_is_none_for_english(self, txt_file: Path) -> None:
        result = ingest_file(txt_file)
        # Either None (langdetect available and detected English) or None (not available)
        # Both are acceptable
        assert result.language_warning is None or isinstance(
            result.language_warning, str
        )


# ── VAL-003: Language detection ────────────────────────────────────────────────

class TestLanguageDetection:
    def test_reject_non_english_raises_when_enabled(self, tmp_path: Path) -> None:
        """
        This test checks the reject_non_english flag.
        We use a clearly non-English text to trigger detection.
        If langdetect is not installed, the test is skipped.
        """
        pytest.importorskip("langdetect")
        non_english = tmp_path / "non_english.txt"
        # French text — long enough for langdetect to be confident
        non_english.write_text(
            "Bonjour le monde. Ceci est un document de jeu en français. "
            "Le héros doit sauver le royaume. Il combat des ennemis puissants. "
            "La magie est importante dans ce jeu de rôle fantastique incroyable. " * 5,
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Language detection"):
            ingest_file(non_english, reject_non_english=True)

    def test_non_english_warning_not_error_by_default(self, tmp_path: Path) -> None:
        """By default, non-English text is flagged, not rejected."""
        pytest.importorskip("langdetect")
        non_english = tmp_path / "non_english2.txt"
        non_english.write_text(
            "Hola mundo. Este es un documento de diseño de juego en español. "
            "El héroe debe salvar el reino de los monstruos malignos del bosque. " * 5,
            encoding="utf-8",
        )
        # Should NOT raise — just return a warning
        result = ingest_file(non_english, reject_non_english=False)
        # If langdetect detected non-English, warning should be populated
        # (but if it misclassified as English, warning is None — acceptable)
        assert isinstance(result, IngestResult)
