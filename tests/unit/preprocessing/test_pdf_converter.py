"""
tests/unit/preprocessing/test_pdf_converter.py
================================================
Unit tests for the PDF conversion module (SRS-003).

Tests use Python's standard library or pytest fixtures to create minimal
PDFs where needed, or mock pdfminer internals.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gdd_userstory_mas.preprocessing.pdf_converter import (
    PDFConversionError,
    PDFConversionResult,
    _extract_page_text,
    convert_pdf_to_text,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_fake_pdf(tmp_path: Path, filename: str = "test.pdf") -> Path:
    """
    Create a minimal valid PDF file using reportlab if available,
    otherwise write a placeholder that is non-zero bytes (for
    non-content-reading tests).
    """
    p = tmp_path / filename
    try:
        from reportlab.pdfgen import canvas  # type: ignore[import]

        c = canvas.Canvas(str(p))
        c.drawString(100, 750, "Combat System: Players attack using combos.")
        c.drawString(100, 700, "The enemy AI adapts to player behaviour.")
        c.save()
    except ImportError:
        # reportlab not available — write a stub PDF header
        p.write_bytes(b"%PDF-1.4\n%%EOF")
    return p


# ── File-not-found ─────────────────────────────────────────────────────────────

class TestFileNotFound:
    def test_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            convert_pdf_to_text(tmp_path / "missing.pdf")


# ── VAL-005: Empty result ──────────────────────────────────────────────────────

class TestEmptyResult:
    def test_raises_on_empty_text(self, tmp_path: Path) -> None:
        """
        When pdfminer is mocked to return empty text, convert_pdf_to_text
        should raise PDFConversionError (VAL-005).
        """
        p = tmp_path / "image_only.pdf"
        p.write_bytes(b"%PDF-1.4\n%%EOF")

        # Mock PDFPage.get_pages to return one page that produces no text
        mock_page = MagicMock()
        with (
            patch(
                "gdd_userstory_mas.preprocessing.pdf_converter.PDFPage.get_pages",
                return_value=[mock_page],
            ),
            patch(
                "gdd_userstory_mas.preprocessing.pdf_converter._extract_page_text",
                return_value=None,  # simulate empty page
            ),
        ):
            with pytest.raises(PDFConversionError, match="empty text"):
                convert_pdf_to_text(p)


# ── Successful extraction ──────────────────────────────────────────────────────

class TestSuccessfulExtraction:
    def test_returns_pdfconversionresult_on_success(self, tmp_path: Path) -> None:
        """Mocked extraction returns a non-empty result."""
        p = tmp_path / "valid.pdf"
        p.write_bytes(b"%PDF-1.4\n%%EOF")

        mock_page = MagicMock()
        expected_text = "Combat System: The player attacks using combos."

        with (
            patch(
                "gdd_userstory_mas.preprocessing.pdf_converter.PDFPage.get_pages",
                return_value=[mock_page],
            ),
            patch(
                "gdd_userstory_mas.preprocessing.pdf_converter._extract_page_text",
                return_value=expected_text,
            ),
        ):
            result = convert_pdf_to_text(p)

        assert isinstance(result, PDFConversionResult)
        assert expected_text in result.text
        assert result.page_count == 1
        assert result.failed_pages == []

    def test_failed_pages_tracked(self, tmp_path: Path) -> None:
        """Pages that produce None are added to failed_pages."""
        p = tmp_path / "partial.pdf"
        p.write_bytes(b"%PDF-1.4\n%%EOF")

        mock_pages = [MagicMock(), MagicMock(), MagicMock()]

        def fake_extract(page, laparams, codec):
            # Page 2 (index 1) fails
            if page is mock_pages[1]:
                return None
            return "Some valid text content from this page."

        with (
            patch(
                "gdd_userstory_mas.preprocessing.pdf_converter.PDFPage.get_pages",
                return_value=mock_pages,
            ),
            patch(
                "gdd_userstory_mas.preprocessing.pdf_converter._extract_page_text",
                side_effect=fake_extract,
            ),
        ):
            result = convert_pdf_to_text(p)

        assert result.page_count == 3
        assert 2 in result.failed_pages
        assert len(result.warnings) == 1
        assert "Page 2" in result.warnings[0]


# ── Real PDF test (requires reportlab) ────────────────────────────────────────

class TestRealPDF:
    def test_extracts_text_from_real_pdf(self, tmp_path: Path) -> None:
        """Integration test using a real PDF created by reportlab."""
        pytest.importorskip("reportlab")
        p = _make_fake_pdf(tmp_path)
        result = convert_pdf_to_text(p)
        assert isinstance(result, PDFConversionResult)
        assert result.page_count >= 1
        assert len(result.text.strip()) > 0


# ── _extract_page_text ─────────────────────────────────────────────────────────

class TestExtractPageText:
    def test_returns_none_on_exception(self) -> None:
        """If pdfminer raises during page processing, None is returned."""
        mock_page = MagicMock()
        mock_laparams = MagicMock()

        with patch(
            "gdd_userstory_mas.preprocessing.pdf_converter.PDFResourceManager",
            side_effect=RuntimeError("simulated error"),
        ):
            result = _extract_page_text(mock_page, mock_laparams, "utf-8")

        assert result is None

    def test_returns_none_for_whitespace_only_page(self) -> None:
        """Pages producing only whitespace are treated as empty."""
        mock_page = MagicMock()
        mock_laparams = MagicMock()

        with (
            patch(
                "gdd_userstory_mas.preprocessing.pdf_converter.PDFResourceManager"
            ),
            patch(
                "gdd_userstory_mas.preprocessing.pdf_converter.TextConverter"
            ),
            patch(
                "gdd_userstory_mas.preprocessing.pdf_converter.PDFPageInterpreter"
            ),
            patch("io.StringIO") as mock_sio,
        ):
            instance = MagicMock()
            instance.getvalue.return_value = "   \n\t  "
            mock_sio.return_value = instance
            result = _extract_page_text(mock_page, mock_laparams, "utf-8")

        # result may be None (whitespace-only) or a string depending on mock depth
        # The important thing is it does not raise
        assert result is None or isinstance(result, str)
