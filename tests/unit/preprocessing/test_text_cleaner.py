"""
tests/unit/preprocessing/test_text_cleaner.py
===============================================
Unit tests for the text cleaning module (SRS-004).

All tests are pure Python — no LLM, no external services.
"""
import pytest

from gdd_userstory_mas.preprocessing.text_cleaner import clean_text


# ── VAL-007: Valid UTF-8 output ────────────────────────────────────────────────

class TestUTF8Normalization:
    def test_removes_null_bytes(self) -> None:
        raw = "Hello\x00World"
        result = clean_text(raw)
        assert "\x00" not in result
        assert "HelloWorld" in result

    def test_removes_form_feed(self) -> None:
        raw = "Page one\x0cPage two"
        result = clean_text(raw)
        assert "\x0c" not in result

    def test_removes_soft_hyphen(self) -> None:
        raw = "action\xadRPG game"
        result = clean_text(raw)
        assert "\xad" not in result

    def test_preserves_tab_and_newline(self) -> None:
        raw = "Line one\n\tIndented line two\n"
        result = clean_text(raw)
        assert "\n" in result
        assert "\t" in result

    def test_normalises_windows_line_endings(self) -> None:
        raw = "Line one\r\nLine two\r\nLine three"
        result = clean_text(raw)
        assert "\r\n" not in result
        assert "\r" not in result
        assert "Line one" in result
        assert "Line two" in result

    def test_normalises_old_mac_line_endings(self) -> None:
        raw = "Line one\rLine two\rLine three"
        result = clean_text(raw)
        assert "\r" not in result

    def test_unicode_nfc_normalization(self) -> None:
        # "é" can be encoded as U+00E9 (precomposed) or U+0065 + U+0301 (decomposed)
        decomposed = "e\u0301"  # e + combining acute accent
        precomposed = "\u00e9"
        raw = f"caf{decomposed} is a {decomposed}nvironment"
        result = clean_text(raw)
        assert precomposed in result


# ── Page number removal ────────────────────────────────────────────────────────

class TestPageNumberRemoval:
    def test_removes_standalone_page_numbers(self) -> None:
        raw = "## Combat System\n\nFight enemies.\n\n42\n\nMore content here."
        result = clean_text(raw, remove_page_numbers=True)
        lines = result.split("\n")
        # "42" alone should be removed
        assert not any(line.strip() == "42" for line in lines)

    def test_preserves_numbers_in_content(self) -> None:
        raw = "The player has 42 health points and 10 mana."
        result = clean_text(raw)
        assert "42" in result

    def test_preserves_numbered_headings(self) -> None:
        raw = "1. Overview\n\nSome content.\n\n2. Combat\n\nMore content."
        result = clean_text(raw)
        assert "1. Overview" in result
        assert "2. Combat" in result

    def test_page_number_removal_off(self) -> None:
        raw = "Content.\n\n5\n\nMore content."
        result = clean_text(raw, remove_page_numbers=False)
        assert "5" in result


# ── Repeated header/footer removal ────────────────────────────────────────────

class TestRepeatedHeaderRemoval:
    def test_removes_repeated_short_lines(self) -> None:
        repeated_header = "Chronicles of Aethermoor"
        raw = "\n".join(
            [repeated_header, "Content A.", repeated_header, "Content B.",
             repeated_header, "Content C.", repeated_header, "Content D."]
        )
        result = clean_text(raw, remove_repeated_headers=True, header_min_occurrences=3)
        assert repeated_header not in result

    def test_preserves_unique_short_lines(self) -> None:
        raw = "Introduction\n\nSome text here.\n\nConclusion\n\nFinal text."
        result = clean_text(raw, remove_repeated_headers=True, header_min_occurrences=3)
        assert "Introduction" in result
        assert "Conclusion" in result

    def test_never_removes_numbered_headings(self) -> None:
        repeated = "1. Overview"
        raw = "\n".join([repeated] * 5 + ["Content."])
        result = clean_text(raw, remove_repeated_headers=True, header_min_occurrences=3)
        # Numbered headings are protected from removal
        assert repeated in result

    def test_repeated_header_removal_off(self) -> None:
        repeated = "Game Title"
        raw = "\n".join([repeated] * 5 + ["Content."])
        result = clean_text(raw, remove_repeated_headers=False)
        assert repeated in result


# ── Whitespace normalisation ───────────────────────────────────────────────────

class TestWhitespaceNormalization:
    def test_collapses_excessive_blank_lines(self) -> None:
        raw = "Paragraph one.\n\n\n\n\n\nParagraph two."
        result = clean_text(raw)
        assert "\n\n\n" not in result
        assert "Paragraph one." in result
        assert "Paragraph two." in result

    def test_strips_leading_and_trailing_whitespace(self) -> None:
        raw = "\n\n   Content   \n\n"
        result = clean_text(raw)
        assert not result.startswith("\n")
        assert not result.endswith("\n")


# ── Type guard ─────────────────────────────────────────────────────────────────

class TestTypeGuard:
    def test_raises_on_non_string_input(self) -> None:
        with pytest.raises(ValueError, match="must be a str"):
            clean_text(12345)  # type: ignore[arg-type]


# ── Fixture-based smoke test ───────────────────────────────────────────────────

class TestFixtureFile:
    def test_cleans_sample_gdd_fixture(self) -> None:
        from pathlib import Path

        fixture = Path(__file__).parent.parent.parent / "fixtures" / "sample_gdd_small.txt"
        raw = fixture.read_text(encoding="utf-8")
        result = clean_text(raw)
        assert len(result) > 0
        assert "GAME OVERVIEW" in result or "game" in result.lower()
