"""
tests/unit/mas/test_reader_output_schema.py
============================================
Unit tests for the ReaderOutput and EvidencedItem Pydantic schemas
(Data Contracts §4).

Tests are OFFLINE — no LLM calls, no file I/O.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from gdd_userstory_mas.schemas.reader_output import EvidencedItem, ReaderOutput


# ─────────────────────────────────────────────────────────────────────────────
# EvidencedItem tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEvidencedItem:

    def test_valid_with_excerpt(self):
        item = EvidencedItem(
            content="Player can sprint by holding Shift",
            source_excerpt="The player can sprint by holding the Shift key",
        )
        assert item.content == "Player can sprint by holding Shift"
        assert item.source_excerpt == "The player can sprint by holding the Shift key"

    def test_valid_without_excerpt(self):
        item = EvidencedItem(content="Save system uses checkpoints")
        assert item.source_excerpt is None

    def test_content_stripped_of_whitespace(self):
        item = EvidencedItem(content="  Combat mechanics  ")
        assert item.content == "Combat mechanics"

    def test_source_excerpt_stripped(self):
        item = EvidencedItem(
            content="Something",
            source_excerpt="  raw quote  ",
        )
        assert item.source_excerpt == "raw quote"

    def test_source_excerpt_empty_string_becomes_none(self):
        item = EvidencedItem(content="Something", source_excerpt="   ")
        assert item.source_excerpt is None

    def test_empty_content_raises(self):
        with pytest.raises(ValidationError, match="content must not be empty"):
            EvidencedItem(content="")

    def test_whitespace_only_content_raises(self):
        with pytest.raises(ValidationError, match="content must not be empty"):
            EvidencedItem(content="   ")

    def test_missing_content_raises(self):
        with pytest.raises(ValidationError):
            EvidencedItem()  # type: ignore[call-arg]

    def test_frozen(self):
        item = EvidencedItem(content="Something")
        with pytest.raises(Exception):
            item.content = "Changed"  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
# ReaderOutput tests
# ─────────────────────────────────────────────────────────────────────────────

def _make_item(content: str, excerpt: str | None = None) -> EvidencedItem:
    return EvidencedItem(content=content, source_excerpt=excerpt)


def _make_output(**kwargs) -> ReaderOutput:
    defaults = dict(
        chunk_id="doc1__chunk_0000",
        document_id="doc1",
        themes=[],
        gameplay_elements=[],
        systems=[],
        characters=[],
        ui_elements=[],
        narrative=[],
    )
    defaults.update(kwargs)
    return ReaderOutput(**defaults)


class TestReaderOutputConstruction:

    def test_minimal_valid(self):
        """All list fields empty — valid (agent found nothing to extract)."""
        out = _make_output()
        assert out.chunk_id == "doc1__chunk_0000"
        assert out.document_id == "doc1"
        assert out.themes == []
        assert out.gameplay_elements == []

    def test_full_valid(self):
        out = _make_output(
            themes=["survival", "base building"],
            gameplay_elements=[
                _make_item("Player gathers resources", "player gathers"),
                _make_item("Enemy AI patrols areas"),
            ],
            systems=[_make_item("Crafting system", "craft items")],
            characters=[_make_item("Protagonist: unnamed survivor")],
            ui_elements=[_make_item("Health bar in top-left corner")],
            narrative=[_make_item("Story set in post-apocalyptic world")],
        )
        assert len(out.themes) == 2
        assert len(out.gameplay_elements) == 2
        assert len(out.systems) == 1
        assert out.total_items == 6   # 2 gameplay + 1 system + 1 character + 1 ui + 1 narrative
        assert not out.is_empty

    def test_all_empty_is_valid(self):
        """Zero items in all categories is a valid outcome."""
        out = _make_output()
        assert out.is_empty is True
        assert out.total_items == 0

    def test_themes_only(self):
        out = _make_output(themes=["action", "RPG"])
        assert not out.is_empty  # themes count toward non-empty check
        assert out.total_items == 0  # themes don't count in total_items

    def test_empty_theme_strings_stripped(self):
        out = _make_output(themes=["", "  ", "valid_theme"])
        assert out.themes == ["valid_theme"]


class TestReaderOutputTraceability:

    def test_chunk_id_required(self):
        with pytest.raises(ValidationError, match="chunk_id"):
            ReaderOutput(
                chunk_id="",
                document_id="doc1",
                themes=[],
                gameplay_elements=[],
                systems=[],
                characters=[],
                ui_elements=[],
                narrative=[],
            )

    def test_document_id_required(self):
        with pytest.raises(ValidationError, match="document_id"):
            ReaderOutput(
                chunk_id="chunk1",
                document_id="",
                themes=[],
                gameplay_elements=[],
                systems=[],
                characters=[],
                ui_elements=[],
                narrative=[],
            )

    def test_missing_chunk_id_raises(self):
        with pytest.raises(ValidationError):
            ReaderOutput(
                document_id="doc1",
                themes=[],
                gameplay_elements=[],
                systems=[],
                characters=[],
                ui_elements=[],
                narrative=[],
            )


class TestReaderOutputHelpers:

    def test_category_counts(self):
        out = _make_output(
            themes=["theme1"],
            gameplay_elements=[_make_item("mechanic")],
            systems=[_make_item("sys1"), _make_item("sys2")],
        )
        counts = out.category_counts()
        assert counts["themes"] == 1
        assert counts["gameplay_elements"] == 1
        assert counts["systems"] == 2
        assert counts["characters"] == 0
        assert counts["ui_elements"] == 0
        assert counts["narrative"] == 0

    def test_total_items_excludes_themes(self):
        out = _make_output(
            themes=["t1", "t2", "t3"],  # 3 themes
            gameplay_elements=[_make_item("m")],  # 1
        )
        assert out.total_items == 1  # themes not counted

    def test_frozen_output(self):
        out = _make_output()
        with pytest.raises(Exception):
            out.chunk_id = "changed"  # type: ignore[misc]
