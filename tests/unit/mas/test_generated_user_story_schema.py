"""
tests/unit/mas/test_generated_user_story_schema.py
====================================================
Unit tests for the GeneratedUserStory Pydantic schema (Data Contracts §6).
All tests are OFFLINE — no LLM calls, no file I/O.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from gdd_userstory_mas.schemas.generated_user_story import GeneratedUserStory


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_story(**kwargs) -> GeneratedUserStory:
    defaults = dict(
        draft_id="doc1__cand_0000__draft",
        candidate_id="doc1__cand_0000",
        source_chunk_id="doc1__chunk_0000",
        document_id="doc1",
        role="player",
        action="attack enemies using melee weapons",
        benefit="I can defeat them and progress through the level",
        full_text=(
            "As a player, I want attack enemies using melee weapons, "
            "so that I can defeat them and progress through the level."
        ),
        iteration_count=0,
        previous_draft_id_chain=[],
    )
    defaults.update(kwargs)
    return GeneratedUserStory(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# Construction
# ─────────────────────────────────────────────────────────────────────────────

class TestGeneratedUserStoryConstruction:

    def test_valid_first_draft(self):
        story = _make_story()
        assert story.draft_id == "doc1__cand_0000__draft"
        assert story.iteration_count == 0
        assert story.is_first_draft()
        assert story.previous_draft_id_chain == []

    def test_valid_revision(self):
        story = _make_story(iteration_count=2, previous_draft_id_chain=["v0", "v1"])
        assert not story.is_first_draft()
        assert story.iteration_count == 2
        assert len(story.previous_draft_id_chain) == 2

    def test_frozen(self):
        story = _make_story()
        with pytest.raises(Exception):
            story.role = "changed"  # type: ignore[misc]

    def test_build_full_text_helper(self):
        text = GeneratedUserStory.build_full_text("player", "jump", "reach higher platforms")
        assert text == "As a player, I want jump, so that reach higher platforms."


# ─────────────────────────────────────────────────────────────────────────────
# full_text format validation (VAL-020)
# ─────────────────────────────────────────────────────────────────────────────

class TestFullTextFormat:

    def test_correct_format_passes(self):
        story = _make_story()
        assert story.full_text.startswith("As a ")
        assert story.full_text.endswith(".")

    def test_missing_as_a_raises(self):
        with pytest.raises(ValidationError, match="VAL-020"):
            _make_story(
                role="player",
                action="jump",
                benefit="reach higher areas",
                full_text="I want to jump, so that I can reach higher areas.",
            )

    def test_missing_i_want_raises(self):
        with pytest.raises(ValidationError, match="VAL-020"):
            _make_story(
                role="player",
                action="jump",
                benefit="reach higher areas",
                full_text="As a player, so that I can reach higher areas.",
            )

    def test_missing_so_that_raises(self):
        with pytest.raises(ValidationError, match="VAL-020"):
            _make_story(
                role="player",
                action="jump",
                benefit="reach higher areas",
                full_text="As a player, I want to jump.",
            )

    def test_missing_trailing_period_raises(self):
        with pytest.raises(ValidationError, match="VAL-020"):
            _make_story(
                role="player",
                action="jump",
                benefit="reach higher areas",
                full_text="As a player, I want jump, so that reach higher areas",
            )

    def test_empty_full_text_raises(self):
        with pytest.raises(ValidationError):
            _make_story(full_text="")


# ─────────────────────────────────────────────────────────────────────────────
# Component consistency (VAL-020 cross-check)
# ─────────────────────────────────────────────────────────────────────────────

class TestComponentConsistency:

    def test_mismatched_role_raises(self):
        with pytest.raises(ValidationError, match="VAL-020"):
            _make_story(
                role="game designer",  # role says "game designer"
                action="jump",
                benefit="reach higher areas",
                full_text=(  # but full_text says "player"
                    "As a player, I want jump, so that reach higher areas."
                ),
            )

    def test_mismatched_action_raises(self):
        with pytest.raises(ValidationError, match="VAL-020"):
            _make_story(
                role="player",
                action="jump",
                benefit="reach higher areas",
                full_text=(
                    "As a player, I want run fast, so that reach higher areas."
                ),
            )

    def test_mismatched_benefit_raises(self):
        with pytest.raises(ValidationError, match="VAL-020"):
            _make_story(
                role="player",
                action="jump",
                benefit="reach higher areas",
                full_text=(
                    "As a player, I want jump, so that defeat enemies."
                ),
            )


# ─────────────────────────────────────────────────────────────────────────────
# Field validation
# ─────────────────────────────────────────────────────────────────────────────

class TestFieldValidation:

    def test_empty_role_raises(self):
        with pytest.raises(ValidationError):
            _make_story(role="", action="jump", benefit="reach higher",
                        full_text="As a , I want jump, so that reach higher.")

    def test_empty_action_raises(self):
        with pytest.raises(ValidationError):
            _make_story(role="player", action="", benefit="reach higher",
                        full_text="As a player, I want , so that reach higher.")

    def test_empty_benefit_raises(self):
        with pytest.raises(ValidationError):
            _make_story(role="player", action="jump", benefit="",
                        full_text="As a player, I want jump, so that .")

    def test_empty_draft_id_raises(self):
        with pytest.raises(ValidationError):
            _make_story(draft_id="")

    def test_empty_candidate_id_raises(self):
        with pytest.raises(ValidationError):
            _make_story(candidate_id="")

    def test_empty_source_chunk_id_raises(self):
        with pytest.raises(ValidationError):
            _make_story(source_chunk_id="")

    def test_empty_document_id_raises(self):
        with pytest.raises(ValidationError):
            _make_story(document_id="")

    def test_negative_iteration_count_raises(self):
        with pytest.raises(ValidationError):
            _make_story(iteration_count=-1)

    def test_role_stripped(self):
        story = _make_story(
            role="  player  ",
            action="attack enemies using melee weapons",
            benefit="I can defeat them and progress through the level",
            full_text=(
                "As a player, I want attack enemies using melee weapons, "
                "so that I can defeat them and progress through the level."
            ),
        )
        assert story.role == "player"
