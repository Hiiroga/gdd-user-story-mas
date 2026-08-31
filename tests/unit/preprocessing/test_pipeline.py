"""
tests/unit/preprocessing/test_pipeline.py
==========================================
Unit tests for the PreprocessingPipeline orchestrator.

These tests mock the individual step functions so the pipeline's
wiring and error-handling behaviour is tested in isolation.  All tests
are offline — no LLM, no PDF library required.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gdd_userstory_mas.preprocessing.pipeline import (
    PreprocessingConfig,
    PreprocessingPipeline,
    PreprocessingResult,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def txt_fixture(tmp_path: Path) -> Path:
    p = tmp_path / "sample.txt"
    p.write_text(
        "# Overview\n\nThis is an action-RPG about exploring dungeons.\n"
        "The player fights enemies and collects loot.\n\n"
        "## Combat\n\nPlayers use combos to defeat enemies.\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture()
def pdf_fixture(tmp_path: Path) -> Path:
    p = tmp_path / "sample.pdf"
    p.write_bytes(b"%PDF-1.4\n%%EOF")
    return p


@pytest.fixture()
def config() -> PreprocessingConfig:
    return PreprocessingConfig.defaults()


# ── PreprocessingConfig ────────────────────────────────────────────────────────

class TestPreprocessingConfig:
    def test_defaults_are_sensible(self) -> None:
        config = PreprocessingConfig.defaults()
        assert config.max_chunk_tokens == 1500
        assert config.overlap_tokens == 0
        assert config.sentence_splitter == "nltk"
        assert len(config.heading_patterns) > 0

    def test_from_yaml_loads_pipeline_yaml(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "pipeline.yaml"
        yaml_path.write_text(
            "chunking:\n  max_chunk_tokens: 800\n  overlap_tokens: 50\n"
            "segmentation:\n  min_heading_length: 3\n  max_heading_length: 100\n"
            "  heading_patterns: []\n"
            "language_detection:\n  reject_non_english: false\n",
            encoding="utf-8",
        )
        config = PreprocessingConfig.from_yaml(yaml_path)
        assert config.max_chunk_tokens == 800
        assert config.overlap_tokens == 50
        assert config.min_heading_length == 3


# ── Successful .txt run ────────────────────────────────────────────────────────

class TestTxtPipelineSuccess:
    def test_run_returns_preprocessing_result(self, txt_fixture: Path, config: PreprocessingConfig) -> None:
        pipeline = PreprocessingPipeline(config=config)
        result = pipeline.run(txt_fixture)
        assert isinstance(result, PreprocessingResult)

    def test_document_id_is_set(self, txt_fixture: Path, config: PreprocessingConfig) -> None:
        pipeline = PreprocessingPipeline(config=config)
        result = pipeline.run(txt_fixture)
        assert result.document.document_id
        assert result.metadata.document_id == result.document.document_id

    def test_chunks_are_produced(self, txt_fixture: Path, config: PreprocessingConfig) -> None:
        pipeline = PreprocessingPipeline(config=config)
        result = pipeline.run(txt_fixture)
        assert len(result.chunks) >= 1

    def test_all_chunks_have_document_id(self, txt_fixture: Path, config: PreprocessingConfig) -> None:
        pipeline = PreprocessingPipeline(config=config)
        result = pipeline.run(txt_fixture)
        for chunk in result.chunks:
            assert chunk.document_id == result.document.document_id

    def test_chunks_have_unique_ids(self, txt_fixture: Path, config: PreprocessingConfig) -> None:
        pipeline = PreprocessingPipeline(config=config)
        result = pipeline.run(txt_fixture)
        chunk_ids = [c.chunk_id for c in result.chunks]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_chunks_are_ordered_by_position(self, txt_fixture: Path, config: PreprocessingConfig) -> None:
        pipeline = PreprocessingPipeline(config=config)
        result = pipeline.run(txt_fixture)
        positions = [c.position for c in result.chunks]
        assert positions == sorted(positions)

    def test_custom_document_id_is_respected(self, txt_fixture: Path, config: PreprocessingConfig) -> None:
        pipeline = PreprocessingPipeline(config=config)
        result = pipeline.run(txt_fixture, document_id="my-doc-001")
        assert result.document.document_id == "my-doc-001"

    def test_no_errors_for_valid_input(self, txt_fixture: Path, config: PreprocessingConfig) -> None:
        pipeline = PreprocessingPipeline(config=config)
        result = pipeline.run(txt_fixture)
        assert result.errors == []

    def test_structure_detected_for_headings(self, txt_fixture: Path, config: PreprocessingConfig) -> None:
        pipeline = PreprocessingPipeline(config=config)
        result = pipeline.run(txt_fixture)
        # The fixture has headings, so structure should be detected
        assert result.metadata.structure_detected is True

    def test_table_of_contents_populated(self, txt_fixture: Path, config: PreprocessingConfig) -> None:
        pipeline = PreprocessingPipeline(config=config)
        result = pipeline.run(txt_fixture)
        assert len(result.metadata.table_of_contents) >= 1


# ── Output persistence ─────────────────────────────────────────────────────────

class TestOutputPersistence:
    def test_output_files_created(self, txt_fixture: Path, config: PreprocessingConfig, tmp_path: Path) -> None:
        output_dir = tmp_path / "outputs" / "run-001"
        pipeline = PreprocessingPipeline(config=config, output_dir=output_dir)
        pipeline.run(txt_fixture)

        pre_dir = output_dir / "01_preprocessing"
        assert (pre_dir / "chunks.json").exists()
        assert (pre_dir / "gdd_document.json").exists()
        assert (pre_dir / "gdd_metadata.json").exists()

    def test_chunks_json_is_valid_json(self, txt_fixture: Path, config: PreprocessingConfig, tmp_path: Path) -> None:
        import json
        output_dir = tmp_path / "outputs" / "run-002"
        pipeline = PreprocessingPipeline(config=config, output_dir=output_dir)
        pipeline.run(txt_fixture)

        chunks_path = output_dir / "01_preprocessing" / "chunks.json"
        data = json.loads(chunks_path.read_text(encoding="utf-8"))
        assert isinstance(data, list)
        assert len(data) >= 1
        # Each entry should have required GDDChunk fields
        assert "chunk_id" in data[0]
        assert "document_id" in data[0]
        assert "text" in data[0]


# ── Error handling ─────────────────────────────────────────────────────────────

class TestErrorHandling:
    def test_raises_on_missing_file(self, config: PreprocessingConfig, tmp_path: Path) -> None:
        pipeline = PreprocessingPipeline(config=config)
        with pytest.raises(RuntimeError, match="Ingestion failed"):
            pipeline.run(tmp_path / "nonexistent.txt")

    def test_raises_on_unsupported_extension(self, config: PreprocessingConfig, tmp_path: Path) -> None:
        bad = tmp_path / "document.docx"
        bad.write_text("content", encoding="utf-8")
        pipeline = PreprocessingPipeline(config=config)
        with pytest.raises(RuntimeError, match="Ingestion failed"):
            pipeline.run(bad)

    def test_raises_on_empty_file(self, config: PreprocessingConfig, tmp_path: Path) -> None:
        empty = tmp_path / "empty.txt"
        empty.write_bytes(b"")
        pipeline = PreprocessingPipeline(config=config)
        with pytest.raises(RuntimeError):
            pipeline.run(empty)


# ── PDF pipeline (mocked) ──────────────────────────────────────────────────────

class TestPDFPipelineMocked:
    def test_pdf_pipeline_calls_convert(self, pdf_fixture: Path, config: PreprocessingConfig) -> None:
        """Verify that PDF files route through the PDF converter."""
        mock_pdf_result = MagicMock()
        mock_pdf_result.text = (
            "# Overview\n\nThis game features combat and exploration.\n"
        )
        mock_pdf_result.page_count = 1
        mock_pdf_result.failed_pages = []
        mock_pdf_result.warnings = []

        with patch(
            "gdd_userstory_mas.preprocessing.pipeline.convert_pdf_to_text",
            return_value=mock_pdf_result,
        ) as mock_convert:
            pipeline = PreprocessingPipeline(config=config)
            result = pipeline.run(pdf_fixture)

        mock_convert.assert_called_once()
        assert isinstance(result, PreprocessingResult)
        assert len(result.chunks) >= 1

    def test_pdf_conversion_failure_raises_runtime_error(
        self, pdf_fixture: Path, config: PreprocessingConfig
    ) -> None:
        from gdd_userstory_mas.preprocessing.pdf_converter import PDFConversionError

        with patch(
            "gdd_userstory_mas.preprocessing.pipeline.convert_pdf_to_text",
            side_effect=PDFConversionError("Image-only PDF"),
        ):
            pipeline = PreprocessingPipeline(config=config)
            with pytest.raises(RuntimeError, match="PDF conversion failed"):
                pipeline.run(pdf_fixture)


# ── Fixture-based end-to-end smoke test ───────────────────────────────────────

class TestFixtureSmoke:
    def test_full_pipeline_on_sample_gdd(self) -> None:
        fixture = Path(__file__).parent.parent.parent / "fixtures" / "sample_gdd_small.txt"
        config = PreprocessingConfig.defaults()
        pipeline = PreprocessingPipeline(config=config)
        result = pipeline.run(fixture)

        assert len(result.chunks) >= 1
        assert result.metadata.segment_count >= 1
        # All chunks within the token limit
        for chunk in result.chunks:
            assert chunk.token_count <= config.max_chunk_tokens
        # Traceability: every chunk has document_id
        for chunk in result.chunks:
            assert chunk.document_id == result.document.document_id
