"""
examples/run_reader_agent.py
==============================
Example script showing how to run the GDD Reader Agent on one chunk
from a preprocessed GDD document.

This is a MANUAL TESTING / DEMONSTRATION script only.
It requires a real LLM API key and a GDD file.

Usage:
    cd "c:\\Data D\\Tugas Akhir\\gdd-userstory-mas"
    $env:PYTHONIOENCODING="utf-8"
    python examples/run_reader_agent.py
    python examples/run_reader_agent.py datasets/raw/doombible.pdf --chunk 3
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Ensure project src is on path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the GDD Reader Agent on one chunk from a GDD file."
    )
    parser.add_argument(
        "gdd_file",
        nargs="?",
        default="datasets/raw/doombible.pdf",
        help="Path to the GDD file (default: datasets/raw/doombible.pdf)",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=0,
        help="Zero-based chunk index to analyze (default: 0)",
    )
    parser.add_argument(
        "--config",
        default="config/mas.yaml",
        help="Path to MAS config YAML (default: config/mas.yaml)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.resolve()
    gdd_path = Path(args.gdd_file)
    if not gdd_path.is_absolute():
        gdd_path = project_root / gdd_path

    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        print("ERROR: LLM_API_KEY not found in environment / .env file.")
        sys.exit(1)

    # ── Step 1: Preprocessing (shared with MAS and Baseline) ──────────────────
    print(f"\n[1/3] Preprocessing: {gdd_path.name}")
    from gdd_userstory_mas.preprocessing.pipeline import (
        PreprocessingConfig,
        PreprocessingPipeline,
    )

    pre_config = PreprocessingConfig.from_yaml(project_root / "config" / "pipeline.yaml")
    pipeline = PreprocessingPipeline(config=pre_config)
    pre_result = pipeline.run(gdd_path)

    total_chunks = len(pre_result.chunks)
    print(f"    Document ID : {pre_result.document.document_id}")
    print(f"    Chunks      : {total_chunks}")

    if args.chunk >= total_chunks:
        print(
            f"ERROR: --chunk {args.chunk} is out of range "
            f"(document has {total_chunks} chunks, max index = {total_chunks - 1})."
        )
        sys.exit(1)

    chunk = pre_result.chunks[args.chunk]
    print(f"    Analyzing   : chunk[{args.chunk}] — '{chunk.chapter or 'Unknown'}' / '{chunk.section or 'Unknown'}'")
    print(f"    Tokens      : {chunk.token_count}")
    print(f"    Preview     : {chunk.text[:120].replace(chr(10), ' ')}...")

    # ── Step 2: Reader Agent ───────────────────────────────────────────────────
    print(f"\n[2/3] Running GDD Reader Agent (model: see {args.config})")
    from gdd_userstory_mas.mas.reader_agent import GDDReaderAgent, ReaderAgentConfig
    from gdd_userstory_mas.mas.reader_output_validator import validate_reader_output

    config = ReaderAgentConfig.from_yaml(
        project_root / args.config,
        project_root=project_root,
    )
    agent = GDDReaderAgent(config=config, api_key=api_key, project_root=project_root)

    reader_output = agent.process_chunk(chunk, total_chunks=total_chunks)

    # ── Step 3: Validate ───────────────────────────────────────────────────────
    print("\n[3/3] Validating output...")
    report = validate_reader_output(reader_output, chunk)
    print(f"    Validation  : {'PASSED' if report.passed else 'FAILED'}")
    for w in report.warnings:
        print(f"    WARN : {w}")
    for e in report.errors:
        print(f"    ERROR: {e}")

    # ── Print results ──────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  READER OUTPUT")
    print("═" * 70)
    print(f"  Chunk ID    : {reader_output.chunk_id}")
    print(f"  Document ID : {reader_output.document_id}")
    print(f"  Total items : {reader_output.total_items}")
    print(f"  Counts      : {reader_output.category_counts()}")

    if reader_output.themes:
        print(f"\n  Themes      : {', '.join(reader_output.themes)}")

    for category in ("gameplay_elements", "systems", "characters", "ui_elements", "narrative"):
        items = getattr(reader_output, category)
        if items:
            print(f"\n  {category.replace('_', ' ').title()} ({len(items)}):")
            for item in items:
                print(f"    • {item.content}")
                if item.source_excerpt:
                    excerpt = item.source_excerpt
                    if len(excerpt) > 80:
                        excerpt = excerpt[:77] + "..."
                    print(f"      ↳ \"{excerpt}\"")

    print("\n" + "═" * 70)
    print(f"  Full JSON output:")
    print("═" * 70)
    print(json.dumps(reader_output.model_dump(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
