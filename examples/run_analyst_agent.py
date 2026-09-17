"""
examples/run_analyst_agent.py
==============================
Example script: preprocess a GDD → run Reader Agent → run Analyst Agent
on one chunk, validate, and print candidate requirements.

Usage:
    cd "c:\\Data D\\Tugas Akhir\\gdd-userstory-mas"
    $env:PYTHONIOENCODING="utf-8"
    python examples/run_analyst_agent.py
    python examples/run_analyst_agent.py datasets/raw/doombible.pdf --chunk 3
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

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
        description="Run Reader + Analyst Agent on one chunk of a GDD file."
    )
    parser.add_argument("gdd_file", nargs="?", default="datasets/raw/doombible.pdf")
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--config", default="config/mas.yaml")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.resolve()
    gdd_path = Path(args.gdd_file)
    if not gdd_path.is_absolute():
        gdd_path = project_root / gdd_path

    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        print("ERROR: LLM_API_KEY not found in environment / .env file.")
        sys.exit(1)

    # ── Step 1: Preprocessing ─────────────────────────────────────────────────
    print(f"\n[1/4] Preprocessing: {gdd_path.name}")
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
        print(f"ERROR: chunk {args.chunk} >= total {total_chunks}")
        sys.exit(1)

    chunk = pre_result.chunks[args.chunk]
    print(f"    Chunk [{args.chunk}] : '{chunk.chapter or 'Unknown'}' / tokens={chunk.token_count}")

    # ── Step 2: Reader Agent ──────────────────────────────────────────────────
    print(f"\n[2/4] Running GDD Reader Agent…")
    from gdd_userstory_mas.mas.reader_agent import GDDReaderAgent, ReaderAgentConfig
    from gdd_userstory_mas.mas.reader_output_validator import validate_reader_output

    reader_config = ReaderAgentConfig.from_yaml(
        project_root / args.config, project_root=project_root
    )
    reader_agent = GDDReaderAgent(
        config=reader_config, api_key=api_key, project_root=project_root
    )
    reader_output = reader_agent.process_chunk(chunk, total_chunks=total_chunks)
    r_report = validate_reader_output(reader_output, chunk)
    print(f"    Validation  : {'PASSED' if r_report.passed else 'FAILED'}")
    print(f"    Items found : {reader_output.category_counts()}")

    # ── Step 3: Analyst Agent ─────────────────────────────────────────────────
    print(f"\n[3/4] Running Requirements Analyst Agent…")
    from gdd_userstory_mas.mas.analyst_agent import AnalystAgentConfig, RequirementsAnalystAgent
    from gdd_userstory_mas.mas.analyst_output_validator import validate_analyst_output

    analyst_config = AnalystAgentConfig.from_yaml(
        project_root / args.config, project_root=project_root
    )
    analyst_agent = RequirementsAnalystAgent(
        config=analyst_config, api_key=api_key, project_root=project_root
    )
    analyst_output = analyst_agent.process_reader_output(
        reader_output, id_offset=0, total_chunks=total_chunks
    )

    # ── Step 4: Validate ──────────────────────────────────────────────────────
    print(f"\n[4/4] Validating Analyst output…")
    a_report = validate_analyst_output(analyst_output, reader_output)
    print(f"    Validation  : {'PASSED' if a_report.passed else 'FAILED'}")
    for w in a_report.warnings:
        print(f"    WARN : {w}")
    for e in a_report.errors:
        print(f"    ERROR: {e}")

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  ANALYST OUTPUT")
    print("═" * 70)
    print(f"  Source chunk : {analyst_output.source_chunk_id}")
    print(f"  Candidates   : {analyst_output.candidate_count}")
    print(f"  Perspectives : {analyst_output.perspective_counts()}")
    print(f"  Domains      : {analyst_output.domain_counts()}")

    for i, c in enumerate(analyst_output.candidates):
        print(f"\n  [{i+1}] {c.candidate_id}")
        print(f"  perspective : {c.perspective}  |  domain : {c.domain}")
        print(f"  text        : {c.requirement_text}")
        if c.source_reader_output_ref:
            ref = c.source_reader_output_ref
            print(f"  derived from: \"{ref[:80]}{'…' if len(ref) > 80 else ''}\"")

    print("\n" + "═" * 70)
    print("  Full JSON:")
    print("═" * 70)
    print(json.dumps(
        {
            "source_chunk_id": analyst_output.source_chunk_id,
            "document_id": analyst_output.document_id,
            "candidates": [c.model_dump() for c in analyst_output.candidates],
        },
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
