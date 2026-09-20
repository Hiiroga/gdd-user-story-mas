"""
examples/run_reviewer_agent.py
=================================
Example: Preprocess GDD → Reader → Analyst → Generator → Reviewer.
Runs the full pipeline up to user story review on one chunk.

Usage:
    cd "c:\\Data D\\Tugas Akhir\\gdd-userstory-mas"
    $env:PYTHONIOENCODING="utf-8"
    python examples/run_reviewer_agent.py
    python examples/run_reviewer_agent.py datasets/raw/doombible.pdf --chunk 3
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
        description="Run Reader → Analyst → Generator → Reviewer on one GDD chunk."
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
        print("ERROR: LLM_API_KEY not set.")
        sys.exit(1)

    config_path = project_root / args.config

    # ── Step 1: Preprocessing ─────────────────────────────────────────────────
    print(f"\n[1/6] Preprocessing: {gdd_path.name}")
    from gdd_userstory_mas.preprocessing.pipeline import (
        PreprocessingConfig, PreprocessingPipeline,
    )
    pre_result = PreprocessingPipeline(
        config=PreprocessingConfig.from_yaml(project_root / "config" / "pipeline.yaml")
    ).run(gdd_path)
    total_chunks = len(pre_result.chunks)
    print(f"    Chunks: {total_chunks}")

    if args.chunk >= total_chunks:
        print(f"ERROR: chunk {args.chunk} >= total {total_chunks}")
        sys.exit(1)

    chunk = pre_result.chunks[args.chunk]
    print(f"    Chunk [{args.chunk}]: tokens={chunk.token_count}")

    # ── Step 2: Reader Agent ──────────────────────────────────────────────────
    print("\n[2/6] Reader Agent…")
    from gdd_userstory_mas.mas.reader_agent import GDDReaderAgent, ReaderAgentConfig
    from gdd_userstory_mas.mas.reader_output_validator import validate_reader_output

    reader_agent = GDDReaderAgent(
        config=ReaderAgentConfig.from_yaml(config_path, project_root=project_root),
        api_key=api_key, project_root=project_root,
    )
    reader_output = reader_agent.process_chunk(chunk, total_chunks=total_chunks)
    r_report = validate_reader_output(reader_output, chunk)
    print(f"    Validation: {'PASSED' if r_report.passed else 'FAILED'}")
    print(f"    Items: {reader_output.category_counts()}")

    # ── Step 3: Analyst Agent ─────────────────────────────────────────────────
    print("\n[3/6] Analyst Agent…")
    from gdd_userstory_mas.mas.analyst_agent import AnalystAgentConfig, RequirementsAnalystAgent
    from gdd_userstory_mas.mas.analyst_output_validator import validate_analyst_output

    analyst_agent = RequirementsAnalystAgent(
        config=AnalystAgentConfig.from_yaml(config_path, project_root=project_root),
        api_key=api_key, project_root=project_root,
    )
    analyst_output = analyst_agent.process_reader_output(
        reader_output, id_offset=0, total_chunks=total_chunks
    )
    a_report = validate_analyst_output(analyst_output, reader_output)
    print(f"    Validation: {'PASSED' if a_report.passed else 'FAILED'}")
    print(f"    Candidates: {analyst_output.candidate_count}")

    if analyst_output.is_empty:
        print("\n    No candidates found — nothing to generate. Exiting.")
        return

    # ── Step 4: Generator Agent ───────────────────────────────────────────────
    print("\n[4/6] Generator Agent (one story per candidate)…")
    from gdd_userstory_mas.mas.generator_agent import (
        GeneratorAgentConfig, UserStoryGeneratorAgent,
    )
    from gdd_userstory_mas.mas.generator_output_validator import validate_generator_output

    generator_agent = UserStoryGeneratorAgent(
        config=GeneratorAgentConfig.from_yaml(config_path, project_root=project_root),
        api_key=api_key, project_root=project_root,
    )

    stories = []
    for candidate in analyst_output.candidates:
        story = generator_agent.generate(candidate, iteration_count=0)
        g_report = validate_generator_output(story, candidate)
        stories.append((story, g_report))
        status = "✓" if g_report.passed else "✗"
        print(f"    [{status}] {story.draft_id}: {story.full_text[:70]}…")

    # ── Step 5: Reviewer Agent ────────────────────────────────────────────────
    print("\n[5/6] Reviewer Agent (one review per story)…")
    from gdd_userstory_mas.mas.reviewer_agent import ReviewerAgent, ReviewerAgentConfig
    from gdd_userstory_mas.mas.reviewer_output_validator import validate_reviewer_output

    reviewer_agent = ReviewerAgent(
        config=ReviewerAgentConfig.from_yaml(config_path, project_root=project_root),
        api_key=api_key, project_root=project_root,
    )

    reviews = []
    for story, _ in stories:
        result, feedback = reviewer_agent.review(story, chunk.text)
        v_report = validate_reviewer_output(result, feedback, story)
        reviews.append((story, result, feedback, v_report))
        icon = "✓" if result.status == "Valid" else "✗"
        print(f"    [{icon}] {story.draft_id}: {result.status}")
        if feedback:
            print(f"        Feedback: {feedback.feedback_text[:80]}…")

    # ── Step 6: Results ───────────────────────────────────────────────────────
    valid_count = sum(1 for _, r, _, _ in reviews if r.status == "Valid")
    invalid_count = sum(1 for _, r, _, _ in reviews if r.status == "Invalid")
    print(f"\n[6/6] Results — {valid_count} Valid, {invalid_count} Invalid "
          f"out of {len(reviews)} stories")
    print("\n" + "═" * 70)
    print("  REVIEW RESULTS")
    print("═" * 70)

    for i, (story, result, feedback, v_report) in enumerate(reviews):
        print(f"\n  [{i+1}] {story.draft_id}")
        print(f"  Story   : {story.full_text}")
        print(f"  Verdict : {result.status}")
        if result.failing_criteria:
            print(f"  Failed  : {', '.join(result.failing_criteria)}")
        if feedback:
            print(f"  Feedback: {feedback.feedback_text}")
            if feedback.unsupported_claim_excerpt:
                print(f"  Unsupported: {feedback.unsupported_claim_excerpt}")
        val_status = "PASSED" if v_report.passed else "FAILED"
        print(f"  Validation: {val_status}")
        if v_report.warnings:
            for w in v_report.warnings:
                print(f"  WARN: {w}")

    print("\n" + "═" * 70)
    print("  Full JSON:")
    print("═" * 70)
    output = []
    for story, result, feedback, _ in reviews:
        entry = {
            "story": story.model_dump(),
            "review": result.model_dump(),
        }
        if feedback:
            entry["feedback"] = feedback.model_dump()
        output.append(entry)
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
