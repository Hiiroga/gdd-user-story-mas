"""
examples/run_redundancy_agent.py
===================================
Example: Run the full MAS pipeline up to Redundancy Checking.
Preprocesses ALL chunks, runs Reader → Analyst → Generator → Reviewer
on each chunk, collects all Valid stories, then runs the Redundancy
Checker on the complete collection.

Usage:
    cd "c:\\Data D\\Tugas Akhir\\gdd-userstory-mas"
    $env:PYTHONIOENCODING="utf-8"
    python examples/run_redundancy_agent.py
    python examples/run_redundancy_agent.py datasets/raw/doombible.pdf --max-chunks 3
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
        description="Run full MAS pipeline up to Redundancy Checker."
    )
    parser.add_argument("gdd_file", nargs="?", default="datasets/raw/doombible.pdf")
    parser.add_argument("--max-chunks", type=int, default=3,
                        help="Max chunks to process (default 3 for demo)")
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
    process_count = min(total_chunks, args.max_chunks)
    print(f"    Total chunks: {total_chunks}, processing: {process_count}")

    # ── Steps 2-5: Per-chunk pipeline ─────────────────────────────────────────
    from gdd_userstory_mas.mas.reader_agent import GDDReaderAgent, ReaderAgentConfig
    from gdd_userstory_mas.mas.analyst_agent import AnalystAgentConfig, RequirementsAnalystAgent
    from gdd_userstory_mas.mas.generator_agent import GeneratorAgentConfig, UserStoryGeneratorAgent
    from gdd_userstory_mas.mas.reviewer_agent import ReviewerAgent, ReviewerAgentConfig

    reader_agent = GDDReaderAgent(
        config=ReaderAgentConfig.from_yaml(config_path, project_root=project_root),
        api_key=api_key, project_root=project_root,
    )
    analyst_agent = RequirementsAnalystAgent(
        config=AnalystAgentConfig.from_yaml(config_path, project_root=project_root),
        api_key=api_key, project_root=project_root,
    )
    generator_agent = UserStoryGeneratorAgent(
        config=GeneratorAgentConfig.from_yaml(config_path, project_root=project_root),
        api_key=api_key, project_root=project_root,
    )
    reviewer_agent = ReviewerAgent(
        config=ReviewerAgentConfig.from_yaml(config_path, project_root=project_root),
        api_key=api_key, project_root=project_root,
    )

    valid_stories = []
    id_offset = 0

    for ci in range(process_count):
        chunk = pre_result.chunks[ci]
        print(f"\n[2-5] Chunk [{ci}] (tokens={chunk.token_count})")

        # Reader
        print("    Reader…", end=" ")
        reader_output = reader_agent.process_chunk(chunk, total_chunks=total_chunks)
        print(f"items={reader_output.category_counts()}")

        # Analyst
        print("    Analyst…", end=" ")
        analyst_output = analyst_agent.process_reader_output(
            reader_output, id_offset=id_offset, total_chunks=total_chunks
        )
        id_offset += analyst_output.candidate_count
        print(f"candidates={analyst_output.candidate_count}")

        if analyst_output.is_empty:
            print("    No candidates — skipping.")
            continue

        # Generator + Reviewer per candidate
        for candidate in analyst_output.candidates:
            story = generator_agent.generate(candidate, iteration_count=0)
            result, feedback = reviewer_agent.review(story, chunk.text)
            icon = "✓" if result.status == "Valid" else "✗"
            print(f"    [{icon}] {story.draft_id}: {result.status}")
            if result.status == "Valid":
                valid_stories.append(story)
            elif feedback:
                print(f"        → {feedback.feedback_text[:60]}…")

    print(f"\n{'═' * 70}")
    print(f"  Collected {len(valid_stories)} Valid stories from {process_count} chunks")
    print(f"{'═' * 70}")

    if len(valid_stories) == 0:
        print("\n    No valid stories — nothing to check for redundancy. Exiting.")
        return

    # ── Step 6: Redundancy Checker ────────────────────────────────────────────
    print(f"\n[6/6] Redundancy Checker (analyzing {len(valid_stories)} stories)…")
    from gdd_userstory_mas.mas.redundancy_agent import (
        RedundancyAgentConfig, RedundancyCheckerAgent,
    )
    from gdd_userstory_mas.mas.redundancy_output_validator import validate_redundancy_output

    redundancy_agent = RedundancyCheckerAgent(
        config=RedundancyAgentConfig.from_yaml(config_path, project_root=project_root),
        api_key=api_key, project_root=project_root,
    )

    run_id = f"demo_{gdd_path.stem}"
    analysis = redundancy_agent.check(valid_stories, run_id)
    v_report = validate_redundancy_output(analysis, valid_stories)

    print(f"\n    Status       : {analysis.resolution_status}")
    print(f"    Input stories: {analysis.input_story_count}")
    print(f"    Dup groups   : {len(analysis.duplicate_groups)}")
    print(f"    Merge actions: {len(analysis.merge_actions)}")
    print(f"    Resolved     : {analysis.resolved_story_count}")
    print(f"    Validation   : {'PASSED' if v_report.passed else 'FAILED'}")

    if analysis.duplicate_groups:
        print(f"\n{'─' * 70}")
        print("  DUPLICATE GROUPS")
        print(f"{'─' * 70}")
        for group in analysis.duplicate_groups:
            print(f"\n  Group {group.group_id}:")
            print(f"    Stories: {group.story_ids}")
            print(f"    Basis  : {group.similarity_basis}")

    if analysis.merge_actions:
        print(f"\n{'─' * 70}")
        print("  RESOLUTION ACTIONS")
        print(f"{'─' * 70}")
        for action in analysis.merge_actions:
            print(f"\n  Group {action.group_id}: {action.action}")
            print(f"    Kept   : {action.kept_story_id}")
            print(f"    Removed: {action.removed_story_ids}")
            print(f"    Reason : {action.reason}")
            if action.action == "merge" and action.merged_full_text:
                print(f"    Merged : {action.merged_full_text}")

    # Unique stories
    print(f"\n{'─' * 70}")
    print(f"  UNIQUE STORIES ({len(analysis.unique_story_ids)})")
    print(f"{'─' * 70}")
    for uid in analysis.unique_story_ids:
        story = next((s for s in valid_stories if s.draft_id == uid), None)
        if story:
            print(f"  {uid}: {story.full_text[:80]}…")

    print(f"\n{'═' * 70}")
    print("  Full JSON:")
    print(f"{'═' * 70}")
    print(json.dumps(analysis.model_dump(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
