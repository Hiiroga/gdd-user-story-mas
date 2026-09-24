"""
examples/run_evaluator_agent.py
==================================
Example: Run the full MAS pipeline up to Evaluation.
Reader → Analyst → Generator → Reviewer → Redundancy Checker → Evaluator

Usage:
    cd "c:\\Data D\\Tugas Akhir\\gdd-userstory-mas"
    $env:PYTHONIOENCODING="utf-8"
    python examples/run_evaluator_agent.py
    python examples/run_evaluator_agent.py datasets/raw/doombible.pdf --max-chunks 3
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
        description="Run full MAS pipeline up to Evaluator Agent."
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
    print(f"\n[1/7] Preprocessing: {gdd_path.name}")
    from gdd_userstory_mas.preprocessing.pipeline import (
        PreprocessingConfig, PreprocessingPipeline,
    )
    pre_result = PreprocessingPipeline(
        config=PreprocessingConfig.from_yaml(project_root / "config" / "pipeline.yaml")
    ).run(gdd_path)
    total_chunks = len(pre_result.chunks)
    process_count = min(total_chunks, args.max_chunks)
    print(f"    Total chunks: {total_chunks}, processing: {process_count}")

    # Build GDD chunk text map (all chunks — Evaluator needs full map)
    gdd_chunk_texts = {c.chunk_id: c.text for c in pre_result.chunks}

    # ── Per-chunk pipeline ────────────────────────────────────────────────────
    from gdd_userstory_mas.mas.reader_agent import GDDReaderAgent, ReaderAgentConfig
    from gdd_userstory_mas.mas.analyst_agent import AnalystAgentConfig, RequirementsAnalystAgent
    from gdd_userstory_mas.mas.generator_agent import GeneratorAgentConfig, UserStoryGeneratorAgent
    from gdd_userstory_mas.mas.reviewer_agent import ReviewerAgent, ReviewerAgentConfig
    from gdd_userstory_mas.schemas.final_user_story import (
        ConfidenceEvidence, FinalUserStory,
    )

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

    valid_stories = []      # GeneratedUserStory (for Redundancy Checker)
    valid_stories_rev = {}  # draft_id → reviewer_result (for traceability)
    id_offset = 0

    for ci in range(process_count):
        chunk = pre_result.chunks[ci]
        print(f"\n[2-5] Chunk [{ci}] (tokens={chunk.token_count})")

        reader_output = reader_agent.process_chunk(chunk, total_chunks=total_chunks)
        analyst_output = analyst_agent.process_reader_output(
            reader_output, id_offset=id_offset, total_chunks=total_chunks
        )
        id_offset += analyst_output.candidate_count

        if analyst_output.is_empty:
            print("    No candidates — skipping.")
            continue

        for candidate in analyst_output.candidates:
            story = generator_agent.generate(candidate, iteration_count=0)
            result, feedback = reviewer_agent.review(story, chunk.text)
            icon = "✓" if result.status == "Valid" else "✗"
            print(f"    [{icon}] {story.draft_id}: {result.status}")
            if result.status == "Valid":
                valid_stories.append(story)
                valid_stories_rev[story.draft_id] = result

    print(f"\n{'═'*70}")
    print(f"  Collected {len(valid_stories)} Valid stories from {process_count} chunks")
    print(f"{'═'*70}")

    # ── Step 6: Redundancy Checker ────────────────────────────────────────────
    redundancy_analysis = None
    if valid_stories:
        print(f"\n[6/7] Redundancy Checker ({len(valid_stories)} stories)…")
        from gdd_userstory_mas.mas.redundancy_agent import (
            RedundancyAgentConfig, RedundancyCheckerAgent,
        )
        redundancy_agent = RedundancyCheckerAgent(
            config=RedundancyAgentConfig.from_yaml(config_path, project_root=project_root),
            api_key=api_key, project_root=project_root,
        )
        run_id = f"demo_{gdd_path.stem}"
        redundancy_analysis = redundancy_agent.check(valid_stories, run_id)
        print(f"    Status: {redundancy_analysis.resolution_status}")
        print(f"    Groups: {len(redundancy_analysis.duplicate_groups)}")
        print(f"    Resolved: {redundancy_analysis.resolved_story_count}")
    else:
        run_id = f"demo_{gdd_path.stem}"
        print("\n[6/7] Redundancy Checker — skipped (no valid stories).")

    # ── Build FinalUserStory list ──────────────────────────────────────────────
    # For demo: convert GeneratedUserStory → FinalUserStory (simplified)
    print(f"\n    Building FinalUserStory collection…")
    final_stories = []
    for i, gs in enumerate(valid_stories):
        rev_result = valid_stories_rev.get(gs.draft_id)
        traceability = None
        if rev_result:
            # Extract gdd_traceability criterion from reviewer result
            from gdd_userstory_mas.schemas.reviewer_result import CriterionType
            trac_criterion = next(
                (c for c in (rev_result.criteria_results or [])
                 if c.criterion == CriterionType.GDD_TRACEABILITY),
                None,
            )
            if trac_criterion:
                traceability = trac_criterion.passed

        confidence = ConfidenceEvidence(
            reviewer_traceability_confirmed=traceability
        )
        fs = FinalUserStory(
            id=f"US-{i+1:04d}",
            run_id=run_id,
            pipeline_type="MAS",
            source_document_id=gs.document_id,
            source_chunk_ids=[gs.source_chunk_id],
            role=gs.role,
            action=gs.action,
            benefit=gs.benefit,
            full_text=gs.full_text,
            requirement_type="player",   # default for demo
            game_domain="gameplay",       # default for demo
            validation_status="Valid",
            confidence_evidence=confidence,
            iteration_count_final=gs.iteration_count,
        )
        final_stories.append(fs)

    # ── Step 7: Evaluator Agent ───────────────────────────────────────────────
    print(f"\n[7/7] Evaluator Agent ({len(final_stories)} final stories)…")
    from gdd_userstory_mas.mas.evaluator_agent import (
        EvaluatorAgentConfig, EvaluatorAgent,
    )
    from gdd_userstory_mas.mas.evaluator_output_validator import validate_evaluator_output

    evaluator = EvaluatorAgent(
        config=EvaluatorAgentConfig.from_yaml(config_path, project_root=project_root),
        api_key=api_key,
        project_root=project_root,
    )

    eval_result = evaluator.evaluate(
        stories=final_stories,
        gdd_chunks=gdd_chunk_texts,
        run_id=run_id,
        document_id=gdd_path.stem,
        pipeline_type="MAS",
        total_chunk_count=total_chunks,
        redundancy_analysis=redundancy_analysis,
    )

    v_report = validate_evaluator_output(eval_result, final_stories)

    # ── Print Report ──────────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print("  EVALUATION REPORT")
    print(f"{'═'*70}")
    print(f"  Evaluation ID : {eval_result.evaluation_id}")
    print(f"  Run ID        : {eval_result.run_id}")
    print(f"  Document      : {eval_result.document_id}")
    print(f"  Pipeline      : {eval_result.pipeline_type}")
    print(f"  Total Stories : {eval_result.total_stories}")
    print(f"  Computed At   : {eval_result.computed_at}")
    print()
    print(f"{'─'*70}")
    print("  THESIS METRICS")
    print(f"{'─'*70}")

    def _fmt(val, suffix="%"):
        return f"{val:.2f}{suffix}" if val is not None else "null"

    print(f"  EVAL-003 AQUSA Score     : {_fmt(eval_result.aqusa_score)}")
    print(f"  EVAL-004 Coverage        : {_fmt(eval_result.coverage)}")
    print(f"  EVAL-005 Diversity       : {_fmt(eval_result.diversity, '')}")
    print(f"  EVAL-002 Redundancy Rate : {_fmt(eval_result.redundancy_rate)}")
    print(f"  EVAL-001 Hallucination   : {_fmt(eval_result.hallucination_rate)}")

    if eval_result.aqusa_detail:
        d = eval_result.aqusa_detail
        print(f"\n  AQUSA Detail:")
        print(f"    Total: {d.total_stories}  Passing: {d.stories_passing_all}")
        print(f"    Role empty: {d.role_empty_count}  "
              f"Action empty: {d.action_empty_count}  "
              f"Benefit empty: {d.benefit_empty_count}")
        print(f"    Format violated: {d.format_violated_count}  "
              f"Ambiguous: {d.ambiguity_heuristic_count}")

    if eval_result.null_reasons:
        print(f"\n  Null Reasons:")
        for k, v in eval_result.null_reasons.items():
            print(f"    {k}: {v}")

    if eval_result.consistency_notes:
        print(f"\n  Consistency Notes:")
        for n in eval_result.consistency_notes:
            print(f"    • {n}")

    print(f"\n  Metric Details:")
    for name, md in eval_result.metric_details.items():
        print(f"\n  [{name}]")
        print(f"    Formula : {md.formula}")
        if md.numerator is not None:
            print(f"    Numerator/Denominator: {md.numerator}/{md.denominator}")
        print(f"    Value   : {md.value}")
        print(f"    Interp  : {md.interpretation[:100]}")
        print(f"    Limit   : {md.limitation[:80]}")

    print(f"\n{'─'*70}")
    print(f"  Validation: {'PASSED' if v_report.passed else 'FAILED'}")
    for e in v_report.errors:
        print(f"    ERROR: {e}")
    for w in v_report.warnings:
        print(f"    WARN : {w}")

    print(f"\n{'═'*70}")
    print("  Full JSON:")
    print(f"{'═'*70}")
    print(json.dumps(eval_result.model_dump(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
