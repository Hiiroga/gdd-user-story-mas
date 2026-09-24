"""
run_baseline.py
================
Script untuk menjalankan Single-Agent Baseline pipeline pada GDD.

Cara pakai:
    python run_baseline.py                          # pakai file default (doombible.pdf)
    python run_baseline.py datasets/raw/gdd.txt     # file custom

Output akan disimpan di:
    outputs/<run_id>/02_baseline/final_user_stories.json
    outputs/<run_id>/02_baseline/experiment_run.json
    outputs/<run_id>/02_baseline/validation_report.json
    outputs/<run_id>/02_baseline/token_usage.json
"""
import sys
import os
import json
from pathlib import Path
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Load .env ──────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ .env loaded")
except ImportError:
    print("⚠ python-dotenv tidak terinstall, membaca env var dari sistem...")

# ── Cek API key ────────────────────────────────────────────────────────────────
API_KEY = os.getenv("LLM_API_KEY")
PROVIDER = os.getenv("LLM_PROVIDER", "gemini")

if not API_KEY:
    print("❌ ERROR: LLM_API_KEY tidak ditemukan di .env")
    print("   Pastikan file .env ada dan berisi: LLM_API_KEY=<api_key_anda>")
    sys.exit(1)

print(f"✓ Provider: {PROVIDER}")
print(f"✓ API key: {API_KEY[:8]}...{API_KEY[-4:]}")

import argparse

# ── Tentukan file GDD & Argumen ────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Run Single-Agent Baseline Pipeline")
parser.add_argument(
    "gdd_path",
    nargs="?",
    default=Path("datasets/raw/doombible.pdf"),
    type=Path,
    help="Path ke file GDD (default: datasets/raw/doombible.pdf)",
)
parser.add_argument(
    "--max-chunks",
    type=int,
    default=None,
    help="Batasi pemrosesan hanya pada N chunk pertama (misal untuk perbandingan adil dengan MAS)",
)
args = parser.parse_args()

GDD_FILE = args.gdd_path

if not GDD_FILE.exists():
    print(f"❌ ERROR: File GDD tidak ditemukan: {GDD_FILE}")
    print("   Letakkan file GDD di datasets/raw/ atau berikan path sebagai argumen.")
    sys.exit(1)

print(f"✓ GDD file: {GDD_FILE} ({GDD_FILE.stat().st_size // 1024} KB)")

# ── Setup output dir ───────────────────────────────────────────────────────────
run_id = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
OUTPUT_DIR = Path("outputs") / run_id
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"✓ Output dir: {OUTPUT_DIR}")

PROJECT_ROOT = Path(".")

# ── Import pipeline modules ────────────────────────────────────────────────────
print("\n── Memuat modules... ─────────────────────────────────────────────────────")
try:
    from gdd_userstory_mas.preprocessing.pipeline import PreprocessingPipeline, PreprocessingConfig
    from gdd_userstory_mas.baseline.pipeline import BaselinePipeline, BaselineConfig
    print("✓ Modules berhasil dimuat")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Jalankan: python -m pip install -e '.[dev]'")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Step 1: Preprocessing ─────────────────────────────────────────────────")
print(f"   File: {GDD_FILE}")

try:
    pre_config = PreprocessingConfig.from_yaml(PROJECT_ROOT / "config" / "pipeline.yaml")
    preprocessing = PreprocessingPipeline(
        config=pre_config,
        output_dir=OUTPUT_DIR,
    )
    pre_result = preprocessing.run(GDD_FILE)
    if args.max_chunks is not None:
        pre_result.chunks = pre_result.chunks[:args.max_chunks]

    print(f"✓ Preprocessing selesai:")
    print(f"  - Document ID : {pre_result.document.document_id}")
    print(f"  - Chunks      : {len(pre_result.chunks)}")
    print(f"  - Structure   : {'Terdeteksi' if pre_result.metadata.structure_detected else 'Tidak terdeteksi (single segment fallback)'}")
    print(f"  - Segments    : {pre_result.metadata.segment_count}")
    if pre_result.metadata.table_of_contents:
        print(f"  - TOC preview :")
        for entry in pre_result.metadata.table_of_contents[:5]:
            print(f"      {'  ' * (entry.level - 1 if entry.level else 0)}• {entry.title}")
        if len(pre_result.metadata.table_of_contents) > 5:
            print(f"      ... +{len(pre_result.metadata.table_of_contents) - 5} sections")

except Exception as e:
    print(f"❌ Preprocessing gagal: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — BASELINE LLM EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Step 2: Baseline LLM Extraction ──────────────────────────────────────")
print(f"   Provider : {PROVIDER}")
print(f"   Chunks   : {len(pre_result.chunks)} (satu LLM call per chunk)")
print(f"   Estimasi : ~{len(pre_result.chunks)} API call")
print()

try:
    baseline_config = BaselineConfig.from_yaml(
        PROJECT_ROOT / "config" / "baseline.yaml",
        project_root=PROJECT_ROOT,
    )
    # Override provider dari .env (jika berbeda dengan config)
    baseline_config.provider = PROVIDER

    baseline = BaselinePipeline(
        config=baseline_config,
        api_key=API_KEY,
        output_dir=OUTPUT_DIR,
        project_root=PROJECT_ROOT,
    )

    print("   Memulai ekstraksi user stories...")
    result = baseline.run(
        pre_result.chunks,
        document_id=pre_result.document.document_id,
        run_id=run_id,
    )

except Exception as e:
    print(f"❌ Baseline extraction gagal: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# HASIL
# ══════════════════════════════════════════════════════════════════════════════
print("\n═══════════════════════════════════════════════════════════════════════════")
print("  HASIL BASELINE EXTRACTION")
print("═══════════════════════════════════════════════════════════════════════════")
print(f"  Run ID           : {result.run_id}")
print(f"  Document ID      : {result.document_id}")
print(f"  User stories     : {len(result.stories)}")
print(f"  Errors           : {len(result.errors)}")
print(f"  Prompt tokens    : {result.total_prompt_tokens:,}")
print(f"  Completion tokens: {result.total_completion_tokens:,}")
print(f"  Validation       : {'✓ PASSED' if result.validation_report.passed else '⚠ Ada masalah'}")
print()

if result.stories:
    print("  Contoh user stories yang diekstrak:")
    print("  " + "─" * 67)
    for story in result.stories[:10]:
        domain_tag = f"[{story.game_domain}]"
        type_tag = f"[{story.requirement_type}]"
        print(f"\n  {story.id} {domain_tag} {type_tag}")
        print(f"  {story.full_text}")
        if story.confidence_evidence and story.confidence_evidence.source_excerpt:
            excerpt = story.confidence_evidence.source_excerpt[:80]
            print(f"  → Sumber: \"{excerpt}...\"" if len(story.confidence_evidence.source_excerpt) > 80 else f"  → Sumber: \"{story.confidence_evidence.source_excerpt}\"")

    if len(result.stories) > 10:
        print(f"\n  ... dan {len(result.stories) - 10} user stories lainnya")

print()
print(f"  Output tersimpan di: {OUTPUT_DIR / '02_baseline'}")
print()

# Ringkasan per domain
domains: dict = {}
req_types: dict = {}
for s in result.stories:
    domains[s.game_domain] = domains.get(s.game_domain, 0) + 1
    req_types[s.requirement_type] = req_types.get(s.requirement_type, 0) + 1

if domains:
    print("  Distribusi Game Domain:")
    for d, count in sorted(domains.items(), key=lambda x: -x[1]):
        bar = "█" * min(count, 30)
        print(f"    {d:<15} {bar} {count}")

print()
if req_types:
    print("  Distribusi Requirement Type:")
    for rt, count in sorted(req_types.items(), key=lambda x: -x[1]):
        bar = "█" * min(count, 30)
        print(f"    {rt:<20} {bar} {count}")

print()
print("═══════════════════════════════════════════════════════════════════════════")
print(f"  File output:")
print(f"    {OUTPUT_DIR}/02_baseline/final_user_stories.json")
print(f"    {OUTPUT_DIR}/02_baseline/experiment_run.json")
print(f"    {OUTPUT_DIR}/02_baseline/validation_report.json")
print(f"    {OUTPUT_DIR}/02_baseline/token_usage.json")
print("═══════════════════════════════════════════════════════════════════════════")
