"""Eval runner for the Scripbox KB RAG pipeline.

Runs all fixture cases against the live RAG graph and reports:
  - Retrieval Recall@K: expected source URL substring appears in top-K hits
  - Abstention Rate: out-of-domain queries correctly refused
  - Overall pass rate

Usage:
    python -m eval.runner                      # run all fixtures
    python -m eval.runner --fixtures kyc       # run only kyc fixtures
    python -m eval.runner --threshold 0.3      # override retrieval threshold

Exit code 0 = all pass; 1 = failures detected.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml is required. Install with: pip install pyyaml")
    sys.exit(1)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripbox_kb.config import load_config
from scripbox_kb.kb_store import create_kb_store
from scripbox_kb.service import QueryService

FIXTURES_DIR = Path(__file__).parent / "fixtures"
RESULTS_DIR = Path(__file__).parent / "results"

# Evaluation gates
RECALL_GATE = 0.80       # ≥80% recall@5 required (use 95% when you have 50+ cases)
ABSTENTION_GATE = 1.0    # 100% abstention pass rate required


def load_fixtures(names: list[str] | None = None) -> list[dict[str, Any]]:
    """Load test cases from YAML fixture files."""
    all_cases: list[dict[str, Any]] = []
    files = list(FIXTURES_DIR.glob("*.yaml"))

    if names:
        files = [f for f in files if f.stem in names]

    if not files:
        print(f"No fixture files found in {FIXTURES_DIR}")
        return []

    for path in sorted(files):
        with open(path) as f:
            data = yaml.safe_load(f)
        cases = data.get("cases", [])
        for case in cases:
            case["_fixture_file"] = path.stem
        all_cases.extend(cases)
        print(f"Loaded {len(cases)} cases from {path.name}")

    return all_cases


def run_case(case: dict[str, Any], service: QueryService) -> dict[str, Any]:
    """Run a single eval case and return a result dict."""
    query = case["query"]
    should_abstain = case.get("should_abstain", False)
    expected_sources = case.get("expected_sources", [])

    t0 = time.perf_counter()
    response = service.handle_query(query)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Check abstention
    abstained = response.status == "abstained"
    abstention_ok = (should_abstain == abstained)

    # Check recall (only for non-abstain cases)
    recall_ok = True
    if not should_abstain and expected_sources:
        source_urls = [h.url for h in response.sources]
        recall_ok = any(
            any(exp in url for url in source_urls)
            for exp in expected_sources
        )

    passed = abstention_ok and recall_ok

    return {
        "id": case["id"],
        "fixture": case["_fixture_file"],
        "query": query,
        "should_abstain": should_abstain,
        "abstained": abstained,
        "abstention_ok": abstention_ok,
        "expected_sources": expected_sources,
        "retrieved_urls": [h.url for h in response.sources],
        "recall_ok": recall_ok,
        "rewrite_count": response.rewrite_count,
        "status": response.status,
        "elapsed_ms": round(elapsed_ms, 1),
        "passed": passed,
    }


def print_results(results: list[dict[str, Any]]) -> bool:
    """Print a summary table and per-case results."""
    total = len(results)
    if total == 0:
        return True

    passed = sum(1 for r in results if r["passed"])
    abstain_cases = [r for r in results if r["should_abstain"]]
    abstain_passed = sum(1 for r in abstain_cases if r["abstention_ok"])
    recall_cases = [r for r in results if not r["should_abstain"] and r["expected_sources"]]
    recall_passed = sum(1 for r in recall_cases if r["recall_ok"])

    print("\n" + "=" * 70)
    print("EVAL RESULTS")
    print("=" * 70)

    for r in results:
        status_tag = "[PASS]" if r["passed"] else "[FAIL]"
        rewrite_note = f" [rewrote x{r['rewrite_count']}]" if r["rewrite_count"] else ""
        print(f"{status_tag} [{r['id']}] {r['query'][:60]}{rewrite_note}")
        if not r["passed"]:
            if not r["abstention_ok"]:
                print(f"   abstention FAIL - expected={r['should_abstain']} got={r['abstained']}")
            if not r["recall_ok"]:
                print(f"   recall FAIL - expected={r['expected_sources']}")
                print(f"   retrieved={r['retrieved_urls'][:3]}")

    print("\n" + "-" * 70)
    print(f"Total:            {passed}/{total} passed ({passed/total*100:.0f}%)")
    if recall_cases:
        print(f"Recall@5:         {recall_passed}/{len(recall_cases)} ({recall_passed/len(recall_cases)*100:.0f}%)")
    if abstain_cases:
        print(f"Abstention rate:  {abstain_passed}/{len(abstain_cases)} ({abstain_passed/len(abstain_cases)*100:.0f}%)")
    print("-" * 70)

    # Gate checks
    gates_ok = True
    if recall_cases:
        recall_rate = recall_passed / len(recall_cases)
        gate_status = "[PASS]" if recall_rate >= RECALL_GATE else "[FAIL]"
        print(f"{gate_status} Recall@5 gate: {recall_rate*100:.0f}% (required >={RECALL_GATE*100:.0f}%)")
        if recall_rate < RECALL_GATE:
            gates_ok = False
    if abstain_cases:
        abstain_rate = abstain_passed / len(abstain_cases)
        gate_status = "[PASS]" if abstain_rate >= ABSTENTION_GATE else "[FAIL]"
        print(f"{gate_status} Abstention gate: {abstain_rate*100:.0f}% (required >={ABSTENTION_GATE*100:.0f}%)")
        if abstain_rate < ABSTENTION_GATE:
            gates_ok = False

    print("=" * 70)
    return gates_ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Scripbox KB eval suite")
    parser.add_argument("--fixtures", nargs="*", help="Fixture file names (without .yaml)")
    parser.add_argument("--threshold", type=float, default=None, help="Override retrieval threshold")
    parser.add_argument("--output", type=str, default=None, help="Save JSON results to file")
    parser.add_argument("--mock", action="store_true", help="Run with mock KB store and LLM for CI testing")
    args = parser.parse_args()

    cases = load_fixtures(args.fixtures)
    if not cases:
        print("No test cases to run.")
        return 0

    print("Loading configuration and KB store...")
    config = load_config()

    if args.threshold is not None:
        from dataclasses import replace
        config = replace(config, retrieval_threshold=args.threshold)

    patcher_stream = None
    patcher_call = None

    if args.mock:
        from dataclasses import replace
        from unittest.mock import patch
        from scripbox_kb.kb_store import KBStore, RetrievalHit
        from scripbox_kb.providers import ProviderResult

        config = replace(
            config,
            groq_api_key="mock-key",
            gemini_api_key="mock-key",
            vector_db_type="chroma",
        )

        class MockKB:
            def load(self):
                pass
            def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
                for c in cases:
                    if not c.get("should_abstain") and (c["query"].lower() in query.lower() or query.lower() in c["query"].lower()):
                        src = c.get("expected_sources", ["https://help.scripbox.com/solutions"])[0]
                        return [
                            RetrievalHit(
                                title="Mock Article",
                                url=src,
                                category="General",
                                folder="Folder",
                                document="Relevant knowledge base context for testing.",
                                score=0.95,
                            )
                        ]
                return []
            def health_check(self) -> bool:
                return True

        kb_store: Any = MockKB()

        patcher_stream = patch("scripbox_kb.providers.stream_with_fallback", return_value=ProviderResult(iter(["Mock answer"]), "mock"))
        patcher_call = patch("scripbox_kb.providers.call_once", return_value="UNRELATED")
        patcher_stream.start()
        patcher_call.start()
    else:
        kb_store = create_kb_store(config)
        try:
            kb_store.load()
        except Exception as e:
            print(f"ERROR: Failed to load KB store: {e}")
            print("Make sure the vector DB is built and configured correctly.")
            return 1

    try:
        service = QueryService(config, kb_store)
        print(f"\nRunning {len(cases)} eval cases...")
        results = []
        for i, case in enumerate(cases, 1):
            print(f"  [{i}/{len(cases)}] {case['id']}...", end=" ", flush=True)
            result = run_case(case, service)
            results.append(result)
            status = "PASS" if result["passed"] else "FAIL"
            print(f"{status} ({result['elapsed_ms']}ms)")

        gates_ok = print_results(results)

        # Save results
        RESULTS_DIR.mkdir(exist_ok=True)
        if args.output:
            output_path = Path(args.output)
        else:
            ts = time.strftime("%Y%m%dT%H%M%S")
            output_path = RESULTS_DIR / f"eval_{ts}.json"

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

        return 0 if gates_ok else 1
    finally:
        if patcher_stream:
            patcher_stream.stop()
        if patcher_call:
            patcher_call.stop()


if __name__ == "__main__":
    sys.exit(main())
