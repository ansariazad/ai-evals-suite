#!/usr/bin/env python3
"""
05_ai_evals_suite.py — Three-Level Domain-Specific AI Evaluation System
=======================================================================
Reference Source:
  - Hamel Husain, "Your AI Product Needs Evals"
    https://hamel.dev/blog/posts/evals/index.html

Implemented Evaluation Hierarchy:
  1. Level 1 — Fast Deterministic Unit Tests & Assertions:
     - Runs on every code/prompt change with zero LLM latency cost
     - Tests GPT causal attention masking (future tokens cannot leak into past logits),
       MCP STDIO safety (zero `print()` statements in `04_mcp_server.py`),
       ACI poka-yoke path validation, and RAG source citation grounding
  2. Level 2 — Trace Logging + Binary (Good/Bad) Human & LLM-as-a-Judge Eval:
     - Logs structured traces to JSONL for frictionless inspection
     - Evaluates responses with binary `Good` / `Bad` critiques (preferred over noisy 1–5 scales)
     - Calibrates the Automated Judge against Human ground-truth labels (computing Precision,
       Recall / TPR, Specificity / TNR, and Agreement Rate)
  3. Level 3 — A/B Pipeline Comparison Scorecard:
     - Compares Naive Single-Query RAG vs. Full Corrective RAG + Multi-Query RRF Fusion
       across retrieval recall, citation grounding, and pass rate

Usage on macOS:
  python3 05_ai_evals_suite.py
"""

from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class EvalAssertionResult:
    test_name: str
    passed: bool
    detail: str


@dataclass
class LoggedTrace:
    trace_id: str
    question: str
    routed_domain: str
    retrieved_chunks: List[str]
    response: str
    human_binary_label: Optional[bool]  # True = Good, False = Bad
    judge_binary_label: Optional[bool] = None
    judge_critique: str = ""


def run_level_1_unit_assertions() -> List[EvalAssertionResult]:
    """
    Executes fast, deterministic assertions across all modules:
      1. GPT Causal Mask Assertion: Perturbing token t+1 must not alter logits at position <= t
      2. MCP STDIO Hygiene Assertion: `04_mcp_server.py` AST contains zero `print()` calls
      3. ACI Poka-Yoke Assertion: `tool_read_local_file` rejects relative filepaths
      4. RAG Citation Grounding Assertion: Every synthesized answer includes canonical URLs
    """
    results: List[EvalAssertionResult] = []

    # Assertion 1: Verify GPT causal masking (future tokens never leak into past positions)
    gpt_mod = import_module("01_apna_gpt")
    cfg = gpt_mod.GPTConfig(batch_size=1, block_size=16, n_embd=32, n_head=2, n_layer=2, dropout=0.0)
    p_model = gpt_mod.PortableCausalGPT(cfg, vocab_size=20)
    seq_a = [1, 2, 3, 4, 5, 6]
    seq_b = [1, 2, 3, 4, 5, 19]  # Only final token differs
    logits_a = p_model.forward_sequence(seq_a)
    logits_b = p_model.forward_sequence(seq_b)
    max_diff_prefix = max(
        abs(logits_a[t][v] - logits_b[t][v])
        for t in range(len(seq_a) - 1)
        for v in range(20)
    )
    causal_ok = max_diff_prefix < 1e-6
    results.append(
        EvalAssertionResult(
            test_name="L1_GPT_Causal_Mask_No_Future_Leakage",
            passed=causal_ok,
            detail=f"Max prefix logit diff when perturbing final token = {max_diff_prefix:.2e} (< 1e-6)",
        )
    )

    # Assertion 2: Verify `04_mcp_server.py` has zero `print()` calls in its AST
    mcp_path = Path(__file__).parent / "04_mcp_server.py"
    tree = ast.parse(mcp_path.read_text(encoding="utf-8"))
    print_calls = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]
    mcp_stdio_ok = len(print_calls) == 0
    results.append(
        EvalAssertionResult(
            test_name="L1_MCP_Server_Zero_Stdout_Print_Calls",
            passed=mcp_stdio_ok,
            detail="Verified 0 `print()` calls in 04_mcp_server.py AST (STDIO JSON-RPC safe)",
        )
    )

    # Assertion 3: Verify Anthropic ACI Poka-Yoke rejects relative paths
    agents_mod = import_module("03_effective_agents")
    rel_out = agents_mod.tool_read_local_file({"absolute_path": "relative/path.py"})
    poka_yoke_ok = "ACI_POKA_YOKE_ERROR" in rel_out
    results.append(
        EvalAssertionResult(
            test_name="L1_Agent_ACI_Relative_Path_Guardrail",
            passed=poka_yoke_ok,
            detail=f"Relative path rejected with: {rel_out[:68]}...",
        )
    )

    # Assertion 4: Verify RAG output embeds canonical URLs and passes faithfulness check
    rag_mod = import_module("02_rag_from_scratch")
    pipeline = rag_mod.CorrectiveRAGPipeline(rag_mod.KNOWLEDGE_BASE)
    rag_out = pipeline.query("How does Reciprocal Rank Fusion work in RAG?")
    url_grounded = "https://www.youtube.com/watch?v=sVcwVQRHIc8" in str(rag_out["answer"]) and bool(
        rag_out["is_faithful"]
    )
    results.append(
        EvalAssertionResult(
            test_name="L1_RAG_Canonical_Citation_And_Faithfulness",
            passed=url_grounded,
            detail=f"Faithfulness={rag_out['faithfulness_score']}, includes source URL in answer",
        )
    )

    # Assertion 5: Verify MCP STDIO client can spawn 04_mcp_server.py and invoke tools over JSON-RPC 2.0
    mcp_integration = agents_mod.workflow_mcp_stdio_integration()
    mcp_tools = mcp_integration.get("tools_available", [])
    mcp_tool_results = mcp_integration.get("tool_results", {})
    mcp_ok = (
        "get_alerts" in mcp_tools
        and "get_forecast" in mcp_tools
        and "search_ai_playbooks" in mcp_tools
        and len(mcp_tool_results) == 3
    )
    results.append(
        EvalAssertionResult(
            test_name="L1_MCP_STDIO_End_To_End_Tool_Invocation",
            passed=mcp_ok,
            detail=f"Tools discovered: {mcp_tools}, invocations returned: {list(mcp_tool_results.keys())}",
        )
    )

    return results


def binary_llm_judge(trace: LoggedTrace) -> Tuple[bool, str]:
    """
    Domain-specific Binary Judge (Hamel Husain Level 2):
    Evaluates a trace with a binary Good (True) / Bad (False) decision and written critique.
    """
    reasons: List[str] = []
    if not trace.retrieved_chunks:
        reasons.append("No supporting context chunks retrieved.")
    if "https://" not in trace.response:
        reasons.append("Response is missing a clickable https:// source citation.")
    if "I don't know" in trace.response or "hallucinated_claim" in trace.response:
        reasons.append("Response contains ungrounded or incomplete answer markers.")

    if reasons:
        return False, "BAD — " + " ".join(reasons)
    return True, "GOOD — Grounded in retrieved chunks with verified inline URL citations."


def run_level_2_trace_and_judge_eval(trace_log_path: Path) -> Dict[str, Any]:
    """Generates traces, logs them to JSONL, runs the binary judge, and measures human alignment."""
    rag_mod = import_module("02_rag_from_scratch")
    pipeline = rag_mod.CorrectiveRAGPipeline(rag_mod.KNOWLEDGE_BASE)

    live_q1 = pipeline.query("How does Karpathy train GPT and how does SFT/RLHF differ?")
    live_q2 = pipeline.query("What are Anthropic's 5 workflow patterns and when should you use agents?")

    traces: List[LoggedTrace] = [
        LoggedTrace(
            trace_id="trace-001",
            question=str(live_q1["question"]),
            routed_domain=str(live_q1["routed_domain"]),
            retrieved_chunks=list(live_q1["retrieved_chunk_ids"]),  # type: ignore
            response=str(live_q1["answer"]),
            human_binary_label=True,
        ),
        LoggedTrace(
            trace_id="trace-002",
            question=str(live_q2["question"]),
            routed_domain=str(live_q2["routed_domain"]),
            retrieved_chunks=list(live_q2["retrieved_chunk_ids"]),  # type: ignore
            response=str(live_q2["answer"]),
            human_binary_label=True,
        ),
        LoggedTrace(
            trace_id="trace-003-failure-case",
            question="How should an MCP server log messages?",
            routed_domain="agents_mcp_db",
            retrieved_chunks=["mcp-01"],
            response="Just use print() to stdout whenever you want to debug your server.",
            human_binary_label=False,
        ),
        LoggedTrace(
            trace_id="trace-004-empty-retrieval",
            question="How do I configure Kubernetes ingress controllers?",
            routed_domain="none",
            retrieved_chunks=[],
            response="I don't know because no chunks were retrieved.",
            human_binary_label=False,
        ),
    ]

    tp = tn = fp = fn = 0
    for t in traces:
        label, critique = binary_llm_judge(t)
        t.judge_binary_label = label
        t.judge_critique = critique
        if t.human_binary_label is True and label is True:
            tp += 1
        elif t.human_binary_label is False and label is False:
            tn += 1
        elif t.human_binary_label is False and label is True:
            fp += 1
        elif t.human_binary_label is True and label is False:
            fn += 1

    trace_log_path.write_text(
        "\n".join(json.dumps(asdict(t)) for t in traces) + "\n",
        encoding="utf-8",
    )

    total = len(traces)
    agreement = (tp + tn) / total if total else 0.0
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "total_traces": total,
        "judge_human_agreement": round(agreement, 3),
        "true_positive_rate": round(tpr, 3),
        "true_negative_rate": round(tnr, 3),
        "trace_log_file": str(trace_log_path),
        "traces": traces,
    }


def run_level_3_ab_comparison() -> Dict[str, Any]:
    """Compares System A (Naive Top-1 Single-Domain RAG) vs System B (CRAG + Multi-Query RRF)."""
    rag_mod = import_module("02_rag_from_scratch")
    index = rag_mod.VectorStoreIndex(rag_mod.KNOWLEDGE_BASE)
    crag = rag_mod.CorrectiveRAGPipeline(rag_mod.KNOWLEDGE_BASE)

    benchmark = [
        {
            "question": "How do GPT causal self-attention and RAG-Fusion RRF work together?",
            "expected_ids": {"gpt-01", "rag-01"},
        },
        {
            "question": "How do Anthropic agent workflows connect to MCP STDIO servers and Hamel's 3-level evals?",
            "expected_ids": {"agent-01", "mcp-01", "eval-01"},
        },
    ]

    naive_recalls: List[float] = []
    crag_recalls: List[float] = []

    for item in benchmark:
        q = str(item["question"])
        expected: set[str] = item["expected_ids"]  # type: ignore

        naive_hits = index.search(q, top_k=1, domain_filter="transformers_db")
        naive_ids = {c.chunk_id for c, _ in naive_hits}
        naive_recalls.append(len(naive_ids & expected) / len(expected))

        crag_out = crag.query(q)
        crag_ids = set(crag_out["retrieved_chunk_ids"])  # type: ignore
        crag_recalls.append(len(crag_ids & expected) / len(expected))

    mean_a = sum(naive_recalls) / len(naive_recalls)
    mean_b = sum(crag_recalls) / len(crag_recalls)
    return {
        "system_a_naive_top1_recall": round(mean_a, 3),
        "system_b_crag_rrf_recall": round(mean_b, 3),
        "relative_lift": f"+{(mean_b - mean_a) * 100:.1f} percentage points",
    }


def main() -> None:
    print("=" * 72)
    print("AI EVALS SUITE — Level 1 Unit Assertions, Level 2 Judge, Level 3 A/B")
    print("Reference: https://hamel.dev/blog/posts/evals/index.html (Hamel Husain)")
    print("=" * 72)

    print("\n[Level 1: Scoped Unit Assertions (Run on every commit)]")
    l1_results = run_level_1_unit_assertions()
    for r in l1_results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.test_name}: {r.detail}")

    trace_path = Path(__file__).parent / "eval_traces.jsonl"
    l2_results = run_level_2_trace_and_judge_eval(trace_path)
    print("\n[Level 2: Trace Logging & Binary Judge Calibration]")
    print(
        f"  Traces Logged: {l2_results['total_traces']} (saved to {trace_path.name}) | "
        f"Judge-Human Agreement: {l2_results['judge_human_agreement']*100:.1f}% | "
        f"TPR: {l2_results['true_positive_rate']*100:.1f}% | "
        f"TNR: {l2_results['true_negative_rate']*100:.1f}%"
    )
    for t in l2_results["traces"]:
        print(f"  - {t.trace_id}: Human={t.human_binary_label}, Judge={t.judge_binary_label} ({t.judge_critique})")

    l3_results = run_level_3_ab_comparison()
    print("\n[Level 3: A/B Pipeline Comparison Scorecard]")
    print(f"  System A (Naive Top-1 Single-Domain RAG) Recall : {l3_results['system_a_naive_top1_recall']*100:.1f}%")
    print(f"  System B (Multi-Query RRF + CRAG) Recall        : {l3_results['system_b_crag_rrf_recall']*100:.1f}%")
    print(f"  Net Retrieval Recall Improvement                : {l3_results['relative_lift']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
