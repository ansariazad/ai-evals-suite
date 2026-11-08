# AI Evals Suite

**ai-evals-suite** is a three-level LLM evaluation harness built from scratch following Hamel Husain's ["Your AI Product Needs Evals"](https://hamel.dev/blog/posts/evals/index.html) methodology — fast deterministic unit assertions, LLM-as-judge calibration with human labels, and A/B pipeline benchmarking. 100% pass rate across all 5 assertions.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![Evals](https://img.shields.io/badge/Assertions-5%2F5_PASS-brightgreen)
![Judge](https://img.shields.io/badge/Judge_Agreement-100%25-brightgreen)
![Recall](https://img.shields.io/badge/CRAG_Recall-100%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

---

## What is this?

A complete evaluation framework that tests every component of the AI Engineering suite (`apna-gpt`, `rag-from-scratch`, `effective-agents`, `mcp-server`) across three escalating levels of rigor — from microsecond unit checks to multi-system A/B benchmarks. All results are logged to `eval_traces.jsonl` for inspection.

Follows Hamel Husain's key principles:
- **Binary labels only** — Good/Bad (not 1-5 scales that add noise)
- **Fast assertions first** — catch regressions instantly before expensive LLM calls
- **Judge calibration** — measure TPR, TNR, agreement rate before trusting your judge
- **Structured traces** — every eval run writes JSONL for easy debugging

---

## Evaluation Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                    3-LEVEL EVAL PYRAMID                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Level 3: A/B Benchmarking          ← slowest, highest signal   │
│  ─────────────────────────                                       │
│  System A (Naive RAG) vs System B (CRAG+RRF)                    │
│  Metrics: Recall, Precision, Faithfulness, Pass Rate            │
│                                                                  │
│  Level 2: LLM-as-Judge              ← calibrated against humans │
│  ─────────────────────────                                       │
│  Binary Good/Bad labels on traced responses                      │
│  Metrics: TPR, TNR, Agreement Rate, Precision, Recall           │
│                                                                  │
│  Level 1: Unit Assertions           ← fastest, run on every PR  │
│  ─────────────────────────                                       │
│  Deterministic checks: causal mask, STDIO safety, ACI, RAG      │
│  Zero LLM latency cost                                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Features

- **L1: GPT Causal Mask assertion** — perturbs token `t+1` and verifies it doesn't alter logits at positions `≤ t` (leakage must be `< 1e-6`)
- **L1: MCP STDIO safety check** — AST-parses `04_mcp_server.py` and asserts zero `print()` calls on stdout (would corrupt JSON-RPC stream)
- **L1: ACI path guardrail** — verifies the agent rejects absolute file paths and enforces relative-path-only access
- **L1: RAG faithfulness assertion** — checks that the final answer contains citations grounded in source chunks (threshold: `> 0.9`)
- **L1: MCP STDIO end-to-end** — spawns `04_mcp_server.py` subprocess, calls all 3 tools over JSON-RPC 2.0, verifies responses
- **L2: Judge calibration** — compares automated judge labels against human ground-truth; computes TPR, TNR, agreement
- **L3: A/B benchmark** — runs naive single-query RAG vs. CRAG+RRF across a golden QA set, reports recall improvement
- **JSONL trace logging** — every eval writes structured `LoggedTrace` objects to `eval_traces.jsonl`

---

## Assertions Detail

| ID | Level | Module Tested | What It Checks | Pass Criterion |
|----|-------|--------------|----------------|----------------|
| `L1_GPT_Causal_Mask_No_Future_Leakage` | 1 | apna-gpt | Future token doesn't leak into past logits | Leakage `< 1e-6` |
| `L1_MCP_Server_Zero_Stdout_Print_Calls` | 1 | mcp-server | No `print()` on stdout in server code | 0 print calls found |
| `L1_Agent_ACI_Relative_Path_Guardrail` | 1 | effective-agents | Agent rejects absolute paths | Raises `ValueError` |
| `L1_RAG_Canonical_Citation_And_Faithfulness` | 1 | rag-from-scratch | Answer grounded in source chunks | Faithfulness `≥ 0.9` |
| `L1_MCP_STDIO_End_To_End_Tool_Invocation` | 1 | mcp-server | All 3 tools respond over subprocess STDIO | All 3 tools called |

---

## Eval Results (Last Run)

```
[PASS] L1_GPT_Causal_Mask_No_Future_Leakage      leakage=0.00e+00 < 1e-6
[PASS] L1_MCP_Server_Zero_Stdout_Print_Calls      0 print() calls on stdout
[PASS] L1_Agent_ACI_Relative_Path_Guardrail       ValueError raised correctly
[PASS] L1_RAG_Canonical_Citation_And_Faithfulness Faithfulness=0.974
[PASS] L1_MCP_STDIO_End_To_End_Tool_Invocation    all 3 tools discovered & called

Judge-Human Agreement : 100%
TPR (Sensitivity)     : 100%
TNR (Specificity)     : 100%

System A (Naive RAG)   Recall: 1/4 = 25%
System B (CRAG+RRF)    Recall: 4/4 = 100%  ← +75pp improvement
```

---

## Prerequisites

**Python 3.11+** required. The eval suite imports the sibling modules directly, so all repos must be in the same directory.

```bash
# Place all repos in the same parent folder:
~/Documents/github-repos/
  ├── apna-gpt/
  ├── rag-from-scratch/
  ├── effective-agents/
  ├── mcp-server/
  └── ai-evals-suite/   ← run from here
```

### macOS
```bash
brew install python@3.11
pip install torch numpy httpx
```

### Windows / Linux
```bash
pip install torch numpy httpx
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/ansariazad/ai-evals-suite.git
cd ai-evals-suite

# 2. Run all 3 levels
python3 05_ai_evals_suite.py

# 3. Inspect structured traces
cat eval_traces.jsonl | python3 -m json.tool | head -60
```

---

## Usage

### Run full eval suite
```bash
python3 05_ai_evals_suite.py
```

### Run via pytest
```bash
pytest 05_ai_evals_suite.py -v
```

### Inspect trace output
```bash
# Pretty-print the JSONL traces
cat eval_traces.jsonl | python3 -c "
import json, sys
for line in sys.stdin:
    t = json.loads(line)
    print(f\"[{t['trace_id']}] Q: {t['question'][:60]}\")
    print(f\"  domain={t['routed_domain']} | human={t['human_binary_label']} | judge={t['judge_binary_label']}\")
    print()
"
```

---

## How the Judge Calibration Works

```python
# Level 2 flow:
traces = run_level_2_trace_logging(rag_pipeline)

# Each trace has:
# - human_binary_label: True (Good) / False (Bad)   ← ground truth
# - judge_binary_label: True / False                 ← automated judge

# Compute calibration metrics:
tp = sum(h and j for h, j in zip(human_labels, judge_labels))
tn = sum(not h and not j for ...)
tpr = tp / sum(human_labels)          # Recall / Sensitivity
tnr = tn / sum(not h for h in human_labels)  # Specificity
agreement = sum(h == j for ...) / len(traces)
```

> [!TIP]
> Binary Good/Bad labels are intentionally used instead of 1-5 scores. Hamel Husain's research shows 5-point scales introduce inter-rater noise and make calibration unreliable. Binary labels are easier to collect, more consistent, and correlate better with real user satisfaction.

---

## JSONL Trace Format

Each line in `eval_traces.jsonl`:
```json
{
  "trace_id": "trace-001",
  "question": "How does CRAG improve retrieval?",
  "routed_domain": "rag_db",
  "retrieved_chunks": ["chunk-rag-01", "chunk-rag-02"],
  "response": "CRAG grades retrieved documents...",
  "human_binary_label": true,
  "judge_binary_label": true,
  "judge_critique": "Response is accurate and well-grounded in sources."
}
```

---

## File Structure

```
ai-evals-suite/
├── 05_ai_evals_suite.py  # Full 3-level eval harness
├── eval_traces.jsonl     # Structured trace output from last run
├── setup_and_run_mac.sh  # macOS setup script
├── requirements.txt      # Python dependencies
├── .env.example          # Environment template
├── .gitignore
└── README.md
```

---

## References

- [Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/index.html) — Hamel Husain
- [LLM-as-a-Judge](https://arxiv.org/abs/2306.05685) — Zheng et al., 2023
- [RAGAS: Automated Evaluation of RAG Pipelines](https://arxiv.org/abs/2309.15217) — Es et al., 2023
- [HELMET: How to Evaluate Long-Context LMs](https://github.com/princeton-nlp/HELMET) — Princeton NLP
- [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300) — Hendrycks et al.

---

## Contributing

Pull requests welcome! Suggested additions: LLM latency benchmarking, token cost tracking, multi-model judge ensemble, CI integration via GitHub Actions.

---

## License

MIT © 2026

---

## Author

**Azad Ansari** — [GitHub](https://github.com/ansariazad) | [AutoShorts](https://github.com/ansariazad/AutoShorts)
