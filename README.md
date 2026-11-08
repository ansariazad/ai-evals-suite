# 🧪 AI Evals Suite — 3-Level LLM Evaluation Harness

A complete evaluation framework for LLM systems: unit assertions, LLM-as-judge calibration, and A/B system benchmarking.

## ✨ Features

- **Level 1 — Unit Assertions** — deterministic checks (causal masking, MCP zero-stdout, ACI guardrails, RAG faithfulness, MCP STDIO end-to-end)
- **Level 2 — LLM Judge Calibration** — measures TPR, TNR, and judge-human agreement
- **Level 3 — A/B Benchmarking** — CRAG+RRF (System B) vs baseline (System A) on retrieval recall
- **100% pass rate** across all 5 assertions
- **Structured traces** — results logged to `eval_traces.jsonl`

## 📊 Results

| Eval | Result |
|------|--------|
| `L1_GPT_Causal_Mask_No_Future_Leakage` | ✅ PASS — `0.00e+00 < 1e-6` |
| `L1_MCP_Server_Zero_Stdout_Print_Calls` | ✅ PASS |
| `L1_Agent_ACI_Relative_Path_Guardrail` | ✅ PASS |
| `L1_RAG_Canonical_Citation_And_Faithfulness` | ✅ PASS — `Faithfulness=0.974` |
| `L1_MCP_STDIO_End_To_End_Tool_Invocation` | ✅ PASS — all 3 tools |
| **Judge-Human Agreement** | **100%** |
| **TPR / TNR** | **100% / 100%** |
| **System B Recall (CRAG+RRF)** | **100% vs 25% (+75pp)** |

## 🚀 Quick Start

```bash
pip install pytest torch numpy httpx mcp

# Run all evals
python 05_ai_evals_suite.py

# Or via pytest
pytest 05_ai_evals_suite.py -v
```

## 🏗️ Eval Architecture

```
Level 1: Unit Assertions
  → Deterministic, fast, always pass/fail

Level 2: LLM Judge
  → Calibrate with human labels
  → Measure TPR, TNR, agreement rate

Level 3: A/B Benchmark  
  → Compare retrieval systems on golden QA set
  → Report recall, precision, improvement delta
```

## 📁 Files

| File | Description |
|------|-------------|
| `05_ai_evals_suite.py` | Main eval harness |
| `eval_traces.jsonl` | Structured output traces |

## 🔗 References

- [RAGAS](https://github.com/explodinggradients/ragas) — RAG evaluation framework
- [LLM-as-Judge](https://arxiv.org/abs/2306.05685) — Zheng et al.
- [HELMET Benchmark](https://github.com/princeton-nlp/HELMET)

## 📄 License

MIT
