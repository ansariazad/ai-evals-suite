# 🧪 AI Evals Suite — 3-Level LLM Evaluation Harness, 100% Pass Rate

> **"Evals are the most important thing you can do for your AI product."** — Hamel Husain

A complete, three-level LLM evaluation harness built from scratch following Hamel Husain's methodology: fast **deterministic unit assertions**, **LLM-as-judge calibration** with human labels, and **A/B pipeline benchmarking**. Tests every module in the AI Engineering Suite. 5/5 assertions passing. 100% judge-human agreement.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Assertions](https://img.shields.io/badge/Assertions-5%2F5_PASS-22c55e)](https://github.com/ansariazad/ai-evals-suite)
[![Judge](https://img.shields.io/badge/Judge_Agreement-100%25-22c55e)](https://github.com/ansariazad/ai-evals-suite)
[![Recall](https://img.shields.io/badge/CRAG+RRF_Recall-100%25-22c55e)](https://github.com/ansariazad/ai-evals-suite)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e)](LICENSE)
[![Stars](https://img.shields.io/github/stars/ansariazad/ai-evals-suite?style=social)](https://github.com/ansariazad/ai-evals-suite/stargazers)

---

## 📺 Demo

```
$ python3 05_ai_evals_suite.py

━━━ LEVEL 1: UNIT ASSERTIONS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[PASS] L1_GPT_Causal_Mask_No_Future_Leakage
       Leakage = 0.00e+00  <  1e-06  ✓

[PASS] L1_MCP_Server_Zero_Stdout_Print_Calls
       0 print() calls found on stdout  ✓

[PASS] L1_Agent_ACI_Relative_Path_Guardrail
       ValueError raised correctly on absolute path  ✓

[PASS] L1_RAG_Canonical_Citation_And_Faithfulness
       Faithfulness = 0.974  ≥  0.90  ✓

[PASS] L1_MCP_STDIO_End_To_End_Tool_Invocation
       get_alerts ✓  get_forecast ✓  search_ai_playbooks ✓

5 / 5 assertions passed  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━ LEVEL 2: LLM-AS-JUDGE CALIBRATION ━━━━━━━━━━━━━━━━━━━━━━━━━━━
Judge-Human Agreement : 100%
TPR (Sensitivity)     : 100%
TNR (Specificity)     : 100%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━ LEVEL 3: A/B PIPELINE BENCHMARK ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
System A — Naive single-query RAG   : Recall  25%  (1/4)
System B — CRAG + Multi-Query RRF   : Recall 100%  (4/4)  🏆

Improvement: +75 percentage points
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 📑 Table of Contents

- [Why Evals Matter](#-why-evals-matter)
- [The 3-Level Pyramid](#-the-3-level-pyramid)
- [Assertions Detail](#-assertions-detail)
- [Quick Start](#-quick-start)
- [How Judge Calibration Works](#-how-judge-calibration-works)
- [JSONL Trace Format](#-jsonl-trace-format)
- [Eval Results](#-eval-results)
- [Roadmap](#-roadmap)
- [References](#-references)

---

## 💡 Why Evals Matter

| Without Evals | With This Suite |
|--------------|-----------------|
| "It seems to work" | Quantified pass/fail per component |
| Regressions caught by users | Regressions caught in < 1 second |
| Judge bias unchecked | TPR/TNR calibrated against human labels |
| "System B is better" | +75pp recall improvement, measured |
| Traces lost on restart | Persistent JSONL for every run |

---

## 🏗️ The 3-Level Pyramid

```
┌──────────────────────────────────────────────────────────────────┐
│                     3-LEVEL EVAL PYRAMID                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│   Level 3: A/B Benchmarking             ← highest signal          │
│   ─────────────────────────                                        │
│   System A (Naive RAG) vs System B (CRAG+RRF)                     │
│   Metrics: Recall, Faithfulness, Pass Rate                         │
│   Cost: Medium (runs full pipelines)                               │
│                                                                    │
│   Level 2: LLM-as-Judge Calibration    ← calibrated first         │
│   ──────────────────────────────────                               │
│   Binary Good/Bad labels vs human ground truth                     │
│   Metrics: TPR, TNR, Agreement Rate                                │
│   Cost: Low (binary labels, fast)                                  │
│                                                                    │
│   Level 1: Unit Assertions             ← run on every change       │
│   ─────────────────────────                                        │
│   Deterministic checks, zero LLM cost                             │
│   Speed: < 1 second total                                          │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Assertions Detail

| ID | Level | Module | What It Checks | Pass Criterion |
|----|-------|--------|----------------|----------------|
| `L1_GPT_Causal_Mask_No_Future_Leakage` | 1 | apna-gpt | Perturb token `t+1`, measure logit delta at positions `≤ t` | Leakage `< 1e-6` |
| `L1_MCP_Server_Zero_Stdout_Print_Calls` | 1 | mcp-server | AST-parse server for `print()` calls | 0 print() found |
| `L1_Agent_ACI_Relative_Path_Guardrail` | 1 | effective-agents | Agent rejects absolute file paths | `ValueError` raised |
| `L1_RAG_Canonical_Citation_And_Faithfulness` | 1 | rag-from-scratch | Answer tokens grounded in source chunks | Faithfulness `≥ 0.9` |
| `L1_MCP_STDIO_End_To_End_Tool_Invocation` | 1 | mcp-server | All 3 tools respond over subprocess STDIO | 3/3 tools called |

---

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/ansariazad/ai-evals-suite.git
cd ai-evals-suite

# Place sibling repos in same parent directory:
# ~/Documents/github-repos/
#   ├── apna-gpt/
#   ├── rag-from-scratch/
#   ├── effective-agents/
#   ├── mcp-server/
#   └── ai-evals-suite/   ← here

# Run all 3 eval levels
python3 05_ai_evals_suite.py

# Or via pytest
pip install pytest
pytest 05_ai_evals_suite.py -v
```

---

## 🔎 How Judge Calibration Works

```python
# Step 1: Run the RAG pipeline on golden questions, log traces
traces = run_level_2_trace_logging(rag_pipeline)

# Each trace has two labels:
#   human_binary_label : True (Good) / False (Bad)  ← ground truth
#   judge_binary_label : True / False                ← automated judge

# Step 2: Compute calibration metrics
tp = sum(h and j     for h, j in zip(human, judge))
tn = sum(not h and not j for h, j in zip(human, judge))

tpr       = tp / sum(human)           # Recall / Sensitivity
tnr       = tn / sum(not h for h in human)  # Specificity
agreement = sum(h == j for h, j in zip(human, judge)) / len(traces)
```

> [!TIP]
> Binary Good/Bad labels are intentional. Hamel Husain's research shows 5-point scales introduce inter-rater noise. Binary labels are faster to collect, more consistent, and correlate better with real user satisfaction.

---

## 📄 JSONL Trace Format

Each eval run writes structured traces to `eval_traces.jsonl`:

```json
{
  "trace_id": "trace-001",
  "question": "How does CRAG improve retrieval quality?",
  "routed_domain": "rag_db",
  "retrieved_chunks": ["chunk-rag-01", "chunk-rag-02"],
  "response": "CRAG grades retrieved documents for relevance...",
  "human_binary_label": true,
  "judge_binary_label": true,
  "judge_critique": "Response is accurate and grounded in sources."
}
```

Inspect traces:
```bash
cat eval_traces.jsonl | python3 -m json.tool | head -40
```

---

## 📊 Eval Results

| Assertion | Result | Detail |
|-----------|--------|--------|
| `L1_GPT_Causal_Mask` | ✅ PASS | Leakage = `0.00e+00` |
| `L1_MCP_Zero_Stdout` | ✅ PASS | 0 print() calls |
| `L1_ACI_Path_Guard` | ✅ PASS | ValueError raised |
| `L1_RAG_Faithfulness` | ✅ PASS | 0.974 ≥ 0.90 |
| `L1_MCP_STDIO_E2E` | ✅ PASS | 3/3 tools |
| Judge-Human Agreement | 100% | — |
| TPR / TNR | 100% / 100% | — |
| System B vs A Recall | +75pp | 100% vs 25% |

---

## 🗺️ Roadmap

- [ ] GitHub Actions CI — run L1 assertions on every PR
- [ ] LLM latency & token cost tracking per module
- [ ] Multi-model judge ensemble (GPT-4o + Claude + local)
- [ ] HTML eval report generator
- [ ] Pytest plugin for assertion helpers

---

## 📚 References

| Resource | Authors |
|----------|---------|
| [Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/index.html) | Hamel Husain |
| [LLM-as-a-Judge](https://arxiv.org/abs/2306.05685) | Zheng et al., 2023 |
| [RAGAS: Automated Evaluation of RAG](https://arxiv.org/abs/2309.15217) | Es et al., 2023 |
| [HELMET Benchmark](https://github.com/princeton-nlp/HELMET) | Princeton NLP |
| [Evals for AI agents](https://www.anthropic.com/research/evaluating-ai-systems) | Anthropic |

---

## 🤝 Contributing

PRs welcome! Suggested additions: new Level 1 assertions, CI integration, HTML report output.

---

## ⭐ Support

If this helped you understand how to test AI systems, **star the repo**!

[![GitHub stars](https://img.shields.io/github/stars/ansariazad/ai-evals-suite?style=for-the-badge)](https://github.com/ansariazad/ai-evals-suite/stargazers)

---

## 📄 License

MIT © 2026

---

## 👤 Author

**Azad Ansari** — Python Developer & AI Engineer, Mumbai

[![GitHub](https://img.shields.io/badge/GitHub-ansariazad-181717?logo=github)](https://github.com/ansariazad)
[![Portfolio](https://img.shields.io/badge/Portfolio-ansariazad.github.io-blue)](https://ansariazad.github.io)
