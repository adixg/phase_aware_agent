# Frozen XGBoost final-refit diagnostics

## Three findings

1. **Turn ends remained difficult.** The final refit recalled 8.6% of actual turn ends, compared with 8.7% in the previous diagnostics. The largest confusions by count were Turn ends → Other (569), Turn ends → Interpreter (560), Read → Edit (428). The highest class recalls were Interpreter (65.1%), Network (64.3%), Browser (64.3%).
2. **Top-1 accuracy varied more across harnesses than across steps.** It ranged from 45.5% to 46.0% across steps. Among harnesses with at least 500 requests, it ranged from 37.9% for zcode to 55.7% for copilot.
3. **The previous tool category still dominated grouped split gain.** The leading features were Previous tool category (51.6%), Second previous tool category (13.8%), Harness (9.7%).

## Consistency check and scope

The in-memory refit reproduced **8.23% relative Brier improvement** (paired session-bootstrap 95% CI 7.67%–8.80%). Counts, the pass decision, per-step and macro metrics for both the model and the baseline, and the bootstrap interval match the [original final evaluation](../results/final_evaluation.json) with maximum absolute numerical difference **1.67e-16**.

The refit used 704,880 event-A requests from May 17–Aug 15, 2026, and the shared historical previous-tool baseline with smoothing strength 80. The [frozen configuration](../results/frozen.json) specifies the original encoder and class order, XGBoost 3.1.3, CPU histogram trees, 100 rounds, depth 6, learning rate 0.3, seed 42, and four threads. The reconstructed booster was kept in memory. Its UBJ serialization hashes to `cb0cd3399f4b31108a7bac520a88444dc86fb0187345dd83774f203fd72fc2fa`.

The [analysis script](analyze.py) scores 24,864 event-A requests at steps 3, 5, 10, and 20 on **previously exposed Aug 23–Sep 1 data**. These diagnostics are descriptive and made no model-selection decisions. The date-bounded loader excludes Sep 2–5. The original freeze and evaluation files were left intact.

## Differences from the previous diagnostics

The previous analysis loaded `tuned_model.ubj`, fitted through Aug 8. Across the same evaluation window and step counts, pooled Brier changed from 0.71426 to 0.70861, Top-1 from 45.29% to 45.77%, and Top-3 from 71.76% to 72.46%. The largest absolute recall shifts were Ask User (+2.9 pp, n=139), Edit (+2.7 pp, n=2,499), Browser (+2.5 pp, n=277). The table below records the shift for every class.

The [importance plot](feature_importance.png) uses XGBoost total split gain aggregated across one-hot columns, including missing and unknown columns, for each original feature. Previous-tool gain changed from 51.8% to 51.6%, second-previous-tool gain from 12.9% to 13.8%, and harness gain from 10.5% to 9.7%. Gain measures training-objective reduction attributed to tree splits.

## Class errors

The [confusion matrix](confusion_matrix.png) normalizes rows by actual class. Per-class Top-1 accuracy is recall. Precision is conditioned on the predicted class.

| Actual class | Requests | Recall | Precision | Recall change (pp) | Most frequent wrong prediction |
|---|---:|---:|---:|---:|---|
| Other | 3,662 | 56.6% | 46.9% | +0.7 | Interpreter (344, 9.4%) |
| Search | 2,808 | 45.0% | 44.8% | +1.2 | Read (419, 14.9%) |
| Interpreter | 3,726 | 65.1% | 50.6% | -0.3 | Edit (284, 7.6%) |
| Read | 2,902 | 55.1% | 47.2% | -0.4 | Edit (428, 14.7%) |
| Edit | 2,499 | 54.7% | 46.5% | +2.7 | Read (353, 14.1%) |
| File Ops | 2,009 | 28.4% | 35.6% | -0.5 | Other (336, 16.7%) |
| Write | 1,062 | 18.7% | 34.7% | +1.2 | Interpreter (263, 24.8%) |
| Network | 1,025 | 64.3% | 54.1% | -0.3 | Other (83, 8.1%) |
| Wait/Poll | 787 | 23.6% | 32.6% | +1.7 | Other (215, 27.3%) |
| Git/VCS | 758 | 52.6% | 47.7% | -0.7 | Other (75, 9.9%) |
| Build/Test | 372 | 18.5% | 38.1% | -1.1 | Edit (85, 22.8%) |
| Browser | 277 | 64.3% | 50.4% | +2.5 | Other (29, 10.5%) |
| Packages | 169 | 18.3% | 22.3% | +1.2 | Edit (30, 17.8%) |
| MCP | 264 | 53.4% | 49.0% | +0.0 | Other (32, 12.1%) |
| Ask User | 139 | 7.9% | 40.7% | +2.9 | Other (29, 20.9%) |
| Subagent | 92 | 8.7% | 25.0% | +1.1 | Other (22, 23.9%) |
| Turn ends | 2,313 | 8.6% | 29.4% | -0.0 | Other (569, 24.6%) |

## Request step

The evaluation averages per-step relative Brier reductions with equal step weight. Top-1 and Top-3 follow the repository’s strict-greater rank rule. Log loss floors actual-class probability at 1e-6.

| Step | Requests | Refit Brier | Baseline Brier | Brier reduction | Top-1 | Top-3 | Log loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3 | 9,636 | 0.7125 | 0.7619 | 6.49% | 45.71% | 72.38% | 1.7708 |
| 5 | 7,571 | 0.7067 | 0.7643 | 7.53% | 45.98% | 73.00% | 1.7515 |
| 10 | 4,903 | 0.7057 | 0.7768 | 9.16% | 45.46% | 72.10% | 1.7648 |
| 20 | 2,754 | 0.7056 | 0.7817 | 9.73% | 45.97% | 71.86% | 1.7566 |

## Harness

Harness scores pool all four steps and reflect each harness’s step and target mix. The smallest groups have limited support.

| Harness | Requests | Brier | Top-1 | Top-3 | Log loss |
|---|---:|---:|---:|---:|---:|
| All requests | 24,864 | 0.7086 | 45.8% | 72.5% | 1.7622 |
| hermes | 6,835 | 0.6967 | 46.8% | 73.0% | 1.7360 |
| opencode | 6,794 | 0.7230 | 44.6% | 71.0% | 1.7892 |
| pi | 5,748 | 0.7112 | 45.2% | 72.1% | 1.7896 |
| codex | 997 | 0.7079 | 45.0% | 77.1% | 1.6394 |
| copilot | 936 | 0.6175 | 55.7% | 80.4% | 1.4923 |
| kilo | 880 | 0.6837 | 49.7% | 75.9% | 1.6790 |
| oh-my-pi | 803 | 0.6918 | 47.7% | 73.5% | 1.7125 |
| claude-code | 799 | 0.7404 | 42.6% | 67.3% | 1.9132 |
| zcode | 633 | 0.7810 | 37.9% | 66.2% | 1.9692 |
| cline | 297 | 0.7505 | 41.4% | 73.1% | 1.8217 |
| openclaw | 107 | 0.7396 | 39.3% | 77.6% | 1.8546 |
| eval:swe-bench | 27 | 0.8815 | 37.0% | 48.1% | 3.5269 |
| zoo | 8 | 0.5862 | 50.0% | 62.5% | 1.6485 |
