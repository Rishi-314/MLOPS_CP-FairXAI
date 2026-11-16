# Demo plan – FairXAI MLOps (industry expert review)

Target length: **20 minutes + 10 minutes Q&A**. Every number below was measured on the live stack on
2026-11-16 (rehearsal run); earlier figures come from `docs/evidence/`.

**One-line pitch:** an end-to-end MLOps platform for a credit-default model where **no model can be
promoted unless it passes a hard fairness gate**, every prediction is explainable (SHAP), and live
traffic is monitored for drift, performance and fairness.

---

## 0. Before the demo (T-30 min)

Git Bash, from the repo root:

```bash
source .venv/Scripts/activate
set -a; source .env; set +a
export PYTHONIOENCODING=utf-8 MLFLOW_DISABLE_AGENT_HINT=1
# 1. Start Docker Desktop first, then:
C="docker compose --env-file .env -f infra/compose/docker-compose.yml --profile core"
$C up -d --wait postgres minio && $C run --rm init && $C up -d --wait mlflow serving-blue serving-green gateway prometheus grafana
$C ps                                    # 8 services Up / healthy
# 2. Slots start empty after a restart: load the model into both
curl -s -X POST http://127.0.0.1:8001/admin/load -H "X-API-Key: $ADMIN_API_KEY" -H "Content-Type: application/json" -d '{"model_version": 1}'
curl -s -X POST http://127.0.0.1:8002/admin/load -H "X-API-Key: $ADMIN_API_KEY" -H "Content-Type: application/json" -d '{"model_version": 1}'
curl -s http://127.0.0.1:8080/readyz     # {"ready":true,"active":"blue","model_version":1}
```

Open these browser tabs in order (Grafana login `admin` / `GRAFANA_ADMIN_PASSWORD` from `.env`):

1. GitHub repo / README (Status table)
2. MLflow – http://127.0.0.1:5000 → experiment `fairxai` → run `e29c476a…`
3. MLflow – Models → `fairxai-credit-default` v1 (tags + `candidate` alias)
4. Grafana – http://127.0.0.1:3000 → folder FairXAI → Service, Model, Fairness dashboards
5. Prometheus – http://127.0.0.1:9090/alerts
6. MinIO – http://127.0.0.1:9001 (buckets `mlflow`, `dvc`)

Prepare the request body in the terminal (so it doesn't need typing live):

```bash
BODY='{"features": {"LIMIT_BAL": 20000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24, "PAY_0": 2, "PAY_2": 2, "PAY_3": -1, "PAY_4": -1, "PAY_5": -2, "PAY_6": -2, "BILL_AMT1": 3913, "BILL_AMT2": 3102, "BILL_AMT3": 689, "BILL_AMT4": 0, "BILL_AMT5": 0, "BILL_AMT6": 0, "PAY_AMT1": 0, "PAY_AMT2": 689, "PAY_AMT3": 0, "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0}}'
```

**Backup plan:** if Docker or the network fails, present from `docs/evidence/phase_*.md` and the
screenshots you take during rehearsal (take screenshots of every tab above tonight).

---

## 1. Problem and architecture (2 min) – README

- Dataset: UCI *Default of Credit Card Clients* (Taiwan), **30,000 rows, 22.12 % default rate**.
- Why it matters: credit decisions are regulated; protected attributes **SEX** and **AGE** must not drive
  unfair outcomes, and every decision must be explainable.
- Show the architecture diagram in the README: DVC → training + Optuna → **fairness gate** → MLflow
  registry → blue/green FastAPI slots behind a gateway → Postgres prediction log → monitoring →
  Prometheus/Grafana.
- 8 containers: postgres, minio, mlflow, serving-blue, serving-green, gateway, prometheus, grafana.

## 2. Data versioning and validation (2 min)

```bash
dvc status          # Data and pipelines are up to date
cat dvc.yaml        # prepare -> split stages
```

| Item | Value |
| --- | --- |
| Splits (seeded, deterministic) | train 18,000 / eval 4,500 / test 3,000 / stream 9 × 500 |
| Validation | Great Expectations suites (gate G1 = training frame passes suite `batch`) |
| Lineage | `data_version_id` (SHA-256) + `dvc_train_md5` tagged on every model version |

Talking point: the **test split is never used by the gate**; it is a report only (no leakage into selection).

## 3. Training, tuning and experiment tracking (3 min) – MLflow tab

**7 baselines** (5-fold CV PR-AUC): random_forest 0.535, xgboost 0.512, svm 0.505, logreg 0.501,
knn 0.419, naive_bayes 0.406, decision_tree 0.284.

**Top 3 tuned** with Optuna TPE (seed 42, 25 trials each, 3-fold CV), shown as 72 nested runs in MLflow:

| Model | Tuned CV PR-AUC | Eval PR-AUC | Eval F1 | Threshold |
| --- | --- | --- | --- | --- |
| **xgboost (selected)** | **0.5585** | **0.5501** | 0.5393 | 0.53 |
| random_forest | 0.5537 | 0.5456 | 0.5421 | 0.53 |
| svm | 0.5148 | 0.4989 | 0.5205 | 0.33 |

Selected XGBoost params: n_estimators 398, max_depth 5, learning_rate 0.0104, subsample 0.826,
colsample_bytree 0.654, min_child_weight 2.

**Selected model results**

| Metric | Eval (4,500) | Test (3,000, held out) |
| --- | --- | --- |
| PR-AUC | 0.550 | 0.570 |
| F1 | 0.539 | 0.563 |
| Precision | 0.493 | 0.514 |
| Recall | 0.595 | 0.623 |
| Accuracy | 0.775 | 0.786 |

Talking points:
- **PR-AUC, not accuracy**, because the classes are imbalanced (a model that always predicts "no default" gets 78 % accuracy).
- The threshold (0.53) is chosen from out-of-fold F1, not tuned on eval/test.
- Reproducibility: CV scores were identical across three independent real runs (xgboost 0.5585 each time).

## 4. The fairness gate – the core differentiator (3 min) – MLflow artifacts `gate_report.json`, `fairness_report.json`

Promotion gates G1–G5; all must pass, and **REJECTED / INDETERMINATE models are never promoted**:

| Gate | What it checks | Result |
| --- | --- | --- |
| G1 | Data validation on the training frame | PASS |
| G2 | Eval PR-AUC ≥ floor 0.45 (and vs champion when one exists) | PASS (0.550) |
| G3 | Six hard fairness checks | PASS |
| G4 | Advisory warnings (demographic parity, calibration) | PASS, none |
| G5 | Save/load round-trip, identical probabilities from the registry URI | PASS |

G3 on eval (favorable outcome = predicted *not* to default):

| Check | Eval | Test | Threshold |
| --- | --- | --- | --- |
| Equal-opportunity diff – sex | 0.027 | 0.042 | ≤ 0.10 |
| Disparate impact – sex | 0.961 | 0.922 | ≥ 0.80 |
| EO diff – age band | 0.048 | 0.048 | ≤ 0.10 |
| DI – age band | 0.924 | 0.920 | ≥ 0.80 |
| EO diff – sex × age (intersectional) | 0.075 | 0.075 | ≤ 0.15 |
| DI – sex × age | 0.884 | 0.853 | ≥ 0.80 |

Talking points:
- 0.80 is the "four-fifths rule"; it is an engineering tolerance, **not proof of fairness or legal compliance**.
- A model with higher PR-AUC that fails fairness is **REJECTED** (covered by an 81-case truth-table test).
- Model version is registered with alias `candidate` and tags: `gate_decision`, `decision_threshold`,
  `eval_pr_auc`, `data_version_id`, `git_commit`, `run_id`.

## 5. Live serving and explainability (3 min) – terminal

```bash
curl -s -X POST http://127.0.0.1:8080/v1/predict -H "X-API-Key: $SIM_API_KEY" -H "Content-Type: application/json" -d "$BODY"
curl -s -X POST "http://127.0.0.1:8080/v1/explain?top_k=5" -H "X-API-Key: $SIM_API_KEY" -H "Content-Type: application/json" -d "$BODY"
curl -s http://127.0.0.1:8080/v1/model -H "X-API-Key: $SIM_API_KEY"
```

Expected: `prediction: 1`, `probability_default: 0.841`, threshold 0.53. SHAP (tree_shap, log-odds):
**PAY_0 +1.12**, PAY_2 +0.39, LIMIT_BAL +0.17, PAY_3 −0.13, PAY_AMT1 +0.09. Narrate it: *"this
applicant is 2 months late on the latest payment; that alone pushes the risk up the most."*

Global SHAP (mean |SHAP|): PAY_0 0.406, LIMIT_BAL 0.154, PAY_2 0.144, BILL_AMT1 0.087, PAY_AMT2 0.086.

Show that the API rejects bad input safely (measured today):

| Request | Response |
| --- | --- |
| Wrong API key | 401 |
| `AGE: 7` | 422 `validation_error` |
| Unknown field | 422 |
| Client key on an admin route | 403 |
| Rejected requests written to the prediction log | 0 |

```bash
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://127.0.0.1:8080/v1/predict -H "X-API-Key: wrong" -H "Content-Type: application/json" -d "$BODY"
```

**Latency at 5 requests/s through the gateway** (150 requests each, 0 errors; target p95 ≤ 200 ms):

| Endpoint | p50 | p95 | p99 |
| --- | --- | --- | --- |
| /v1/predict | 22.2 ms | **28.3 ms** | 35.0 ms |
| /v1/explain | 32.5 ms | **43.2 ms** | 49.7 ms |

## 6. Blue-green deployment, zero downtime (2 min) – terminal

```bash
curl -s http://127.0.0.1:8080/admin/state -H "X-API-Key: $ADMIN_API_KEY"
curl -s -X POST http://127.0.0.1:8080/admin/active -H "X-API-Key: $ADMIN_API_KEY" -H "Content-Type: application/json" -d '{"slot":"green"}'
curl -s http://127.0.0.1:8080/v1/model -H "X-API-Key: $SIM_API_KEY"     # "slot":"green"
curl -s -X POST http://127.0.0.1:8080/admin/active -H "X-API-Key: $ADMIN_API_KEY" -H "Content-Type: application/json" -d '{"slot":"blue"}'
```

Measured today: loading v1 into green took **3.0 s**; flipping to green and back **under 10 requests/s of
live traffic: 100/100 requests OK, 0 errors, p95 33 ms**. Safety rules: a flip to a slot with no model
returns **409** and traffic stays put; a failed `/admin/load` keeps the previous model serving; the
active slot is stored in Postgres, so it survives a gateway restart.

## 7. Monitoring: drift, performance, live fairness (4 min) – Grafana

Stream batches are replayed through the gateway, true labels arrive one batch later via `/v1/feedback`,
and the monitoring job evaluates the last 500 predictions (Evidently: KS for numeric, chi-square for
categorical, α = 0.05).

| Window | Preds OK | Feature drift (share) | Pred. drift | PR-AUC (ref 0.550, ALERT ≤ 0.500) | Live fairness | Retrain? |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 500/500 | OK 0.00 | OK | NA (no labels yet) | NA | no |
| 2 | 500/500 | OK 0.00 | OK | OK 0.575 | **ALERT** (eo_age 0.156) | no |
| 3 | 500/500 | **WARN 0.087** (BILL_AMT1, BILL_AMT2) | OK | OK 0.515 | WARN (cal_age 0.055) | no |
| 4 | 500/500 | OK 0.00 | OK | OK 0.508 | WARN (cal_age 0.103) | no |
| 5 | 500/500 | OK 0.00 | OK | OK 0.581 | WARN (cal_age 0.077) | no |

Totals: 2,502 predictions logged, 2,000 labels, 0 failed requests; each monitoring run took 2.5–4.8 s.

Talking points (these show maturity, so tell them up front instead of waiting to be asked):
- **The window-2 fairness ALERT is real and was not tuned away.** Each age band has only about 170 rows per
  500-row window, so TPR estimates are noisy. Windows 3–5 show eo_age 0.04, 0.04, 0.02 (all PASS).
  The fix on the backlog is bootstrap confidence intervals (X-14).
- **Fairness alerts notify; they never trigger retraining on their own** (by design: retraining on noisy
  fairness could make things worse).
- Retraining triggers need **persistence**: 3 consecutive drift-ALERT windows or 2 performance-ALERT
  windows. One drifting window (window 3, 2/23 features, well below the 0.30 ALERT share) correctly did nothing.
- INDETERMINATE (not PASS) is reported when an intersectional cell is too small to measure.

Grafana: **Service** (request rate, p95, 5xx), **Model** (prediction rate, drift, PR-AUC over windows),
**Fairness** (per-check values over time), **Lifecycle** (deployment and retraining history; it has no data in this setup, so skip it).
Prometheus: 3/3 targets up; 5 alert rules (High5xxRate, HighPredictLatencyP95, SlotDown,
ActiveSlotWithoutModel, PredictionLogFailures), all inactive.

## 8. Engineering quality (1 min)

- **258 tests pass** (`pytest tests`, 3 min 48 s, real Postgres; test fixtures are synthetic and never touch real data).
- All thresholds live in validated YAML under `configs/`; a test fails if any threshold is hard-coded.
- CI on every push (ruff lint and format, tree check, tests).
- Decision log and per-phase evidence written from real command output, **including incidents** (e.g.
  an MLflow host-header 403 that was surfaced as a misleading 404, now fixed and covered by a regression test).

## 9. Status and caveats (1 min)

All phases are complete (README Status table). Known caveats to own before they are pointed out:
- Model v1 was trained from uncommitted code (`git_dirty=true`). It is fine for development, but a
  release needs a clean retrain; the release policy already blocks dirty models.
- The serving images are 2.4 GB (SHAP and the scientific stack); not optimised yet.
- PR-AUC 0.55 is typical for this dataset; the gain over the 0.22 base rate is about 2.5×.

---

## Likely expert questions – short answers

| Question | Answer |
| --- | --- |
| Why keep SEX/AGE as features? | The current model **includes** them (`include_protected: true` in `configs/training.yaml`; a tested switch can drop them). Dropping them doesn't remove bias that comes through proxies (e.g. LIMIT_BAL), so we judge the model on **outcome** fairness with the hard gate. Be ready for the follow-up: "did you train with it off and compare?" (not yet done). |
| Why equal opportunity + disparate impact? | EO compares true-positive rates of the favorable outcome (qualified people treated equally); DI compares approval rates (four-fifths rule). Calibration and demographic parity are advisory. |
| How do you avoid leakage? | Seeded deterministic splits, the threshold chosen on out-of-fold predictions, and the test set never read by the gate. |
| What if the new model is better but less fair? | It is REJECTED. The gate is hard, not a weighted trade-off. |
| What happens when labels arrive late? | Performance and fairness report **NA** until ≥ 300 labelled rows are in the window; NA never counts toward a trigger streak. |
| How do you roll back? | Flip the gateway back to the previous slot (one call, measured zero errors). |
| Why Evidently and scipy? | Evidently is the default; a scipy path runs the same tests, and an engine-agreement test (rel. 1e-3) caught a chi-square mismatch. |
| Scaling? | Stateless slots behind a gateway; Postgres for the log. Horizontal scaling and k8s are outside the course scope. |

## Team split – 3 presenters

The talk follows the model's lifecycle: **Build → Trust → Operate**. Each presenter has a part of
about 7 minutes, controls the laptop during their part, and answers the questions on their topic.

| | Presenter 1 – **Build** | Presenter 2 – **Trust** | Presenter 3 – **Operate** |
| --- | --- | --- | --- |
| Sections | 1 Problem and architecture, 2 Data, 3 Training and tracking | 4 Fairness gate, 5 Live serving and SHAP | 6 Blue-green, 7 Monitoring, 8 Engineering quality, 9 Status and caveats |
| Time | 7 min (2 + 2 + 3) | 6 min (3 + 3) | 8 min (2 + 4 + 1 + 1) |
| Screens | README, terminal (`dvc status`), MLflow run + nested runs | MLflow `gate_report.json` / `fairness_report.json`, MLflow model version tags, terminal (predict / explain / bad requests) | Terminal (admin/state, flips), Grafana Service / Model / Fairness, Prometheus alerts |
| Key numbers to know by heart | 30,000 rows, 22.12 % defaults; 18k/4.5k/3k/9×500 splits; 7 baselines → 3 tuned × 25 trials; test PR-AUC 0.570 vs 0.22 base rate | G1–G5 all PASS; EO sex 0.027 / age 0.048 / inter 0.075; DI 0.96 / 0.92 / 0.88; PAY_0 is the top SHAP feature; predict p95 28 ms, explain p95 43 ms | Flip: 100/100 OK, 0 errors; 5 windows, 2,502 predictions, 2,000 labels; window-2 eo_age ALERT 0.156; triggers need 3 drift or 2 performance windows in a row; 258 tests |
| Owns these Q&A rows | Leakage; PR-AUC vs accuracy; scaling | SEX/AGE as features; EO + DI choice; "better but less fair?" | Late labels; rollback; Evidently vs scipy; window-2 fairness alert |

### Handoff lines (say them word for word to keep the story flowing)

- **1 → 2:** "So we have a reproducible, tracked model with PR-AUC 0.57 on held-out data. But in credit,
  *accurate* isn't enough. [Presenter 2] will show why this model was allowed into the registry at all."
- **2 → 3:** "So the model is fair at release time and every decision is explainable. But data changes
  after deployment. [Presenter 3] will show how we ship it safely and keep watching it."
- **3 closes:** the status and caveats slide, then "Questions?" All three stay at the front for Q&A.

### Prep work per person (tonight)

| Presenter 1 | Presenter 2 | Presenter 3 |
| --- | --- | --- |
| Runs the T-30 checklist (section 0) tomorrow and confirms `readyz` is true before the start | Prepares `$BODY` in the terminal and has the predict / explain / bad-key commands in shell history | Has the flip commands in shell history, and Grafana open and logged in with the time range set to cover today's replay |
| Screenshots: README architecture, MLflow run page, nested runs | Screenshots: gate_report, fairness_report, model version tags, explain output | Screenshots: Grafana Service, Model, Fairness dashboards and Prometheus alerts |
| Reads `docs/evidence/phase_2.md`, `phase_3.md`, `phase_4.md` | Reads `phase_5.md`, `phase_7.md` | Reads `phase_6.md`, `phase_8.md` |

**Rehearse once end to end with a timer.** If the run goes over 21 min, Presenter 3 cuts section 8 to one sentence.
If the live stack fails during someone's part, that person switches to their screenshots and keeps going.

---

## MLOps tool ownership – who explains which tool

Each presenter owns the tools that appear in their part, so the person who demos a tool also explains
it. For every tool, be able to say in about 30 seconds **what it does here, where it is in the repo, why
we chose it over the obvious alternative**, and answer the "be ready for" question.

### Presenter 1 – Build (data and experimentation)

| Tool | Version | What it does here | Where | Be ready for |
| --- | --- | --- | --- | --- |
| **DVC** | 3.67.1 | Versions the raw CSV and the `prepare → split` pipeline; `dvc repro` rebuilds identical splits | `dvc.yaml`, `dvc.lock`, `data/raw/*.dvc` | "Why not keep the data in git?" Large files, and hashes in `dvc.lock` prove which data trained the model |
| **MinIO** | chainguard image | S3-compatible storage: DVC remote (`dvc` bucket) and MLflow artifacts (`mlflow` bucket) | `infra/compose/docker-compose.yml` | "Why not real S3?" Runs locally for free and works with the same S3 API, so moving to AWS is a config change |
| **Great Expectations** | 1.24.0 | Data contract: suites `raw_taiwan` (ingestion) and `batch` (training frame = gate G1) | `pipelines/expectations/` | "What does it catch?" Schema, value ranges and allowed categories, e.g. AGE ≥ 18 and valid PAY codes |
| **pandas + scikit-learn** | 3.0.6 / 1.9.1 | Feature pipeline (category remap, encoding), 7 baseline models, CV | `src/fairxai/features/`, `src/fairxai/models/` | "Why a Pipeline object?" The same preprocessing ships with the model, so there is no training/serving skew |
| **XGBoost** | 3.2.0 | The selected model (398 trees, depth 5) | `configs/training.yaml` | "Why not deep learning?" Tabular data with 23 features; trees win here and TreeSHAP is exact and fast |
| **Optuna** | 5.0.0 | TPE tuning, seed 42, 25 trials × top 3 models | `src/fairxai/models/` | "Why not grid search?" Bayesian search reaches a better score in fewer trials, and the seed makes it reproducible |
| **MLflow Tracking** | 3.17.1 | Logs params, metrics, 72 nested trial runs and artifacts to Postgres + MinIO | http://127.0.0.1:5000 | "How do you reproduce a run?" `git_commit` + `data_version_id` + config snapshot are on every run |

### Presenter 2 – Trust (governance, fairness, explainability, serving)

| Tool | Version | What it does here | Where | Be ready for |
| --- | --- | --- | --- | --- |
| **MLflow Model Registry** | 3.17.1 | Versions the model; alias `candidate` only if APPROVED; lineage tags | MLflow → Models | "Why aliases, not stages?" Stages are deprecated in MLflow 3; aliases (`candidate`, later `champion`) are the current API |
| **Fairness gate (our code)** | – | EO / DI / intersectional checks, gates G1–G5, APPROVED / REJECTED / INDETERMINATE | `src/fairxai/fairness/`, `src/fairxai/registry/`, `configs/fairness.yaml` | "Why not Fairlearn / AIF360?" The metrics are a few lines of tested code; what we needed was the **gate logic**, its INDETERMINATE handling and the 81-case truth table, which those libraries don't provide |
| **SHAP** | 0.51.0 | Global importance at training time; per-request explanations at `/v1/explain` (tree_shap, log-odds) | `src/fairxai/explain/` | "Why log-odds?" SHAP values add up exactly in log-odds; in probability space they don't |
| **FastAPI + Pydantic** | 0.143.2 / 2.14.1 | Serving slot API; Pydantic enforces the input schema (422 on bad input, never logged) | `serving/` | "How do you stop bad input?" Strict schema: types, ranges, unknown fields rejected; 24 boundary tests |
| **Uvicorn** | 0.54.0 | ASGI server running the slots and the gateway | `serving/Dockerfile` | – |

### Presenter 3 – Operate (deployment, observability, monitoring, quality)

| Tool | Version | What it does here | Where | Be ready for |
| --- | --- | --- | --- | --- |
| **Docker + Compose** | Compose v2 | 8 services, health checks, `--profile core`, ports bound to 127.0.0.1 only | `infra/compose/docker-compose.yml` | "Why not Kubernetes?" Course scope and a single machine; the services are stateless containers, so they would port to k8s |
| **Gateway (our FastAPI app)** | – | Blue-green routing, `/admin/active` flip, 409 when the target slot isn't ready | `gateway/` | "Why not nginx?" The gateway checks slot readiness before a flip and stores the active slot in Postgres |
| **PostgreSQL** | 16 | Prediction log, labels, monitoring results, events, gateway state; MLflow backend | `infra/postgres/schema.sql` | "Why log every prediction?" Monitoring and later labels need the inputs and outputs, matched by `request_id` |
| **Prometheus** (+ prometheus-client) | v3.5.0 / 0.26.0 | Scrapes `/metrics` from the 3 services; 5 alert rules | `infra/prometheus/` | "Which alerts?" High5xxRate, HighPredictLatencyP95, SlotDown, ActiveSlotWithoutModel, PredictionLogFailures |
| **Grafana** | 12.1.1 | 4 dashboards provisioned from code (no manual clicking), Prometheus + Postgres data sources | `infra/grafana/` | "Dashboards as code?" Yes, JSON files in git, provisioned when the container starts |
| **Evidently** | 0.7.23 | Drift: KS (numeric), chi-square (categorical), α 0.05; a scipy fallback must agree to 1e-3 | `src/fairxai/monitoring/drift.py` | "Why a fallback?" It guards against library changes; the agreement test already caught a chi-square mismatch |
| **GitHub Actions + ruff + pytest** | ruff 0.17.1 / pytest 9.1.1 | CI on every push: lint, format, tree check, 258 tests | `.github/workflows/ci.yml`, `tests/` | "What runs in CI?" Lint, format check, repo-tree check and the full test suite on every push |

### Cross-cutting (anyone can answer)

- **Config:** every threshold is in `configs/*.yaml`, validated by `src/fairxai/config.py`; a test fails on hard-coded thresholds.
- **Secrets:** `.env` is generated by `scripts/make_env.py`, git-ignored; only SHA-256 hashes of API keys are stored.
- **Python 3.11 exactly**, dependencies pinned in `requirements/*.lock`.
