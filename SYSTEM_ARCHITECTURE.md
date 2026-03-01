# SYSTEM_ARCHITECTURE.md
## MLB Win Probability Modeling System — Architecture and Data Flow

---

# 1. High-Level Components

1. Ingestion Layer
   - MLB Stats API client (async, cached)
   - Retrosheet downloader (fallback-capable)

2. Processing Layer
   - Schedule normalization
   - Retrosheet GL parsing
   - Crosswalk builder (Retrosheet → MLB `game_pk`)

3. Feature Layer (planned)
   - deterministic feature computation
   - feature versioning + hashing

4. Model Layer (planned)
   - baseline logistic model
   - extensions: pitcher + lineup
   - calibration module

5. Scoring Layer (planned)
   - daily scoring run
   - immutable prediction snapshots

6. Monitoring Layer (planned)
   - drift computation
   - run metrics logging

---

# 2. Data Flow

## 2.1 Ingestion pipeline (current)

```text
MLB Stats API ──► raw cache JSON (data/raw/mlb_api)
             └─► processed schedule (data/processed/schedule)

Retrosheet sources:
  Chadwick raw ─┐
                ├─► raw GL TXT (data/raw/retrosheet/gamelogs)
Retrosheet ZIP ─┘
                └─► processed gamelogs (data/processed/retrosheet)

processed schedule + processed gamelogs
   └─► crosswalk (data/processed/crosswalk)
```

## 2.2 Scoring pipeline (planned)

```text
schedule + features + expected lineups + projected starters
   └─► scoring
         ├─► immutable snapshot parquet
         └─► drift computation
               ├─► per-season run metrics
               └─► global run metrics
```

---

# 3. Execution Model

## 3.1 Async + rate limiting

All external HTTP calls must:
- go through async clients
- be token-bucket throttled
- use bounded concurrency
- implement retries with backoff

No direct synchronous HTTP calls for external sources.

## 3.2 Multi-season runs

All ingestion scripts accept multiple seasons per invocation.

Orchestration script (`ingest_all.py`) runs:
- schedule ingestion
- retrosheet ingestion
- crosswalk building

in “collect mode” and exits nonzero on any failure.

---

# 4. Module Responsibilities

## 4.1 `src/winprob/mlbapi`

- async Stats API wrapper
- cache responses keyed by endpoint+params
- metadata JSONL for audit

## 4.2 `src/winprob/retrosheet`

- download + parse GL logs
- multiple sources + automatic fallback
- persist provenance metadata

## 4.3 `src/winprob/crosswalk`

- deterministically map Retrosheet games to MLB `game_pk`
- emit unresolved lists
- produce coverage report

## 4.4 Feature module (planned)

- ingest processed datasets
- emit deterministic feature matrices + feature hash
- maintain stable feature schema

## 4.5 Model module (planned)

- train baseline model
- serialize model artifacts with versioning
- provide scoring interface

---

# 5. Failure Modes and Handling

- API 429: respect Retry-After; backoff
- API 5xx: exponential retry; eventual failure with classification
- Retrosheet download failure: fallback source; log reason
- Crosswalk coverage < 99%: emit report and enforce policy
- Schema mismatch: raise SchemaError with diagnostics

---

# 6. Security / Compliance

- Preserve attribution requirements for Retrosheet.
- Do not store secrets; MLB Stats API access is anonymous.

---

# 7. Extensibility Guidelines

Any new module must:
- update DATA_SCHEMA.md when new datasets are introduced
- define deterministic hashes for derived artifacts
- provide tests (future) for schema stability
- document provenance and versioning behavior
