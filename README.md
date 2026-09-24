# Amazon ML Challenge 2026 — Business Entity Resolution

Competitive, precision-oriented entity resolution pipeline designed for Macro $F_{0.5}$ optimization.

## Pipeline Architecture
- **Anchor Source**: Source 1
- **Target Sources**: Source 2, Source 3
- **Objective**: Maximize Entity-Level Macro $F_{0.5}$
- **Core Principles**:
  - High candidate recall in blocking
  - Aggressive false-merge reduction
  - Accurate singleton / empty-match identification
  - Leakage-safe grouped validation on Source 1 IDs

## Directory Structure
- `data/`: Raw and processed challenge data (Source 1, Source 2, Source 3, ground truth)
- `src/`: Core pipeline stages (normalization, blocking, address parsing, feature engineering, modeling, decision policy)
- `evaluation/`: Leakage-safe splits, entity-level metrics, threshold search, error analysis
- `utils/`: Submission formatting and validation helpers
- `configs/`: Pipeline configuration
- `tests/`: Unit tests and adversarial regression suite
- `models/`: Trained model artifacts
- `output/`: Generated submission files (`matching_results.tsv`)
