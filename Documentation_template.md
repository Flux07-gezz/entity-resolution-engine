# Methodology Documentation — Amazon ML Challenge 2026: Business Entity Resolution

## 1. Problem Formulation
- **Anchor**: Source 1 entities mapped to matching entities in Source 2 and Source 3.
- **Cardinality**: 0, 1, or $N$ matches per Source 1 entity (non-bijective linkage).
- **Target Metric**: Entity-level Macro $F_{0.5}$, penalizing false merges twice as heavily as false dismissals.

## 2. Dataset & Profile Summary
- **Source 1 Records**: [Count]
- **Source 2 Records**: [Count]
- **Source 3 Records**: [Count]
- **Ground Truth Match Cardinality**:
  - Zero-match (singletons): [%]
  - 1-match: [%]
  - Multi-match: [%]
  - Max matches for a single S1: [Count]
- **Key Regional / Domain Characteristics**:
  - Missingness patterns, abbreviations, postal and legal suffix structures observed.

## 3. Normalization Approach
- Two-tier representation: `raw_*` preserved alongside `normalized_*`.
- Conservative canonicalization: lowercase, unicode normalisation, punctuation clean-up, preserving address numbers, unit numbers, and essential tokens.

## 4. Blocking Strategy
- Multi-pass union blocking:
  - Pass A: Address number + postal / city key
  - Pass B: City + informative normalized name token
  - Pass C: High-frequency name n-gram / fuzzy candidate generator
- Candidate budget control: [Avg candidates / S1]

## 5. Measured Candidate Recall
- **Candidate Recall**: [Measured % on validation set]
- **S1 entities with 100% recall**: [%]
- **Candidate Distribution**: [Percentiles: P50, P90, P99, Max]

## 6. Feature Engineering
- **Name Features**: Jaro-Winkler, Levenshtein ratio, token sort ratio, token set overlap, shared n-grams, legal entity suffix consistency.
- **Address Features**: House number exact/partial match, street token similarity, city/state agreement, postal code prefix/exact match.
- **Contradiction Features**:
  - Hard: Incompatible house numbers or distinct non-overlapping postal codes (where reliable).
  - Soft: City mismatch flags, unit conflicts.
- **Missingness Features**: Indicators for asymmetric missing components.

## 7. Hard-Negative Strategy
- Challenging negative mining from candidate blocking:
  - Same name, different address / branch (franchise collisions)
  - Same address, different business name
  - Near-phonetic and prefix collisions

## 8. Model Architecture & Grouped Validation
- **Model**: Gradient Boosted Trees (LightGBM / CatBoost).
- **Validation**: GroupKFold grouped strictly on Source 1 IDs (zero S1 leakage across folds).
- **Target**: $P(\text{pair is true match})$.

## 9. Threshold & Calibration Strategy
- Out-of-fold probability calibration (isotonic / sigmoid).
- Threshold selection optimized directly for Entity-Level Macro $F_{0.5}$ via bounded grid search.

## 10. Entity-Level Evaluation Results
- **Candidate Recall**: [Value]
- **Pair Precision / Recall**: [Value]
- **Entity Macro $F_{0.5}$**: [Value]
- **Singleton Accuracy**: [Value]
- **Exact Set Accuracy**: [Value]
- **False Merges / 100 S1**: [Value]
- **Missed Matches / 100 S1**: [Value]

## 11. Error Analysis & False Merge Case Studies
- False Merge Category breakdown (franchise, address collision, parser ambiguity).
- Targeted mitigations applied.

## 12. Reproducibility Instructions
- Step-by-step reproduction command line to recreate outputs, models, and predictions from raw data.
