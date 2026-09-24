"""Feature engineering module for candidate pairs.

Extracts semantic evidence vectors:
1. Name similarity features (Jaro-Winkler, Levenshtein, token set/sort, Jaccard).
2. Address similarity features (full address, street, city, state, postal).
3. Contradiction features (house number conflict, postal conflict, city conflict).
4. Missingness features (asymmetric presence of attributes).
5. Source and blocking metadata features.
"""

from typing import Any, Dict, List, Optional, Set
import numpy as np

try:
    from rapidfuzz import fuzz
    from rapidfuzz.distance import JaroWinkler, Levenshtein
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False


def token_jaccard(tokens1: Set[str], tokens2: Set[str]) -> float:
    """Compute Jaccard similarity over token sets."""
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return intersection / union if union > 0 else 0.0


def extract_pair_features(
    s1_record: Dict[str, Any],
    target_record: Dict[str, Any],
    blocking_passes: Optional[List[str]] = None,
    target_source: str = "s2",
) -> Dict[str, float]:
    """Extract a rich, bounded evidence vector for a single (S1, Target) candidate pair."""
    s1_name = str(s1_record.get("name", "") or "").lower().strip()
    t_name = str(target_record.get("name", "") or "").lower().strip()

    s1_addr = str(s1_record.get("address", "") or "").lower().strip()
    t_addr = str(target_record.get("address", "") or "").lower().strip()

    s1_tokens = set(s1_name.split())
    t_tokens = set(t_name.split())
    s1_addr_tokens = set(s1_addr.split())
    t_addr_tokens = set(t_addr.split())

    feats: Dict[str, float] = {}

    # 1. Name Features
    feats["name_exact_match"] = 1.0 if (s1_name and s1_name == t_name) else 0.0

    if RAPIDFUZZ_AVAILABLE and s1_name and t_name:
        feats["name_jaro_winkler"] = float(JaroWinkler.similarity(s1_name, t_name))
        feats["name_levenshtein"] = float(Levenshtein.normalized_similarity(s1_name, t_name))
        feats["name_token_sort_ratio"] = float(fuzz.token_sort_ratio(s1_name, t_name)) / 100.0
        feats["name_token_set_ratio"] = float(fuzz.token_set_ratio(s1_name, t_name)) / 100.0
    else:
        # Fallback simple string matching if rapidfuzz not yet ready
        common_chars = len(set(s1_name) & set(t_name))
        total_chars = max(len(s1_name), len(t_name), 1)
        feats["name_jaro_winkler"] = common_chars / total_chars
        feats["name_levenshtein"] = feats["name_jaro_winkler"]
        feats["name_token_sort_ratio"] = token_jaccard(s1_tokens, t_tokens)
        feats["name_token_set_ratio"] = token_jaccard(s1_tokens, t_tokens)

    feats["name_token_jaccard"] = token_jaccard(s1_tokens, t_tokens)
    feats["name_shared_tokens"] = float(len(s1_tokens & t_tokens))
    feats["name_len_diff"] = float(abs(len(s1_name) - len(t_name)))
    feats["name_len_ratio"] = (
        min(len(s1_name), len(t_name)) / max(len(s1_name), len(t_name))
        if max(len(s1_name), len(t_name)) > 0
        else 1.0
    )
    # Prefix similarity
    prefix_len = 4
    feats["name_prefix_match"] = (
        1.0 if (len(s1_name) >= prefix_len and len(t_name) >= prefix_len and s1_name[:prefix_len] == t_name[:prefix_len])
        else 0.0
    )

    # 2. Address Features
    feats["addr_exact_match"] = 1.0 if (s1_addr and s1_addr == t_addr) else 0.0
    if RAPIDFUZZ_AVAILABLE and s1_addr and t_addr:
        feats["addr_token_sort_ratio"] = float(fuzz.token_sort_ratio(s1_addr, t_addr)) / 100.0
        feats["addr_token_set_ratio"] = float(fuzz.token_set_ratio(s1_addr, t_addr)) / 100.0
    else:
        feats["addr_token_sort_ratio"] = token_jaccard(s1_addr_tokens, t_addr_tokens)
        feats["addr_token_set_ratio"] = token_jaccard(s1_addr_tokens, t_addr_tokens)

    feats["addr_token_jaccard"] = token_jaccard(s1_addr_tokens, t_addr_tokens)

    # 3. Parsed Components & Contradictions
    s1_house = s1_record.get("house_number")
    t_house = target_record.get("house_number")
    if s1_house and t_house:
        feats["house_num_match"] = 1.0 if s1_house == t_house else 0.0
        feats["house_num_conflict"] = 1.0 if s1_house != t_house else 0.0
    else:
        feats["house_num_match"] = 0.0
        feats["house_num_conflict"] = 0.0

    s1_post = s1_record.get("postal_code")
    t_post = target_record.get("postal_code")
    if s1_post and t_post:
        feats["postal_match"] = 1.0 if s1_post == t_post else 0.0
        feats["postal_conflict"] = 1.0 if s1_post != t_post else 0.0
    else:
        feats["postal_match"] = 0.0
        feats["postal_conflict"] = 0.0

    s1_city = s1_record.get("city")
    t_city = target_record.get("city")
    if s1_city and t_city:
        feats["city_match"] = 1.0 if s1_city == t_city else 0.0
        feats["city_conflict"] = 1.0 if s1_city != t_city else 0.0
    else:
        feats["city_match"] = 0.0
        feats["city_conflict"] = 0.0

    s1_state = s1_record.get("state")
    t_state = target_record.get("state")
    if s1_state and t_state:
        feats["state_match"] = 1.0 if s1_state == t_state else 0.0
        feats["state_conflict"] = 1.0 if s1_state != t_state else 0.0
    else:
        feats["state_match"] = 0.0
        feats["state_conflict"] = 0.0

    s1_unit = s1_record.get("unit")
    t_unit = target_record.get("unit")
    if s1_unit and t_unit:
        feats["unit_match"] = 1.0 if s1_unit == t_unit else 0.0
        feats["unit_conflict"] = 1.0 if s1_unit != t_unit else 0.0
    else:
        feats["unit_match"] = 0.0
        feats["unit_conflict"] = 0.0

    # 4. Missingness & Asymmetry Features
    feats["missing_house_asymm"] = 1.0 if bool(s1_house) != bool(t_house) else 0.0
    feats["missing_postal_asymm"] = 1.0 if bool(s1_post) != bool(t_post) else 0.0
    feats["missing_city_asymm"] = 1.0 if bool(s1_city) != bool(t_city) else 0.0

    # 5. Blocking metadata
    passes = set(blocking_passes or [])
    feats["blocking_pass_count"] = float(len(passes))
    feats["has_pass_a"] = 1.0 if "pass_a" in passes else 0.0
    feats["has_pass_b"] = 1.0 if "pass_b" in passes else 0.0
    feats["has_pass_c"] = 1.0 if "pass_c" in passes else 0.0
    feats["has_pass_d"] = 1.0 if "pass_d" in passes else 0.0

    # Source indicators
    feats["target_is_s2"] = 1.0 if target_source == "s2" else 0.0
    feats["target_is_s3"] = 1.0 if target_source == "s3" else 0.0

    return feats
