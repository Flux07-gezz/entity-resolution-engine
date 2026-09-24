"""Candidate generation (blocking) module for entity resolution.

Implements complementary, high-recall blocking passes:
- Pass A: Address number + postal / state key
- Pass B: City + informative normalized name token
- Pass C: Name prefix / 3-gram character key
- Pass D: Token inverted index (for fuzzy/transposed matching)

Tracks blocking paths, candidate statistics, and candidate recall.
"""

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import numpy as np


STOP_WORDS = {
    "the", "and", "of", "in", "at", "for", "on", "a", "an",
    "inc", "corp", "llc", "ltd", "co", "company", "corporation",
    "services", "group", "holdings", "solutions", "enterprises",
}


def get_informative_tokens(text: str, min_len: int = 3) -> List[str]:
    """Extract non-trivial tokens excluding generic business stop words."""
    if not text:
        return []
    tokens = text.split()
    return [t for t in tokens if len(t) >= min_len and t not in STOP_WORDS]


class CandidateBlocker:
    """Multi-pass candidate generator matching Source 1 to Source 2 and Source 3."""

    def __init__(
        self,
        max_candidates_per_s1: int = 100,
        enable_pass_a: bool = True,  # house_number + postal/state
        enable_pass_b: bool = True,  # city + name_token
        enable_pass_c: bool = True,  # name prefix (3 chars) + city
        enable_pass_d: bool = True,  # longest name token
    ):
        self.max_candidates_per_s1 = max_candidates_per_s1
        self.enable_pass_a = enable_pass_a
        self.enable_pass_b = enable_pass_b
        self.enable_pass_c = enable_pass_c
        self.enable_pass_d = enable_pass_d

    def generate_candidate_pairs(
        self,
        s1_records: Dict[str, Dict[str, Any]],
        target_records: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Generate candidate pairs between S1 and target records (S2 and S3).

        target_records contains records from S2 and S3.
        Each record dict should have:
            - 'name': str
            - 'address': str
            - 'house_number': Optional[str]
            - 'postal_code': Optional[str]
            - 'city': Optional[str]
            - 'state': Optional[str]
            - 'source': 's2' | 's3' (optional)
        """
        # Build indexes on target records
        idx_a = defaultdict(list)  # (house_number, postal_or_state) -> [target_id]
        idx_b = defaultdict(list)  # (city, token) -> [target_id]
        idx_c = defaultdict(list)  # (city, prefix_3) -> [target_id]
        idx_d = defaultdict(list)  # longest_token -> [target_id]

        for tid, t_data in target_records.items():
            name = t_data.get("name", "") or ""
            city = t_data.get("city", "") or ""
            state = t_data.get("state", "") or ""
            postal = t_data.get("postal_code", "") or ""
            house_num = t_data.get("house_number", "") or ""

            # Pass A Index
            loc_key = postal if postal else state
            if house_num and loc_key:
                idx_a[(house_num, loc_key)].append(tid)

            # Informative tokens
            tokens = get_informative_tokens(name)

            # Pass B Index
            if city and tokens:
                for tok in tokens[:3]:
                    idx_b[(city, tok)].append(tid)

            # Pass C Index
            if city and len(name) >= 3:
                prefix = name[:3]
                idx_c[(city, prefix)].append(tid)

            # Pass D Index (longest token)
            if tokens:
                longest_tok = max(tokens, key=len)
                if len(longest_tok) >= 4:
                    idx_d[longest_tok].append(tid)

        candidate_pairs: List[Dict[str, Any]] = []

        for s1_id, s1_data in s1_records.items():
            s1_name = s1_data.get("name", "") or ""
            s1_city = s1_data.get("city", "") or ""
            s1_state = s1_data.get("state", "") or ""
            s1_postal = s1_data.get("postal_code", "") or ""
            s1_house_num = s1_data.get("house_number", "") or ""

            matched_targets: Dict[str, Set[str]] = defaultdict(set)

            # Pass A
            if self.enable_pass_a:
                s1_loc = s1_postal if s1_postal else s1_state
                if s1_house_num and s1_loc:
                    for tid in idx_a.get((s1_house_num, s1_loc), []):
                        matched_targets[tid].add("pass_a")

            s1_tokens = get_informative_tokens(s1_name)

            # Pass B
            if self.enable_pass_b and s1_city and s1_tokens:
                for tok in s1_tokens[:3]:
                    for tid in idx_b.get((s1_city, tok), []):
                        matched_targets[tid].add("pass_b")

            # Pass C
            if self.enable_pass_c and s1_city and len(s1_name) >= 3:
                prefix = s1_name[:3]
                for tid in idx_c.get((s1_city, prefix), []):
                    matched_targets[tid].add("pass_c")

            # Pass D
            if self.enable_pass_d and s1_tokens:
                longest_tok = max(s1_tokens, key=len)
                if len(longest_tok) >= 4:
                    # Guard against overly generic keys with excessive matches
                    candidates_d = idx_d.get(longest_tok, [])
                    if len(candidates_d) <= 200:
                        for tid in candidates_d:
                            matched_targets[tid].add("pass_d")

            # Apply candidate budget per S1
            items = list(matched_targets.items())
            if len(items) > self.max_candidates_per_s1:
                # Prioritize pairs hitting multiple passes
                items.sort(key=lambda x: len(x[1]), reverse=True)
                items = items[:self.max_candidates_per_s1]

            for tid, passes in items:
                candidate_pairs.append({
                    "s1_id": s1_id,
                    "target_id": tid,
                    "blocking_passes": list(passes),
                    "target_source": target_records[tid].get("source", "unknown"),
                })

        return candidate_pairs


def evaluate_blocking(
    candidate_pairs: List[Dict[str, Any]],
    ground_truth: Dict[str, Sequence[str]],
    all_s1_ids: Sequence[str],
) -> Dict[str, float]:
    """Compute candidate recall and candidate distribution metrics."""
    s1_candidates = defaultdict(set)
    for pair in candidate_pairs:
        s1_candidates[pair["s1_id"]].add(pair["target_id"])

    total_true_matches = 0
    recovered_true_matches = 0

    candidate_counts = []
    for s1_id in all_s1_ids:
        cands = s1_candidates[s1_id]
        candidate_counts.append(len(cands))
        true_matches = set(ground_truth.get(s1_id, []))
        total_true_matches += len(true_matches)
        recovered_true_matches += len(true_matches & cands)

    candidate_recall = (
        (recovered_true_matches / total_true_matches)
        if total_true_matches > 0
        else 1.0
    )

    counts_arr = np.array(candidate_counts)
    return {
        "candidate_recall": candidate_recall,
        "recovered_true_matches": float(recovered_true_matches),
        "total_true_matches": float(total_true_matches),
        "avg_candidates_per_s1": float(np.mean(counts_arr)),
        "median_candidates": float(np.median(counts_arr)),
        "p90_candidates": float(np.percentile(counts_arr, 90)),
        "p99_candidates": float(np.percentile(counts_arr, 99)),
        "max_candidates": float(np.max(counts_arr)),
    }
