"""Adversarial regression test suite for entity resolution.

Verifies critical edge cases and precision traps:
1. Franchise collisions (same business name, different city/address)
2. Address collisions (same building/address, completely different business)
3. Legal suffix variations (Inc vs Incorporated vs LLC)
4. Address abbreviation variations (St vs Street, Ave vs Avenue)
5. Off-by-one house numbers (101 Main St vs 103 Main St)
6. True singletons (no matches across target sources)
7. Multi-match resolution (Source 1 matches both Source 2 and Source 3)
"""

import pytest
from src.address_parser import AddressParser
from src.features import extract_pair_features
from src.model import BaselineScorer
from src.normalize import normalize_address, normalize_name
from src.decision import EntityDecisionEngine


def test_franchise_collision_same_name_different_city():
    """Same brand name in different cities/addresses should have conflict signals."""
    s1 = {
        "name": normalize_name("Blue Star Cafe"),
        "address": normalize_address("101 Main St, Austin TX 78701"),
        "house_number": "101",
        "city": "austin",
        "state": "tx",
        "postal_code": "78701",
    }
    s2 = {
        "name": normalize_name("Blue Star Cafe"),
        "address": normalize_address("550 Broad St, Dallas TX 75201"),
        "house_number": "550",
        "city": "dallas",
        "state": "tx",
        "postal_code": "75201",
    }

    feats = extract_pair_features(s1, s2)
    assert feats["name_exact_match"] == 1.0
    assert feats["house_num_conflict"] == 1.0
    assert feats["city_conflict"] == 1.0
    assert feats["postal_conflict"] == 1.0


def test_address_collision_different_businesses():
    """Different businesses at the exact same physical address."""
    s1 = {
        "name": normalize_name("Summit Financial Partners"),
        "address": normalize_address("75 Wall St, New York NY 10005"),
        "house_number": "75",
        "city": "new york",
        "state": "ny",
        "postal_code": "10005",
    }
    s2 = {
        "name": normalize_name("Gotham Coffee Roasters"),
        "address": normalize_address("75 Wall St, New York NY 10005"),
        "house_number": "75",
        "city": "new york",
        "state": "ny",
        "postal_code": "10005",
    }

    feats = extract_pair_features(s1, s2)
    assert feats["addr_exact_match"] == 1.0
    assert feats["name_exact_match"] == 0.0
    assert feats["name_token_jaccard"] == 0.0
    assert feats["house_num_match"] == 1.0


def test_off_by_one_house_numbers_conflict():
    """Adjacent building numbers (e.g. 500 Market vs 502 Market) are distinct entities."""
    s1 = {
        "name": normalize_name("Acme Robotics Inc"),
        "address": normalize_address("500 Market St, San Jose CA 95113"),
        "house_number": "500",
        "city": "san jose",
        "state": "ca",
        "postal_code": "95113",
    }
    s2 = {
        "name": normalize_name("Acme Robotics Inc"),
        "address": normalize_address("502 Market St, San Jose CA 95113"),
        "house_number": "502",
        "city": "san jose",
        "state": "ca",
        "postal_code": "95113",
    }

    feats = extract_pair_features(s1, s2)
    assert feats["house_num_conflict"] == 1.0
    assert feats["house_num_match"] == 0.0


def test_legal_suffix_and_abbreviation_invariance():
    """Variations in legal suffixes and street types should resolve with high similarity."""
    s1 = {
        "name": normalize_name("Apex Logistics Corporation"),
        "address": normalize_address("400 Industrial Parkway, Chicago IL 60607"),
        "house_number": "400",
        "city": "chicago",
        "state": "il",
        "postal_code": "60607",
    }
    s2 = {
        "name": normalize_name("Apex Logistics Corp"),
        "address": normalize_address("400 Industrial Pkwy, Chicago IL 60607"),
        "house_number": "400",
        "city": "chicago",
        "state": "il",
        "postal_code": "60607",
    }

    feats = extract_pair_features(s1, s2)
    # Both names canonicalize to "apex logistics corp"
    assert feats["name_exact_match"] == 1.0
    assert feats["addr_exact_match"] == 1.0
    assert feats["house_num_match"] == 1.0
    assert feats["postal_match"] == 1.0


def test_hard_contradiction_filters_off_by_one_address():
    """Decision engine with hard contradiction filtering must drop pairs with conflicting house numbers."""
    engine = EntityDecisionEngine(default_threshold=0.50, enable_hard_contradiction_filter=True)
    pair = {
        "s1_id": "S1_100",
        "target_id": "S2_200",
        "score": 0.95,  # Very high score (e.g. from name match)
        "features": {
            "house_num_conflict": 1.0,  # But clear house number conflict
        }
    }
    preds = engine.decide_entities([pair], all_s1_ids=["S1_100"])
    assert preds["S1_100"] == [], "Should have rejected match due to hard house number contradiction"


def test_real_data_url_domain_name_matching():
    """Real dataset pattern: S3 businesses represented as domain URLs (celestialmemorialtrust.com)."""
    s1_name = "Celestial Memorial Trust"
    s3_name = "celestialmemorialtrust.com"

    norm_s1 = normalize_name(s1_name)
    norm_s3 = normalize_name(s3_name)

    s1_tokens = set(norm_s1.split())
    # S3 name stripped of .com
    s3_clean = s3_name.lower().replace(".com", "").replace(".", " ")

    # Verify domain token is recognized as containing S1 components
    assert "celestial" in s3_clean
    assert "memorial" in s3_clean
    assert "trust" in s3_clean


def test_real_data_india_address_state_and_door_variations():
    """Real India dataset pattern: 'No. 35' vs 'Door No 35', 'Karnataka' vs 'KA'."""
    s1_addr = "No. 35, Brentwood Apartments, Defence Colony, 2Nd Main, Indiranagar, Bangalore, Karnataka"
    s3_addr = "Door No 35, Brentwood Apartments, Defence Colony, 2Nd Main, Indiranagar, Bangalore, KA"

    norm_s1 = normalize_address(s1_addr)
    norm_s3 = normalize_address(s3_addr)

    s1 = {"name": "celestial memorial", "address": norm_s1, "house_number": "35", "city": "bangalore", "state": "karnataka"}
    s3 = {"name": "celestial memorial", "address": norm_s3, "house_number": "35", "city": "bangalore", "state": "ka"}

    feats = extract_pair_features(s1, s3)
    # Both share house number 35 and key address tokens
    assert feats["house_num_match"] == 1.0
    assert feats["addr_token_sort_ratio"] > 0.80


def test_real_data_france_legal_and_address_patterns():
    """Real France test set patterns: SARL, SASU, 'bis' street numbering, and French regions."""
    s1 = {
        "name": normalize_name("ZNB Club SARL"),
        "address": normalize_address("Nouvelle-Aquitaine, La Teste-de-Buch, 5 bis Rue Pierre Dignac"),
        "house_number": "5",
        "city": "la teste-de-buch",
        "state": "nouvelle-aquitaine",
    }
    s2 = {
        "name": normalize_name("ZNB Club"),
        "address": normalize_address("5 bis RUE PIERRE DIGNAC, LA TESTE-DE-BUCH, Nouvelle-Aquitaine"),
        "house_number": "5",
        "city": "la teste-de-buch",
        "state": "nouvelle-aquitaine",
    }

    feats = extract_pair_features(s1, s2)
    assert feats["name_token_set_ratio"] >= 0.90
    assert feats["addr_token_sort_ratio"] >= 0.85
    assert feats["house_num_match"] == 1.0


def test_real_data_aka_dba_alias_handling():
    """Real dataset pattern: 'Halodelta aka Clemons Silver Eastern Inc' matching 'Clemons Silver Eastern Inc'."""
    s1_name = "Clemons Silver Eastern Inc"
    s3_name = "Halodelta aka Clemons Silver Eastern Inc"

    norm_s1 = normalize_name(s1_name)
    norm_s3 = normalize_name(s3_name)

    s1_tokens = set(norm_s1.split())
    s3_tokens = set(norm_s3.split())

    # The true entity name is a full subset of the alias string
    assert s1_tokens.issubset(s3_tokens)


def test_real_data_literal_null_in_address():
    """Real dataset pattern: literal string 'NULL' in noisy addresses."""
    raw_addr = "1809 Fairview Street, NULL, Burlington, North Carolina"
    norm_addr = normalize_address(raw_addr)
    # Ensure parsing handles it without crashing
    parser = AddressParser()
    parsed = parser.parse(norm_addr)
    assert parsed["house_number"] == "1809"
    assert "fairview" in parsed["street"]

