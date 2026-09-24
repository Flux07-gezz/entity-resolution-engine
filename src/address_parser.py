"""Heuristic address parsing based on observed structural patterns.

No external APIs, databases, or pretrained geocoders (strict compliance with competition rules).
Extracts:
- house_number
- street_name
- unit
- city
- state
- postal_code
"""

import re
from typing import Any, Dict, Optional


class AddressParser:
    """Rules-based address parser for business location strings."""

    def __init__(self):
        # US / Canadian / standard postal code regex (5 digits, 5+4 digits, or alphanumeric A1A 1A1)
        self.postal_re = re.compile(
            r"\b(\d{5}(?:-\d{4})?|[a-z]\d[a-z]\s*\d[a-z]\d)\b", re.IGNORECASE
        )
        # Leading house/building number
        self.house_num_re = re.compile(
            r"^\s*(\d+[a-z]?|\d+\-\d+)\b", re.IGNORECASE
        )
        # Unit / Suite patterns
        self.unit_re = re.compile(
            r"\b(?:ste|suite|apt|apartment|unit|bldg|building|fl|floor|#)\s*([a-z0-9\-]+)\b",
            re.IGNORECASE,
        )
        # Common 2-letter state / province codes before postal or end
        self.state_re = re.compile(
            r"\b(al|ak|az|ar|ca|co|ct|de|fl|ga|hi|id|il|in|ia|ks|ky|la|me|md|ma|mi|mn|ms|mo|mt|ne|nv|nh|nj|nm|ny|nc|nd|oh|ok|or|pa|ri|sc|sd|tn|tx|ut|vt|va|wa|wv|wi|wy|dc|on|bc|ab|mb|sk|qc|ns|nb)\b",
            re.IGNORECASE,
        )

    def parse(self, address_str: Optional[str]) -> Dict[str, Optional[str]]:
        """Parse raw or normalized address into structural components."""
        if not address_str:
            return {
                "house_number": None,
                "unit": None,
                "street": None,
                "city": None,
                "state": None,
                "postal_code": None,
            }

        addr = address_str.strip()
        house_number = None
        unit = None
        postal_code = None
        state = None
        city = None
        street = None

        # 1. Postal code
        post_match = self.postal_re.search(addr)
        if post_match:
            postal_code = post_match.group(1).lower().replace(" ", "")
            # Remove postal code for subsequent parsing
            addr_clean = addr[:post_match.start()] + addr[post_match.end():]
        else:
            addr_clean = addr

        # 2. House number
        house_match = self.house_num_re.search(addr_clean)
        if house_match:
            house_number = house_match.group(1).lower()

        # 3. Unit / suite
        unit_match = self.unit_re.search(addr_clean)
        if unit_match:
            unit = unit_match.group(1).lower()

        # 4. State
        state_matches = list(self.state_re.finditer(addr_clean))
        if state_matches:
            # Often the state is near the end before postal code
            last_state = state_matches[-1]
            state = last_state.group(1).lower()

        # 5. Comma-separated parts heuristic: "street, city, state postal"
        parts = [p.strip() for p in addr.split(",") if p.strip()]
        if len(parts) >= 2:
            street = parts[0]
            # City is typically the second component or second-to-last
            city_candidate = parts[1]
            # If state or postal code got caught in city_candidate, clean it
            city_tokens = [
                tok for tok in city_candidate.split()
                if not self.postal_re.match(tok) and (not state or tok.lower() != state)
            ]
            city = " ".join(city_tokens) if city_tokens else parts[1]
        else:
            street = addr_clean.strip()

        return {
            "house_number": house_number,
            "unit": unit,
            "street": street,
            "city": city,
            "state": state,
            "postal_code": postal_code,
        }
