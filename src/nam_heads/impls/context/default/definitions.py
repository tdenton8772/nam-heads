"""
Context head dimension definitions and seed data.

Fixed contextual dimensions: temporal, geographic, domain, institutional.
Seed data bootstraps the ContextStore on first run. At runtime the store
is authoritative; new resolutions discovered during ingest are persisted
for subsequent lookups.
"""

import re
from typing import Dict, List, Optional

from nam_heads.ontology import bindings as B

# ---------------------------------------------------------------------------
# Dimension registry
# ---------------------------------------------------------------------------
DIMENSIONS = ["temporal", "geographic", "domain", "institutional"]

# ---------------------------------------------------------------------------
# Temporal normalization (pure rules, no store needed for core patterns)
# ---------------------------------------------------------------------------
_YEAR_RE = re.compile(r"\b(\d{4})\b")
_CENTURY_RE = re.compile(r"\b(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)\s+century\b", re.IGNORECASE)
_DECADE_RE = re.compile(r"\b(?:the\s+)?(\d{4})s\b", re.IGNORECASE)
_YEAR_RANGE_RE = re.compile(r"\b(\d{4})\s*(?:to|[-\u2013])\s*(\d{4})\b", re.IGNORECASE)

# Named temporal periods -> canonical temporal value (seeded into store)
TEMPORAL_PERIODS: Dict[str, str] = {
    "world war i": "1910s",
    "world war ii": "1940s",
    "wwi": "1910s",
    "wwii": "1940s",
    "first world war": "1910s",
    "second world war": "1940s",
    "the great war": "1910s",
    "cold war": "1960s",
    "renaissance": "15th_century",
    "the renaissance": "15th_century",
    "middle ages": "12th_century",
    "the middle ages": "12th_century",
    "medieval": "12th_century",
    "industrial revolution": "19th_century",
    "the industrial revolution": "19th_century",
    "enlightenment": "18th_century",
    "the enlightenment": "18th_century",
    "antiquity": "5th_century",
    "classical antiquity": "5th_century",
    "victorian era": "1880s",
    "the victorian era": "1880s",
    "great depression": "1930s",
    "the great depression": "1930s",
    "french revolution": "1790s",
    "the french revolution": "1790s",
    "american revolution": "1770s",
    "the american revolution": "1770s",
    "reformation": "16th_century",
    "the reformation": "16th_century",
    "bronze age": "3rd_millennium_bce",
    "iron age": "1st_millennium_bce",
    "stone age": "prehistory",
}


def normalize_temporal_rules(text: str) -> Optional[str]:
    """Apply rule-based temporal normalization (no store lookup).

    Returns a canonical temporal string or None.
    """
    lower = text.lower().strip()

    # Named period (exact match on lowered text)
    if lower in TEMPORAL_PERIODS:
        return TEMPORAL_PERIODS[lower]

    # Century pattern: "the 14th century" -> "14th_century"
    m = _CENTURY_RE.search(lower)
    if m:
        ordinal = int(m.group(1))
        suffixes = {1: "st", 2: "nd", 3: "rd"}
        suffix = suffixes.get(ordinal, "th")
        return f"{ordinal}{suffix}_century"

    # Decade pattern: "the 1990s" -> "1990s"
    m = _DECADE_RE.search(lower)
    if m:
        decade_base = int(m.group(1))
        return f"{decade_base}s"

    # Year range: "1939 to 1945" -> decade of midpoint
    m = _YEAR_RANGE_RE.search(lower)
    if m:
        y1, y2 = int(m.group(1)), int(m.group(2))
        mid = (y1 + y2) // 2
        decade = (mid // 10) * 10
        return f"{decade}s"

    # 4-digit year -> decade: "1935" -> "1930s"
    m = _YEAR_RE.search(lower)
    if m:
        year = int(m.group(1))
        if 100 <= year <= 2100:
            decade = (year // 10) * 10
            return f"{decade}s"

    return None


# ---------------------------------------------------------------------------
# Geographic seed data: country/territory -> region
# ---------------------------------------------------------------------------
GEOGRAPHIC_SEED: Dict[str, str] = {
    # Europe
    "united kingdom": "united_kingdom",
    "uk": "united_kingdom",
    "great britain": "united_kingdom",
    "england": "united_kingdom",
    "scotland": "united_kingdom",
    "wales": "united_kingdom",
    "northern ireland": "united_kingdom",
    "france": "france",
    "germany": "germany",
    "bavaria": "germany",
    "prussia": "germany",
    "saxony": "germany",
    "italy": "italy",
    "spain": "spain",
    "portugal": "portugal",
    "netherlands": "netherlands",
    "holland": "netherlands",
    "belgium": "belgium",
    "luxembourg": "luxembourg",
    "switzerland": "switzerland",
    "austria": "austria",
    "ireland": "ireland",
    "denmark": "denmark",
    "norway": "norway",
    "sweden": "sweden",
    "finland": "finland",
    "iceland": "iceland",
    "poland": "poland",
    "czech republic": "czech_republic",
    "czechia": "czech_republic",
    "slovakia": "slovakia",
    "hungary": "hungary",
    "romania": "romania",
    "bulgaria": "bulgaria",
    "greece": "greece",
    "croatia": "croatia",
    "serbia": "serbia",
    "slovenia": "slovenia",
    "bosnia": "bosnia_and_herzegovina",
    "bosnia and herzegovina": "bosnia_and_herzegovina",
    "north macedonia": "north_macedonia",
    "macedonia": "north_macedonia",
    "montenegro": "montenegro",
    "albania": "albania",
    "kosovo": "kosovo",
    "ukraine": "ukraine",
    "belarus": "belarus",
    "moldova": "moldova",
    "lithuania": "lithuania",
    "latvia": "latvia",
    "estonia": "estonia",
    "cyprus": "cyprus",
    "malta": "malta",
    "europe": "europe",
    # Americas
    "united states": "united_states",
    "usa": "united_states",
    "us": "united_states",
    "america": "united_states",
    "canada": "canada",
    "mexico": "mexico",
    "brazil": "brazil",
    "argentina": "argentina",
    "chile": "chile",
    "colombia": "colombia",
    "peru": "peru",
    "venezuela": "venezuela",
    "ecuador": "ecuador",
    "bolivia": "bolivia",
    "paraguay": "paraguay",
    "uruguay": "uruguay",
    "cuba": "cuba",
    "jamaica": "jamaica",
    "haiti": "haiti",
    "dominican republic": "dominican_republic",
    "puerto rico": "puerto_rico",
    "costa rica": "costa_rica",
    "panama": "panama",
    "guatemala": "guatemala",
    "honduras": "honduras",
    "el salvador": "el_salvador",
    "nicaragua": "nicaragua",
    "trinidad and tobago": "trinidad_and_tobago",
    # US states (common ones)
    "texas": "united_states",
    "california": "united_states",
    "new york": "united_states",
    "florida": "united_states",
    "illinois": "united_states",
    "pennsylvania": "united_states",
    "ohio": "united_states",
    "georgia": "united_states",
    "virginia": "united_states",
    "massachusetts": "united_states",
    # Asia
    "china": "china",
    "japan": "japan",
    "south korea": "south_korea",
    "korea": "south_korea",
    "north korea": "north_korea",
    "india": "india",
    "pakistan": "pakistan",
    "bangladesh": "bangladesh",
    "sri lanka": "sri_lanka",
    "nepal": "nepal",
    "myanmar": "myanmar",
    "burma": "myanmar",
    "thailand": "thailand",
    "vietnam": "vietnam",
    "cambodia": "cambodia",
    "laos": "laos",
    "malaysia": "malaysia",
    "singapore": "singapore",
    "indonesia": "indonesia",
    "philippines": "philippines",
    "taiwan": "taiwan",
    "mongolia": "mongolia",
    "afghanistan": "afghanistan",
    "asia": "asia",
    # Middle East
    "iran": "iran",
    "iraq": "iraq",
    "saudi arabia": "saudi_arabia",
    "yemen": "yemen",
    "oman": "oman",
    "qatar": "qatar",
    "bahrain": "bahrain",
    "kuwait": "kuwait",
    "united arab emirates": "united_arab_emirates",
    "uae": "united_arab_emirates",
    "jordan": "jordan",
    "lebanon": "lebanon",
    "syria": "syria",
    "israel": "israel",
    "palestine": "palestine",
    "turkey": "turkey",
    "middle east": "middle_east",
    # Africa
    "egypt": "egypt",
    "south africa": "south_africa",
    "nigeria": "nigeria",
    "kenya": "kenya",
    "ethiopia": "ethiopia",
    "tanzania": "tanzania",
    "uganda": "uganda",
    "ghana": "ghana",
    "morocco": "morocco",
    "algeria": "algeria",
    "tunisia": "tunisia",
    "libya": "libya",
    "sudan": "sudan",
    "democratic republic of the congo": "dr_congo",
    "congo": "congo",
    "cameroon": "cameroon",
    "ivory coast": "ivory_coast",
    "senegal": "senegal",
    "zimbabwe": "zimbabwe",
    "mozambique": "mozambique",
    "madagascar": "madagascar",
    "angola": "angola",
    "somalia": "somalia",
    "africa": "africa",
    # Oceania
    "australia": "australia",
    "new zealand": "new_zealand",
    "papua new guinea": "papua_new_guinea",
    "fiji": "fiji",
    "oceania": "oceania",
    # Russia / Central Asia
    "russia": "russia",
    "soviet union": "russia",
    "ussr": "russia",
    "kazakhstan": "kazakhstan",
    "uzbekistan": "uzbekistan",
    "turkmenistan": "turkmenistan",
    "kyrgyzstan": "kyrgyzstan",
    "tajikistan": "tajikistan",
    "georgia": "georgia",
    "armenia": "armenia",
    "azerbaijan": "azerbaijan",
}


# ---------------------------------------------------------------------------
# Domain seed data: ontology type -> default domain label
# ---------------------------------------------------------------------------
DOMAIN_SEEDS: Dict[str, str] = {
    B.PERSON: "biography",  # person
    B.ORGANIZATION: "institutional",  # organization
    B.PROCESS: "historical",  # event
    B.GEOSPATIAL_REGION: "geographic",  # location
    B.GEOSPATIAL_LOCATION: "geographic",  # place
    B.ACT: "narrative",  # action
    B.MATERIAL_ARTIFACT: "descriptive",  # object
    B.STASIS: "descriptive",  # state
    B.GROUP_OF_PERSONS: "institutional",  # group
    B.TEMPORAL_REGION: "historical",  # time
}

# NER-type overrides: if these NER labels are present, override domain
DOMAIN_NER_OVERRIDES: Dict[str, str] = {
    "EVENT": "historical",
    "WORK_OF_ART": "cultural",
    "LAW": "legal",
    "LANGUAGE": "cultural",
}

# Keyword patterns in ORG text that refine domain when ontology=organization
DOMAIN_ORG_KEYWORDS: Dict[str, str] = {
    "academic": "university institute college school academy faculty",
    "military": "army navy regiment battalion military defense air force",
    "scientific": "laboratory research observatory",
}


# ---------------------------------------------------------------------------
# Institutional seed data: keyword -> institutional type
# ---------------------------------------------------------------------------
INSTITUTIONAL_KEYWORDS: Dict[str, str] = {
    "academic": "university college institute school academy faculty",
    "military": "army navy regiment battalion military defense air force marine corps",
    "government": "government parliament congress senate ministry department cabinet",
    "religious": "church mosque temple diocese cathedral synagogue monastery",
    "sports": "fc united club team league athletic",
    "media": "newspaper magazine broadcast radio television network press",
}

# Default for unmatched ORG entities
INSTITUTIONAL_DEFAULT = "commercial"


# ---------------------------------------------------------------------------
# Domain keyword fallback: when NER and ontology produce no domain,
# scan text tokens against these dictionaries.  Require >= 2 matches
# from the same domain to activate (avoids single-word false positives).
# ---------------------------------------------------------------------------
DOMAIN_KEYWORD_PATTERNS: Dict[str, List[str]] = {
    "political": ["president", "senator", "congress", "parliament", "election",
                   "vote", "govern", "political", "minister", "chancellor",
                   "democracy", "republican", "democrat", "legislature"],
    "military": ["war", "battle", "army", "navy", "soldier", "general",
                 "military", "combat", "regiment", "colonel", "siege",
                 "campaign", "invasion", "troop"],
    "scientific": ["science", "research", "experiment", "theory", "physics",
                   "chemistry", "biology", "mathematical", "hypothesis",
                   "laboratory", "discovery", "scientific", "quantum"],
    "cultural": ["art", "music", "literature", "film", "theater", "culture",
                 "painting", "sculpture", "novel", "poem", "dance", "opera",
                 "gallery", "artistic"],
    "medical": ["medical", "health", "disease", "hospital", "doctor",
                "treatment", "surgery", "diagnosis", "clinical", "patient",
                "medicine", "pharmaceutical"],
    "legal": ["law", "court", "judge", "legal", "statute", "constitution",
              "trial", "attorney", "legislation", "verdict", "criminal",
              "civil", "justice"],
    "religious": ["church", "religion", "faith", "god", "temple", "monastery",
                  "priest", "bishop", "sacred", "divine", "spiritual",
                  "theological", "worship"],
    "economic": ["economy", "trade", "market", "bank", "finance", "commerce",
                 "currency", "inflation", "stock", "investment", "fiscal",
                 "monetary"],
    "technological": ["technology", "computer", "software", "internet",
                      "digital", "engineering", "invention", "patent",
                      "algorithm", "electronic", "mechanical"],
    "educational": ["education", "school", "university", "teacher", "student",
                    "academic", "curriculum", "professor", "lecture",
                    "scholarship"],
}

# Temporal keyword patterns for era/period references (when NER finds no DATE)
TEMPORAL_KEYWORD_PATTERNS: Dict[str, str] = {
    "ancient": "antiquity",
    "medieval": "medieval_era",
    "renaissance": "renaissance_era",
    "colonial": "colonial_era",
    "victorian": "victorian_era",
    "modern": "modern_era",
    "contemporary": "contemporary_era",
    "prehistoric": "prehistoric_era",
    "postwar": "postwar_era",
    "prewar": "prewar_era",
    "industrial": "industrial_era",
    "enlightenment": "enlightenment_era",
}


# Every BFO/CCO class a table above is keyed by. The ontology head emits any
# certified class; ``_get_ontology_type`` resolves it to the nearest of these
# (itself or an ancestor, never inheriting from bfo:entity), so a subclass such
# as City uses its Geospatial Region entries.
ONTOLOGY_KEYS = frozenset().union(DOMAIN_SEEDS)
