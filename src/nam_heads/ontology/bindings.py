"""Named BFO 2020 / CCO v2.2 classes referenced in code.

Constants for the classes code names directly (the unclassified root, the
BFO branches, the classes tables are keyed by). Every constant must exist in
``classes.json``; ``registry`` refuses to load otherwise.
"""

from typing import Dict

# CURIE prefixes. BFO uses the OBO convention, CCO its own `cco:` prefix.
BFO_IRI = "http://purl.obolibrary.org/obo/BFO_"
CCO_IRI = "https://www.commoncoreontologies.org/"
PREFIXES = {"BFO": BFO_IRI, "cco": CCO_IRI}

# ---- BFO 2020 (ISO/IEC 21838-2) ------------------------------------------
ENTITY = "BFO:0000001"
CONTINUANT = "BFO:0000002"
OCCURRENT = "BFO:0000003"
INDEPENDENT_CONTINUANT = "BFO:0000004"
SPECIFICALLY_DEPENDENT_CONTINUANT = "BFO:0000020"
GENERICALLY_DEPENDENT_CONTINUANT = "BFO:0000031"
PROCESS = "BFO:0000015"
TEMPORAL_REGION = "BFO:0000008"
QUALITY = "BFO:0000019"
RELATIONAL_QUALITY = "BFO:0000145"

# ---- CCO v2.2 -------------------------------------------------------------
PERSON = "cco:ont00001262"
GROUP_OF_PERSONS = "cco:ont00000914"
ORGANIZATION = "cco:ont00001180"
MATERIAL_ARTIFACT = "cco:ont00000995"
FACILITY = "cco:ont00000192"
GEOSPATIAL_REGION = "cco:ont00000472"
GEOSPATIAL_LOCATION = "cco:ont00000487"
ACT = "cco:ont00000005"
CHANGE = "cco:ont00000004"
STASIS = "cco:ont00000819"
CAUSE = "cco:ont00000978"
EFFECT = "cco:ont00000660"
ACT_OF_OWNERSHIP = "cco:ont00001336"
ACT_OF_REQUESTING = "cco:ont00000769"
INFORMATION_CONTENT_ENTITY = "cco:ont00000958"
DESIGNATIVE_ICE = "cco:ont00000686"
DESCRIPTIVE_ICE = "cco:ont00000853"
PRESCRIPTIVE_ICE = "cco:ont00000965"
MEASUREMENT_ICE = "cco:ont00001163"
ORDINAL_MEASUREMENT_ICE = "cco:ont00001010"
DOCUMENT_CONTENT_ENTITY = "cco:ont00002039"
OBJECTIVE = "cco:ont00000476"
NATURAL_LANGUAGE = "cco:ont00000496"

# The class written when the ontology head abstains. `entity` is the BFO
# root: it asserts only that the thing exists, which is all that is known.
UNCLASSIFIED = ENTITY


def named_classes() -> set:
    """Every class constant defined in this module."""
    return {v for k, v in globals().items()
            if k.isupper() and isinstance(v, str) and ":" in v and not v.startswith("http")}
