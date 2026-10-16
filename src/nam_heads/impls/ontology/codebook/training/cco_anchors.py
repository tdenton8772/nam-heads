"""Wikidata anchor classes -> BFO/CCO classes, for the codebook's training labels.

A record's subject is labelled by walking its Wikidata "instance of" values up
"subclass of" to the NEAREST anchor below (breadth-first); at equal distance the
earlier entry wins, so specific anchors precede the generic catch-alls at the
end. Every QID's English label is checked against the fetched Wikidata class
file by ``tests/test_cco_anchors.py`` -- a mistyped QID fails the test.

Judgment calls (BFO is realist; CCO models particulars):
- A taxon article is about the organisms of that kind -> cco:Organism.
- Chemical kinds / genes / proteins -> bfo:material entity (CCO has no class).
- Fictional characters are content, not persons -> Information Content Entity.
- A car model is typed by what it models -> cco:Automobile.
- Disciplines, awards, games, software -> Information Content Entity.
- Settlements without an administrative status -> cco:Geospatial Region.
"""

from typing import Dict, FrozenSet, List, Tuple

# ---- not entities: never labelled, never trained ---------------------------
EXCLUDE: FrozenSet[str] = frozenset({
    "Q4167410",    # Wikimedia disambiguation page
    "Q22808320",   # Wikimedia human name disambiguation page
    "Q4167836",    # Wikimedia category
    "Q11266439",   # Wikimedia template
    "Q17442446",   # Wikimedia internal item
})

# (Wikidata QID, its English label, BFO/CCO class)
ANCHOR_ROWS: List[Tuple[str, str, str]] = [
    # ---- persons and groups --------------------------------------------
    ("Q15632617", "fictional human", "cco:ont00000958"),  # Information Content Entity
    ("Q95074", "character", "cco:ont00000958"),  # Information Content Entity
    ("Q5", "human", "cco:ont00001262"),  # Person
    ("Q41710", "ethnic group", "cco:ont00000507"),  # Ethnic Group
    ("Q8436", "family", "cco:ont00000837"),  # Family
    ("Q215380", "musical group", "cco:ont00000914"),  # Group of Persons
    ("Q26213387", "Olympic delegation", "cco:ont00000914"),  # Group of Persons
    ("Q16334295", "group of humans", "cco:ont00000914"),  # Group of Persons
    # ---- organizations -------------------------------------------------
    ("Q176799", "military unit", "cco:ont00000274"),  # Armed Force
    ("Q327333", "government agency", "cco:ont00000408"),  # Government Organization
    ("Q3918", "university", "cco:ont00000564"),  # Educational Organization
    ("Q9826", "high school", "cco:ont00000564"),  # Educational Organization
    ("Q3914", "school", "cco:ont00000564"),  # Educational Organization
    ("Q2385804", "educational institution", "cco:ont00000564"),  # Educational Organization
    ("Q7278", "political party", "cco:ont00001302"),  # Civil Organization
    ("Q163740", "nonprofit organization", "cco:ont00001302"),  # Civil Organization
    ("Q4830453", "business", "cco:ont00000443"),  # Commercial Organization
    ("Q476028", "association football club", "cco:ont00001180"),  # Organization
    ("Q847017", "sports club", "cco:ont00001180"),  # Organization
    ("Q14350", "radio station", "cco:ont00001180"),  # Organization
    ("Q1616075", "television station", "cco:ont00001180"),  # Organization
    # ---- administrative and geographic places ---------------------------
    ("Q5107", "continent", "cco:ont00000251"),  # Continent
    ("Q6256", "country", "cco:ont00000139"),  # Domain of a Country
    ("Q3624078", "sovereign state", "cco:ont00000139"),  # Domain of a Country
    ("Q35657", "U.S. state", "cco:ont00000126"),  # First-Order Administrative Region
    ("Q107390", "federated state", "cco:ont00000934"),  # Constituent State
    ("Q34876", "province", "cco:ont00000046"),  # Province
    ("Q28575", "county", "cco:ont00001164"),  # County
    ("Q515", "city", "cco:ont00000887"),  # City
    ("Q3957", "town", "cco:ont00000071"),  # Town
    ("Q532", "village", "cco:ont00000718"),  # Village
    ("Q15284", "municipality", "cco:ont00000460"),  # Local Administrative Region
    ("Q1115575", "civil parish", "cco:ont00000460"),  # Local Administrative Region
    ("Q56061", "administrative territorial entity", "cco:ont00001152"),  # Government Domain
    ("Q123705", "neighborhood", "cco:ont00000487"),  # Geospatial Location
    ("Q839954", "archaeological site", "cco:ont00000487"),  # Geospatial Location
    ("Q486972", "human settlement", "cco:ont00000472"),  # Geospatial Region
    ("Q17343829", "unincorporated community", "cco:ont00000472"),  # Geospatial Region
    ("Q4022", "river", "cco:ont00000563"),  # Hydrographic Feature
    ("Q23397", "lake", "cco:ont00000563"),  # Hydrographic Feature
    ("Q8502", "mountain", "cco:ont00000873"),  # Physiographic Feature
    ("Q23442", "island", "cco:ont00000873"),  # Physiographic Feature
    ("Q22698", "park", "cco:ont00000400"),  # Park
    ("Q618123", "geographical feature", "cco:ont00001081"),  # Geographic Feature
    # ---- facilities and artifacts --------------------------------------
    ("Q1248784", "airport", "cco:ont00001078"),  # Airport
    ("Q55488", "railway station", "cco:ont00000226"),  # Transportation Facility
    ("Q34442", "road", "cco:ont00000247"),  # Road
    ("Q12280", "bridge", "cco:ont00001099"),  # Bridge
    ("Q483110", "stadium", "cco:ont00000641"),  # Entertainment Facility
    ("Q16970", "church building", "cco:ont00000032"),  # Church
    ("Q41176", "building", "cco:ont00000192"),  # Facility
    ("Q11446", "ship", "cco:ont00000549"),  # Water Transportation Artifact
    ("Q11436", "aircraft", "cco:ont00001043"),  # Aircraft
    ("Q3231690", "car model", "cco:ont00001011"),  # Automobile
    ("Q728", "weapon", "cco:ont00000445"),  # Weapon
    # ---- information content -------------------------------------------
    ("Q11424", "film", "cco:ont00002009"),  # Video
    ("Q5398426", "television series", "cco:ont00002009"),  # Video
    ("Q21191270", "television series episode", "cco:ont00002009"),  # Video
    ("Q571", "book", "cco:ont00002040"),  # Book
    ("Q7725634", "literary work", "cco:ont00002039"),  # Document Content Entity
    ("Q47461344", "written work", "cco:ont00002039"),  # Document Content Entity
    ("Q1002697", "periodical", "cco:ont00002039"),  # Document Content Entity
    ("Q34770", "language", "cco:ont00000496"),  # Natural Language
    ("Q9174", "religion", "cco:ont00000616"),  # Religion
    ("Q131569", "treaty", "cco:ont00000965"),  # Prescriptive Information Content Entity
    ("Q29883540", "International Labour Organization Convention", "cco:ont00000965"),  # Prescriptive Information Content Entity
    ("Q101352", "family name", "cco:ont00001014"),  # Proper Name
    ("Q202444", "given name", "cco:ont00001014"),  # Proper Name
    ("Q482994", "album", "cco:ont00000958"),  # Information Content Entity
    ("Q134556", "single", "cco:ont00000958"),  # Information Content Entity
    ("Q105543609", "musical work/composition", "cco:ont00000958"),  # Information Content Entity
    ("Q7889", "video game", "cco:ont00000958"),  # Information Content Entity
    ("Q618779", "award", "cco:ont00000958"),  # Information Content Entity
    ("Q38033430", "class of award", "cco:ont00000958"),  # Information Content Entity
    ("Q11862829", "academic discipline", "cco:ont00000958"),  # Information Content Entity
    # ---- acts, processes, time -------------------------------------------
    ("Q178561", "battle", "cco:ont00000109"),  # Act of Military Force
    ("Q198", "war", "cco:ont00000109"),  # Act of Military Force
    ("Q40231", "public election", "cco:ont00000265"),  # Election
    ("Q4504495", "award ceremony", "cco:ont00000455"),  # Act of Ceremony
    ("Q27020041", "sports season", "cco:ont00000228"),  # Planned Act
    ("Q8065", "natural disaster", "cco:ont00000007"),  # Natural Process
    ("Q14795564", "point in time with respect to recurrent timeframe", "cco:ont00000921"),  # Calendar Day
    ("Q3186692", "calendar year", "cco:ont00000674"),  # Calendar Year
    ("Q39911", "decade", "cco:ont00001088"),  # Decade
    ("Q578", "century", "cco:ont00001206"),  # Multi-Year Temporal Interval
    ("Q18340514", "events in a specific year or time period", "BFO:0000015"),  # process
    # ---- biology and chemistry -------------------------------------------
    ("Q16521", "taxon", "cco:ont00000551"),  # Organism
    ("Q729", "animal", "cco:ont00000562"),  # Animal
    ("Q756", "plant", "cco:ont00000871"),  # Plant
    ("Q12136", "disease", "cco:ont00000318"),  # Disease
    ("Q113145171", "type of chemical entity", "BFO:0000040"),  # material entity
    ("Q11173", "chemical compound", "BFO:0000040"),  # material entity
    ("Q11344", "chemical element", "BFO:0000040"),  # material entity
    # ---- kinds modelled as "subclass of" and other measured gaps ---------
    ("Q112193867", "type of disease", "cco:ont00000318"),  # Disease
    ("Q65943", "theorem", "cco:ont00000853"),  # Descriptive Information Content Entity
    # List and index pages describe a set (List of Polydor Records artists,
    # county NRHP listings, discographies, timelines). Excluded, they were
    # never trained and landed in bfo:entity (live census 2026-10-04).
    ("Q13406463", "Wikimedia list article", "cco:ont00000853"),  # Descriptive ICE
    ("Q273057", "discography", "cco:ont00000853"),  # Descriptive ICE
    ("Q186117", "timeline", "cco:ont00000853"),  # Descriptive ICE
    ("Q24034552", "mathematical concept", "cco:ont00000958"),  # Information Content Entity
    ("Q188451", "music genre", "cco:ont00000958"),  # Information Content Entity
    ("Q223393", "literary genre", "cco:ont00000958"),  # Information Content Entity
    ("Q2135465", "legal term or legal concept", "cco:ont00000958"),  # Information Content Entity
    ("Q28108", "political system", "cco:ont00000958"),  # Information Content Entity
    ("Q1799072", "method", "cco:ont00000965"),  # Prescriptive Information Content Entity
    ("Q25295", "language family", "cco:ont00001175"),  # Language
    ("Q33384", "dialect", "cco:ont00000496"),  # Natural Language
    ("Q4164871", "position", "BFO:0000023"),  # role
    ("Q294414", "public office", "BFO:0000023"),  # role
    ("Q523", "star", "BFO:0000030"),  # object
    ("Q11416914", "periodic comet", "BFO:0000030"),  # object
    ("Q39367", "dog breed", "cco:ont00000562"),  # Animal
    ("Q12089225", "mineral species", "BFO:0000040"),  # material entity
    ("Q112826905", "class of anatomical entity", "BFO:0000040"),  # material entity
    ("Q417841", "protein family", "BFO:0000040"),  # material entity
    ("Q124078422", "weapon type", "cco:ont00000445"),  # Weapon
    ("Q110295396", "type of musical instrument", "cco:ont00000995"),  # Material Artifact
    ("Q113813711", "coin type", "cco:ont00000995"),  # Material Artifact
    ("Q13073773", "Catholic Church of an area", "cco:ont00001180"),  # Organization
    ("Q12075477", "Islam of an area", "cco:ont00000616"),  # Religion
    ("Q66374263", "religion of an area", "cco:ont00000616"),  # Religion
    # ---- generic catch-alls (last: only win when nothing nearer matches) ----
    ("Q43229", "organization", "cco:ont00001180"),  # Organization
    ("Q2221906", "geographic location", "cco:ont00000487"),  # Geospatial Location
    ("Q17537576", "creative work", "cco:ont00000958"),  # Information Content Entity
    ("Q1656682", "planned event", "cco:ont00000228"),  # Planned Act
    ("Q1190554", "occurrence", "BFO:0000015"),  # process
    ("Q7239", "organism", "cco:ont00000551"),  # Organism
    ("Q16686448", "artificial object", "cco:ont00000995"),  # Material Artifact
]

ANCHORS: Dict[str, str] = {q: c for q, _, c in ANCHOR_ROWS}
