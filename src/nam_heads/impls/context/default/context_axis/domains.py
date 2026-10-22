"""Domain context: the field of activity a sentence is in, from its own words.

Value: a field name ("association_football", "film", "christianity",
"lepidoptera"), at the granularity of a library subject heading -- fine
enough that no field dominates an encyclopedic corpus. A field is emitted
only for a sentence containing one of its trigger words; never from the
entity's type and never as a default.

Trigger words are unambiguous nouns/adjectives of their field; words with a
common other sense ("line", "play", "star", "founded", "party", "mine") are
excluded. Triggers are matched on the lowercased token text, so the rule lemmatizer's
un-doubling ("called" -> "cal") cannot hide them; a plural matches its
singular's trigger ("architects", "universities"), and a few two-word
triggers name what neither word does ("board game", "American football").

Senses:
* a trigger inside a named period or place is part of that name ("Song
  dynasty" is no music, "Church Stretton" no church) -- ``claimed``;
* a capitalized religious / Olympic adjective heading a longer proper name
  that is not an institution of its field is that name ("Hindu Kush",
  "Christian Bale"), not the field;
* "musical" before a noun is music ("musical interests") unless the noun
  names a field itself ("musical theater"); alone it is the stage work
  (theatre);
* a capitalized trigger that is a person's given name -- before the last
  token of a PERSON span, or a known first name before a capitalized word --
  is that name ("Gene Cernan");
* "football" is one sport per document: American football when the document
  has an American-football trigger, association football otherwise.
"""

from typing import Dict, List, Optional, Set, Tuple

FIELD_TRIGGERS: Dict[str, str] = {
    "association_football": "footballer footballers soccer midfielder goalkeeper fifa uefa fc f.c.",
    "american_football": "quarterback touchdown nfl linebacker",
    # resolved per document to association_football / american_football
    "football": "football",
    "baseball": "baseball pitcher mlb inning innings shortstop",
    "basketball": "basketball nba",
    "ice_hockey": "hockey nhl",
    "cricket": "cricket cricketer batsman wicket wickets bowler",
    "rugby": "rugby",
    "tennis": "tennis wimbledon",
    "golf": "golf golfer pga",
    "motorsport": "racing driver nascar prix motorsport",
    "cycling": "cyclist cycling",
    "athletics": "athlete sprinter marathon athletics",
    "combat_sport": "boxer boxing wrestler wrestling ufc martial",
    "skating": "skating skater",
    "skiing": "skiing skier",
    "biathlon": "biathlon biathlete",
    "olympic_games": "olympics olympic",
    "music": "album albums song songs singer songwriter band musician guitarist drummer bassist rapper "
             "composer music singing vocal vocals choir choral concert punk jazz hip-hop guitar",
    "classical_music": "symphony orchestra concerto opera",
    "film": "film films movie screenplay actor actress starring filmmaker cinema screenwriter",
    "television": "television episode episodes sitcom tv",
    "theatre": "theatre theater playwright broadway musical",
    "literature": "novel novels novelist poet poem poems poetry fiction literary writer verse",
    "comics": "comics superhero manga cartoonist",
    "games": "tabletop role-playing rpg boardgame wargame dice gaming",
    "journalism": "newspaper journalist magazine",
    "video_games": "gameplay playstation xbox videogame",
    "visual_art": "painter painting paintings sculptor sculpture artist pottery ceramics potter "
                  "printmaking watercolour watercolor illustrator",
    "architecture": "architect architecture architectural",
    "politics": "politician political parliament senator governor mayor congressman legislature "
                "politics parliamentary legislative diplomat diplomacy diplomatic",
    "elections": "election elected elections electoral constituency candidate vote votes",
    "law": "court judge lawyer law legal attorney supreme",
    "crime": "murder crime criminal convicted prison sentenced arrested homicide cartel gang "
             "trafficking smuggling fraud robbery kidnapping terrorism",
    "military": "war battle army military soldier soldiers regiment navy naval troops infantry "
                "warfare artillery cavalry squadron battalion garrison",
    "christianity": "church bishop priest diocese christian catholic saint parish cathedral monastery "
                    "christianity mormon evangelical baptist anglican protestant methodist lutheran "
                    "presbyterian",
    "islam": "islam islamic muslim mosque quran imam",
    "judaism": "jewish rabbi synagogue judaism torah",
    "buddhism_hinduism": "buddhist buddhism hindu temple hinduism",
    "philosophy": "philosopher philosophy philosophical",
    "linguistics": "linguistics linguist grammar dialect phonology phonetics",
    "psychology": "psychology psychologist psychological psychiatry psychiatrist psychiatric phobia",
    "education": "school schools university college student students professor teacher",
    "business": "company business corporation brand products corporate ceo entrepreneur shareholder "
                "retailer retail",
    "economics": "economics economist economy inflation macroeconomics microeconomics econometrics "
                 "recession gdp",
    "finance": "bank banking finance investment",
    "agriculture": "farm farming agriculture agricultural crop",
    "mining_energy": "mining coal oil",
    "aviation": "aircraft airline airport flight pilot aviation aviator",
    "rail_transport": "railway railroad train trains",
    "road_transport": "highway",
    "shipping": "ship ships vessel harbour",
    "spaceflight": "spacecraft satellite nasa orbit astronaut cosmonaut spaceflight",
    "computing": "software computer internet programming",
    "mathematics": "mathematician mathematics theorem mathematical topology topological algebra "
                   "algebraic geometry calculus",
    "physics_chemistry": "physicist physics chemist chemistry",
    "astronomy": "astronomer galaxy planet asteroid astronomical telescope astronomy astrophysics cosmology",
    "climatology": "climate climatology climatic meteorology meteorological",
    "medicine": "disease medical hospital physician patients treatment tumor tumour cancer surgical "
                "surgeon dental dentistry clinical symptom therapy pathology lesion",
    "genetics": "gene genetic genetics allele chromosome dna genome",
    "lepidoptera": "moth moths butterfly",
    "entomology": "beetle insect",
    "botany": "flowering shrub flora botanical botanist botany angiosperm cultivar",
    "biology": "genus species taxon biology biologist",
    "zoology": "mammal mammals bird birds fish snail fauna zoologist zoology reptile amphibian rodent "
               "primate",
}
# Two-word triggers (lowercased "first second"): a field neither word names.
BIGRAM_TRIGGERS: Dict[str, str] = {
    "board game": "games", "board games": "games", "card game": "games", "card games": "games",
    "video game": "video_games", "video games": "video_games",
    "computer game": "video_games", "computer games": "video_games",
    "comic book": "comics", "comic books": "comics",
    "american football": "american_football", "association football": "association_football",
    "rugby football": "rugby", "gaelic football": "gaelic_football",
    "speed skating": "speed_skating", "figure skating": "figure_skating",
    "field hockey": "field_hockey", "ice hockey": "ice_hockey",
    "armed forces": "military", "air force": "military",
    "space exploration": "spaceflight",
}
TRIGGERS: Dict[str, str] = {}
for field, words in FIELD_TRIGGERS.items():
    for w in words.split():
        TRIGGERS.setdefault(w, field)
for w, field in BIGRAM_TRIGGERS.items():
    TRIGGERS.setdefault(w, field)

# Capitalized adjectives of a religion or the Olympics: heading a longer
# proper name that is not an institution of the field, they are that name.
PROPER_ADJECTIVES = frozenset({"hindu", "christian", "catholic", "jewish", "islamic", "muslim",
                               "buddhist", "olympic", "anglican", "protestant", "baptist",
                               "methodist", "lutheran", "presbyterian", "evangelical", "mormon"})
# "musical" before a noun is music; alone it names a stage work (its trigger).
ADJECTIVE_SENSES = {"musical": "music"}
_NOUNISH = frozenset({"NOUN", "PROPN", "ADJ"})


# Singular triggers whose plural is no evidence of the field: "artists"
# covers musicians and performers as often as painters.
NO_PLURAL = frozenset({"artist"})


def singular(w: str) -> Optional[str]:
    """The singular a plural token may be of ("architects", "churches",
    "universities"); None for a word that cannot be a plural."""
    if len(w) < 4 or not w.endswith("s") or w.endswith("ss") or w[:-1] in NO_PLURAL:
        return None
    if w.endswith("ies"):
        return w[:-3] + "y"
    return w[:-1]


def field_of(fields, w: str) -> Optional[str]:
    """The field token ``w`` (lowercased) triggers: itself, else its singular."""
    f = fields.get(w)
    if f is None:
        sg = singular(w)
        if sg is not None:
            f = fields.get(sg)
            if f is None and w.endswith("es"):
                f = fields.get(w[:-2])
    return f


def football_sense(toks, fields) -> str:
    """The sport "football" means in this document."""
    n = len(toks)
    for i in range(n):
        w = toks[i].text.lower()
        if fields.get(w) == "american_football":
            return "american_football"
        if w == "american" and i + 1 < n and toks[i + 1].text.lower() == "football":
            return "american_football"
    return "association_football"


def _first_names():
    from .places import first_names
    return first_names()


def _proper_name(toks, i: int, lex) -> bool:
    """toks[i], a capitalized PROPER_ADJECTIVES word, heads a longer proper
    name that is not an institution of its field ("Hindu Kush")."""
    if i + 1 >= len(toks):
        return False
    nxt = toks[i + 1].text
    if not nxt[:1].isupper():
        return False
    from .institutions import NOT_ORG_TAIL
    return field_of(lex.fields, nxt.lower()) is None and nxt not in lex.org_heads \
        and nxt not in NOT_ORG_TAIL


def extract(toks, sents, lex, ents=(), claimed: Optional[Set[int]] = None) -> List[Tuple[int, int, str, str]]:
    """Field triggers (``lex.fields``), the current head's keyword fields
    (``lex.legacy_fields``, >= 2 distinct keywords of one field per sentence)
    and its NER-label fields (``lex.ner_fields``: an EVENT / WORK_OF_ART /
    LAW / LANGUAGE span is evidence for its sentence). ``claimed``: tokens
    of named periods and places, whose words are no field's."""
    from .legacy import LEGACY_MIN_HITS
    claimed = claimed or set()
    football = None
    # Given-name positions of multi-token PERSON spans (every token but the
    # last: the rule NER also labels "Chinese Buddhism" PERSON, whose last
    # word is no given name).
    person_tokens = {k for sp in ents if sp.label == "PERSON" and sp.end - sp.start >= 2
                     for k in range(sp.start, sp.end - 1)}
    out = []
    for s, e in sents:
        seen = set()
        lhits: Dict[str, set] = {}
        lfirst: Dict[str, int] = {}
        skip = -1
        for i in range(s, e):
            if i < skip or i in claimed:
                continue
            txt = toks[i].text
            w = txt.lower()
            end = i + 1
            f = None
            if i + 1 < e and (i + 1) not in claimed:
                f = lex.fields.get(f"{w} {toks[i + 1].text.lower()}")
                if f:
                    end = i + 2
            if f is None:
                f = field_of(lex.fields, w)
                if f and w in PROPER_ADJECTIVES and txt[:1].isupper() and _proper_name(toks, i, lex):
                    f = None
                if f and txt[:1].isupper() and (i in person_tokens or (
                        i + 1 < len(toks) and toks[i + 1].text[:1].isupper() and w in _first_names())):
                    f = None
                if f and w in ADJECTIVE_SENSES and i + 1 < e and \
                        getattr(toks[i + 1], "pos_", "") in _NOUNISH:
                    # "musical interests" is music; "musical theater" is
                    # what its noun says (the noun's own trigger fires).
                    nxt = field_of(lex.fields, toks[i + 1].text.lower())
                    f = None if nxt else ADJECTIVE_SENSES[w]
            if f == "football":
                if football is None:
                    football = football_sense(toks, lex.fields)
                f = football
            if f and f not in seen:
                seen.add(f)
                out.append((i, end, f, "field_trigger"))
            skip = end
            lf = lex.legacy_fields.get(w)
            if lf:
                lhits.setdefault(lf, set()).add(w)
                lfirst.setdefault(lf, i)
        for lf, words in lhits.items():
            if len(words) >= LEGACY_MIN_HITS and lf not in seen:
                seen.add(lf)
                out.append((lfirst[lf], lfirst[lf] + 1, lf, "legacy_keywords"))
        for sp in ents:
            f = lex.ner_fields.get(sp.label)
            if f and s <= sp.start < e and f not in seen:
                seen.add(f)
                out.append((sp.start, sp.end, f, f"ner_{sp.label.lower()}"))
    return out
