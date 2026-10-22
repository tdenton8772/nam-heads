# Attribute Head

Answers: **"What does the text say each entity IS or HAS?"** -- nationality,
role and occupation nouns, kind nouns and descriptors predicated of a subject.

## Behavior (predication mode, the default)
- Deterministic, lexical; no embeddings. Runs on the rule NLP backend's
  ParsedDocument (`ctx["spacy_doc"]`).
- `predication.py` reads predications by token order, POS and bracket depth
  (the rule backend's dependency arcs are unreliable for these):
  copular (`X (born ...) is an American actor, comedian`; also become /
  remain, and `'s` after a pronoun only: `Gary's impact` is a possessive),
  relative (`X, who was a naturalist`, `... a leader who became the second
  president`), appositive (`Lincoln, the president,`, `Acapulco, a major
  seaport city,`, `Zhang Sixun, a Chinese astronomer and mechanical
  engineer`), title (`the physicist Richard Feynman`, `King Charles`,
  `creators Len Wein, Marv Wolfman, and John Warner`), as-role (`served as
  Minister`, `known as`, `established itself as`, `As mayor of X, he ...`),
  kind-of (`the town of Lostwithiel`), kinds-such-as (`genera like
  Leidyosuchus`), pre-nominal modifier (`large snakes`).
- The main subject is the record title's mention in the text
  (`ctx["title"]`; ingest prefixes the title; the qualifier in
  `Yves Rocher (company)` is dropped and kept as evidence), else the first
  copular subject. he/she/it/they/this and kind-noun anaphors resolve to
  it: `This airfield was ...`, `fashioned this unusually tall manuscript`,
  `The company expanded` (the anaphor's noun and its adjectives become
  values of the main subject). Whether a resolution holds is decided at
  settle time from the store (`resolution_holds`): he/she not for a subject
  known not to be a person, it/this not for a person; an anaphor's noun not
  when the store marks it a non-kind (`this story`), and naming a kind of
  person exactly when the subject is one; a definite `the <noun>` only when
  the text ties the noun to the subject (title word, qualifier, an earlier
  predicate or demonstrative) or the store knows it as a kind. A rejected
  resolution leaves the predication anchored on the pronoun / noun phrase
  itself (the bundler's topic carry-over may still bind a pronoun). The
  main subject's person-ness: the qualifier, else what the text predicates
  of a title mention, else a PERSON-labelled title plus he/she in the text.
- Parser defects worked around in the head (the NLP output is pinned by the
  ontology codebook): curly `She’s` (three tokens), a 3sg verb tagged NOUN
  after an anaphor (`This family exemplifies`), participles tagged VERB
  opening an NP (`a landlocked country` -> `landlocked`), the lemmatizer on
  JJ adjectives and -ing nouns (`challenging`, `meeting`, not `challenge`,
  `meet`).
- The value is the word's own normalised lemma (synonyms folded via
  `config.synonyms`). Values are NOT mapped onto ontology classes: the axis
  must keep the text's cardinality, or it collapses into a copy of the
  ontology segment.
- Junk filtering: ordinals, determiner-like and vague evaluative adjectives,
  generic heads (`name`, `part`, `life`, `piece`, ...), proper names and
  capitalised non-role heads (`known as the Heart of Asia`; `was Duke of
  Modena` keeps `duke`), name-internal capitalised modifiers (`Central
  America`, `a Royal Air Force station`), directional parts of places
  (`southern Norway`), section headings (`External links`), adjectives with
  a to-complement (`central to`), Latin `de facto`, possessors inside a
  predicate (`Angola's national military`). Title and fronted-as heads must
  name a kind of person (the store's role flag, seeded from
  `person_nouns.txt`, generated offline from WordNet by
  `scripts/gen_person_nouns.py`); kind-of heads must not.
- The legacy extraction (`_extract_from_doc`: ADJ/amod/acomp/participial/
  compound/appositive/acl) still runs on every text with its record-level
  selection (`max_attributes`, 3) and `_auto_classify`; its values gain
  anchors from token adjacency (unless structurally junk) and are merged
  with the predication values. A legacy value with no such anchor is
  emitted but never bound to an entity (the bundler binds anchored values
  only). Predication values (copular, relative,
  appositive, title, as-role, pronoun, query) are scored with the legacy
  scoring per reading -- config link weight, ontology-aware stopword penalty,
  the category (ontology-scoped row first, then global; curated rows only)
  under `ONTOLOGY_RELEVANCE` (a 0.8-1.3 nudge, incl. evidence categories
  role / kind / nationality), NER relevance -- then the legacy adaptive
  selection per subject and sentence (`max_attributes_per_subject`, 8), then
  `max_predicated_per_record` (16). min_len and the wordlist apply.
- `EVIDENCE_EXCLUSIONS` is an explicit hard filter (role nouns are not
  attributes of non-agents; non-person kind nouns are not attributes of a
  person). Switch: `evidence_exclusions` / `NAM_ATTRIBUTE_EVIDENCE_EXCLUSIONS=0`
  (head), `NAM_ATTRIBUTE_CONDITIONING=0` (bundler).
- The steering ontology is each sentence's (`sentence_ontologies`; the query path calls
  `encoders/sentence_ontology.attach_sentence_ontologies`), else the record's;
  `OTHER` falls back to the record ontology.
  Tables resolve through `nam_heads/ontology/lookup.py` (one resolver, shared with
  the affordance head).
- Every selected value carries `metadata.anchors` (subject token spans) and
  `metadata.kinds`. The bundler links a value to an entity when one of the
  entity's mentions is the anchor's subject (`_anchor_binds`): equal to it,
  covering it, crossing its edge, or its head (`Congress` of `United States
  Congress`) -- not a mention that modifies a longer name (`Brigham Young`
  in `Brigham Young University`), except the entity head's partial reading
  of the title (a capitalised name found only inside title mentions, not
  after the title's `of`). A value that is a word of the entity's own
  mention is dropped (`Salt Lake City` -> salt). The dependency walk runs
  only for a head that emits no anchors (legacy mode). Then it refines by
  the entity's ontology through the same table, orders by head emission
  order and keeps at most `NAM_ATTRIBUTE_MAX_PER_BUNDLE` (8).
- Store (`acat::` rows): seeded with `person_nouns.txt` (role=True) and
  `abstract_nouns.txt` (kind=False: discourse, abstract and relational
  nouns -- `story`, `impact`, `heart`, `capital`); every kept reading is
  observed (kinds, ontologies, classes). A noun is learned as a role after
  `learning.ROLE_SUPPORT` (3) distinct records with a person cue, and as a
  kind (kind=True, overriding the seed) after `learning.KIND_SUPPORT` (3)
  distinct records predicating it of a named subject without a relational
  complement (`Kucura is a village`; not `the capital of Y`). Extraction
  reads the flag through `learning.kind_status`. Changed rows go through
  the LMDB relay. Queries read, never write.
- Query side: `encode_query` also reads the canonicalized query's leading NP
  as a predicate; the coordinator drops canonicalizer-injected descriptors
  from attribute results unless typed.
- No evidence -> a single `__null__`.

## Switches
| Setting | Env | Default |
|---|---|---|
| `predication_mode` | `NAM_ATTRIBUTE_PREDICATION=0` -> legacy ADJ extraction, capped at `max_attributes` | on |
| `sentence_steering` | `NAM_ATTRIBUTE_SENTENCE_STEER=0` -> steer by the record ontology | on |
| bundler conditioning | `NAM_ATTRIBUTE_CONDITIONING=0` | on |
| bundle bound | `NAM_ATTRIBUTE_MAX_PER_BUNDLE` | 8 |

## Output
- `concept`: lowercase value, e.g. `american`, `actor`, `large`
- `confidence`: by link (copular 0.9 ... modifier 0.6)
- `metadata`: `{method: <link>, anchors: [[start, end], ...], kinds, category?}`
