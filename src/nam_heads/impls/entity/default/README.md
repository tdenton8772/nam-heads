# Entity Encoder Head

## Purpose:
Resolve references to specific real-world things into stable, canonical entity identifiers.

This head answers:

“Which exact thing is being referred to?”

Examples:
- "Tyler Denton", "Denton, Tyler", "Tyler E. Denton" → same entity
- "Goulds pump 23", "the big red Goulds pump", "3rd Goulds pump" → same entity
- "Central Park", "the park" → same location entity

## What This Head Does (and Does Not Do)
### Does
- Extract entity candidates from text
- Normalize surface forms into canonical representations
- Resolve aliases and variants into stable entity IDs
- Persist entity mappings in the entity store (`storage/`: session KV behind an LMDB cache in
  production, memory in tests), batch calls only
- Return one semantic address per entity, concept = the entity id (see Entity ID Format)

### Does Not
- Classify ontology (“person”, “object”, etc.) → that’s the Ontology head
- Infer attributes (“red”, “large”) → that’s the Attribute head
- Infer actions or affordances → separate heads
- Attempt fuzzy semantic reasoning (no embeddings here)

This head is deterministic, structural, and referential.

## Design Principles

1. Deterministic
- Same input → same entity ID
- No randomness, no temperature, no LLM calls
2. Self-contained
- All dependencies loaded inside the head
- No global requirements leakage
- Safe to disable without side effects
3. Composable
- Designed to work alongside Ontology, Attribute, Affordance heads
- Does not assume downstream storage or query model
4. Incrementally improving
- Entity resolution improves as more surface forms are observed
- No retraining required

## Pipeline Overview
1. Candidate extraction (`normalization.py`, on the rule NLP parse)
- **Name spans** (`name_spans`): runs of capitalised tokens joined by name connectors
  ("Battle of Lostwithiel", "Leonardo da Vinci", "Bunnies & Burrows"), abbreviation dots
  ("Great Wyrley F.C.", "U.S. Army") and one trailing number or regnal numeral ("Apollo 15",
  "Charles I"). Read from capitalisation, because the rule tagger checks its dictionary before
  case (`rule_backend/pos_tagger.py`, tier 3 before tier 4) and tags "Young" ADJ, "United" VERB.
- **Noun chunks**, cleaned: articles/demonstratives and punctuation skipped, cut at the first
  preposition, the "s" a split curly apostrophe leaves ("Earth ’ s exploration") dropped. The POS
  pattern and token count are read from these content tokens, so "the town" is the noun "town".
- A chunk inside a name span gives way to it; a name span plus lower-case words
  ("entrepreneur Yves Rocher") is the name alone. Words leading into a name whose last word is
  lower case are dropped and the name alone is the candidate ("president Abraham Lincoln",
  "southern Sweden", "genus Huia", "English monk Bede" -> "Abraham Lincoln", "Sweden", "Huia",
  "Bede"; `_leads_into_name`). Never after a capitalised word or a name particle: "New York"
  never yields "York", "Académie des Beaux-Arts" never "Beaux-Arts".
- **Record title**: `ctx["title"]` (run_head passes the record's title) is one name candidate, in
  front, disambiguator and comma qualifier removed ("Yves Rocher (company)" -> "Yves Rocher").
  The adapter's "<title>. " prefix contributes no candidates of its own.

2. Filters (`head.py`)
- Temporal values and 1-2 character fragments are never entities (`temporal_filter.py`).
- Generic nouns are never entities (`generic_nouns.py`): a frozen, versioned list
  (`codebook/generic_nouns_v1.json`, built by `scripts/build_entity_generic_nouns.py` from
  article-level document frequency), applied to one-word mentions and to a listed noun with only
  adjectives in front ("political landscape"). Ingest only; a query keeps what the user typed.
- A **name** (`EntityCandidate.is_name`: a name span, the title, or a mention whose words are all
  capitalised with one not opening a sentence) is never a codebook rejection (the type becomes
  `proper_noun`) and always passes the compound filter. Other mentions: codebook rejection, then
  the compound filter (`codebook/compound_filter.py`; a lower-case compound introduced by an
  article is a description).
- **Concept gate** (`concepts.py`): a lower-case phrase of two or more words with no proper noun
  ("powerful neighbors", "sedimentary rock") is an entity only as a concept. A seed concept --
  a sealed list of 2.25M sentence-case Wikipedia titles (`codebook/concept_lexicon_v1.npy`, built
  by `scripts/axes/build_concept_lexicon.py`) -- passes the compound filter's word-list rules
  ("high school"). Any other phrase is learned when it is shaped like a term and its modifier is
  not a free one (`codebook/concept_free_modifiers_v1.json`, built by
  `scripts/build_entity_free_modifiers.py`: "pivotal", "intricate", "powerful"), on first sight
  by default (`NAM_ENTITY_CONCEPT_K`). `NAM_ENTITY_CONCEPT_GATE=0` turns the gate off. Ingest only.
- A candidate that is not meaningful keeps its address (the sentence sketch resolves mentions
  through it) but gets no store entry, so a description never resolves as an entity later.

Normalization: `normalize_mention` (snake_case, ASCII-folded, punctuation dropped):
"Great Wyrley F.C." -> `great_wyrley_fc`, "Just Glü It" -> `just_glu_it`.

3. Resolution

Each normalized form is resolved against the local entity store:
- Exact match → reuse entity ID
- Fuzzy match → entity comparator scores candidates (Jaro-Winkler for persons, abbreviation matching for orgs, token overlap for default/verbs)
- No match (ingest mode) → create new entity ID
- No match (query mode) → emit best-effort entity ID without store registration (0.5x confidence penalty, `unresolved: true`)
- Alias → add surface form to existing entity

**Query-mode behavior (v0.0.2):** The entity head is read-only during query execution. It resolves existing entities via exact and fuzzy matching but never creates new entities. This prevents store pollution from short query text and ensures entity IDs remain consistent between ingest and query.

4. Address Emission

Each resolved entity produces one semantic address:
```python
SemanticAddress(
    head_type="entity",
    concept="578b261ef1af",                 # sha256("strom_thurmond")[:12]
    confidence=0.6,
    metadata={
        "surface_form": "Strom Thurmond",   # as the parse has it, token by token
        "normalized": "strom_thurmond",
        "entity_type": "person",            # codebook type
        "meaningful": True,                 # compound filter + concept gate; False = no bundle, no store entry
    }
)
```

## Entity ID Format

`_hash_entity_id`: the first 12 hex digits of sha256(canonical form), no type prefix, so the same
mention gives the same id on every pod, at ingest and at query. The canonical form is the
normalized form with its last word singularised (`_canonical_entity_id`: lemma, then a plural
"s" dropped; never for -ss/-us/-is words such as "business", "campus"), except that a name ending
in a capitalised word or a number keeps its form ("angolan_armed_forces", "united_states").
The entity type stays in address metadata.

## Dependencies

Rule NLP parse from the shared pipeline doc (`ctx["spacy_doc"]`); the head parses with the rule
backend itself only when no doc is passed (query path). The codebook (`codebook/entity_codebook.npz`)
and the generic-noun list are package data loaded at first use.

Output Characteristics
- Multiple entities per record are expected
- Entity count is typically small (1–5)
- No explosion of address space
- Entity IDs are reused across records

Example:

Text:

“Tyler Denton took his children to Central Park.”

Entities: `tyler_denton`, `central_park` ("children" is a generic noun: no entity).

Relationship to Other Heads

| Head |    Responsibility | 
| -----| ------------------- |
| Ontology |    What kind of thing |
| Entity |  Which specific thing |
| Attribute |   Descriptive properties |
| Affordance |  What can be done |
| Context |     Situational qualifiers |

Each head contributes orthogonal meaning.

## Why This Matters

This encoder enables:
- Deduplication
- Referential joins
- Long-term memory
- Cross-document identity
- Stable semantic indexing

Without it, every system collapses into text similarity.