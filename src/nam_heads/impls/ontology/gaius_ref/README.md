# gaius_ref: gaius's TBox typing, reproduced (slot `ontology`, implementation `gaius_ref`)

This implementation types a record the way github.com/zndx/gaius types text,
and with gaius's own ontology. The pins are gaius commit `06cbf95d` and its
`external/sdg-corpora` submodule at `c4138b4d`. gaius trunk at `177e419`
(2026-10-07) changes nothing under `agents/theta`, `engine/embeddings` or
`external`. It sits beside the `gaius` implementation, which is
gaius-inspired and differs from gaius at most stages
(`docs/research/gaius-fidelity-audit.md`).

## What it reproduces

| gaius stage (file:line at 06cbf95d) | gaius_ref |
|---|---|
| TBox: `external/sdg-corpora/ontology/sdg-ontology.owl` (`agents/theta/tbox.py:19`) | the same file, pinned by sha256 (`tbox.py` `OWL_SHA256`; byte-identical at sdg-corpora `c2079e67` and `c4138b4d`) |
| loading: the CCO import from the local `imports/cco-module.ttl`, and the 13 property/class puns no longer typed as classes (`tbox.py:35-82,128-149,198-200`) | gaius's two loaders, copied line for line (`tbox.py` in this package) |
| classes: every named class owlready2 then enumerates, minus Thing/Nothing (`tbox.py:21,165-167,201`) | all 905 (`artifacts/tbox_classes.json`) |
| class text: first label, else IRI local name; IAO_0000115 values joined, else first comment; `f"{label}. {extra}"` (`tbox.py:170-185`) | the same functions, copied line for line |
| model: `lightonai/ColBERT-Zero` through PyLate (`engine/embeddings/colbert.py:27,119-137`) | the same weights (HF revision `43aefda0`, `head.py` `MODEL_PIN`), exported to ONNX fp32 |
| class side encoded as documents, `search_document: ` (`tbox_maxsim.py:116-118`) | the gaius head's ONNX encoder (`impls/ontology/gaius/encoder.py`) |
| query: the item text as given, `search_query: ` (`tbox_maxsim.py:147`) | the record text the ontology stage passes in, unparsed |
| score: Qdrant MAX_SIM over cosine, divided by the query token count (`tbox_maxsim.py:105-114,154-161`) | `ClassBank.scores`: the mean over query tokens of the best class-token cosine |
| output: top 8 IRIs with scores, no threshold, no back-off (`tbox_maxsim.py:135-162`) | the record class = rank 1; the top 8 in `metadata["top_k"]` (below) |
| empty text: no classes (`tbox_maxsim.py:142-143`) | `BFO:0000001`, `abstained: "no_text"` |

Verification (2026-10-07):

- **Class texts.** gaius's own `tbox.py` was run unmodified, on its own
  submodule checkout, under owlready2 0.49 and rdflib 7.5.0 (gaius's
  `uv.lock`). Its `class_verbalizations()` returns 905 classes, all
  indexable, and they match `artifacts/tbox_classes.json` exactly: the same
  905 (IRI, label, text) rows. Its 1,875 entailed subsumption pairs are the
  same as well.
- **Order.** gaius's own enumeration order is not stable. rdflib writes the
  `.owlready/sdg-ontology.nopun.owl` cache in hash-seed order: 6 fresh
  builds gave 6 orders of the same 905 rows. The artifact keeps one fixed
  order, by IRI. Order only breaks exact score ties.
- **gaius's own code against gaius_ref.** What ran: gaius's unmodified
  `tbox.py`, `tbox_maxsim.py` and `colbert.py`, end to end. That covers its
  own loader, the batched `materialize_tbox()` (905 points) and
  `maxsim_tbox_topk(text, k=8)`. The stack was pylate 1.6.0,
  sentence-transformers 5.3.0, transformers 5.15.0, torch 2.13.0, and
  qdrant-client 1.12.1 in local mode. Only the Qdrant client and the CPU
  embedder were replaced. Over 34 inputs (30 held-out records and 4 edge
  cases):
  - gaius's top 8 equals gaius_ref's for 34 of 34, in the same order;
  - the largest score difference is 3.0e-7.

  The report is `artifacts/gaius_native_compare.json`.
- **Encoder parity.** gaius's encode call (`ColBERT.encode([text],
  batch_size=1, is_query, prompt_name)`) was compared with this package's
  ONNX encoder over all 905 class texts and the 34 queries:
  - token ids identical for all;
  - largest vector difference 8.1e-7, largest score difference 5.4e-7.

  The report is `artifacts/parity_pylate.json`.

## Scope: what gaius does with this, and what this head does

- **Admission.** gaius MaxSim-grounds only `admitted_item` rows
  (`clt_incidence.py:92,153`): 512-token windows that its SDG aperture
  admitted. The aperture has 33 SDG domain anchors and `domain_tau` 0.1
  (`clt_skos_admit.py:95-112`; sdg-strategy
  `components/lens/admission_filter.json`). gaius never types arbitrary
  text. This head types whatever text the ontology stage gives it, so its
  coverage below is a property of nam's pipeline and this harness, not of
  gaius. The aperture's anchor vectors are in no repository, so admission is
  not reproduced.
- **Windows.** gaius types each 512-token window, at stride 256
  (`flows/prospects/windows.py:12-13`). The query encoder keeps only the
  first 39 tokens of a window, so typing the record text gives window 0's
  classes exactly. Later windows are not typed.
- **Top k, not top 1.** gaius uses the top 8 as a grounding set, not a
  record type: every hit, weighted by its score, pairs with hits on other
  items that share a CLT feature, and those pairs become candidate novel
  subsumptions (`clt_incidence.py:26-56`).

## The slot contract and gaius's output

`ontology-v1` wants one class per record, so this head reports:
- `addresses`: one address, gaius's rank-1 class, with gaius's own score as
  the confidence. The bundler keys the record by it;
- `metadata["top_k"]`: gaius's whole grounding set, `[[curie, iri, score],
  ...]`, top 8 in gaius's order. Head-result metadata survives the pipeline
  hops as JSON (`serialization/converters.py:185-210`).

Nothing is thresholded, calibrated or backed off. An empty text, where gaius
returns nothing, is `BFO:0000001` with an empty `top_k`. gaius types items,
not sentences, so `sentence_ontologies` is empty.

## Its ontology in nam

`IMPL_META["ontology"]` brings the SDG ontology
(`artifacts/sdg_ontology.json`, schema `nam-ontology-extension/1`). While
the manifest selects gaius_ref, every process that reads the manifest adds
those classes to the ontology registry (`nam_heads/ontology/registry.py`
`use_extension`). The bundler then keys a record by its SDG class. The
design is in NAM's `docs/design-head-slots.md`, "An ontology implementation's own
ontology".

**CURIEs.** 168 of the 905 classes are BFO/CCO classes; they keep their
BFO/CCO CURIE (`cco:ont00001262` is gaius's "person"). The other 737 are:
- `sdg:<name>`: 735 classes;
- `xsd:<name>`: 2 datatypes the file types as classes.

No class is mapped to another. The longest CURIE is 49 characters
(`sdg:ContinuousEmissionsMonitoringSystemEvaluation`); a BFO/CCO CURIE is 15.

**Labels** are gaius's (SDG's own `rdfs:label` first), and **definitions**
are gaius's text after the label.

**Parents** are the TBox's told subsumers:
- named `rdfs:subClassOf` superclasses (what gaius's `tbox.py:213-215` reads);
- named `owl:equivalentClass` classes;
- the named conjuncts of an intersection in either (A = B and ... entails
  A subClassOf B).

gaius reads `is_a` only, so 437 of the 737 would have no parent. Adding the
told subsumers leaves 51 parentless (`artifacts/sdg_ontology.json` `notes`).
505 of the 737 fall under a BFO branch. The hierarchy decides nam's tiering
and back-off, never gaius's typing.

**BFO/CCO stays as it is.**
- A BFO/CCO class keeps BFO/CCO's own parents. The SDG file asserts 5 edges
  on BFO/CCO classes that BFO/CCO lacks; they are not applied.
- `OntologyRegistry.extended` refuses any artifact that would change a
  BFO/CCO answer.

## Measured

gaius at `06cbf95d` publishes no evaluation and no example inputs with
expected classes for TBox typing. Its tests load a 3-class fixture OWL
(`tests/engine/test_theta_cycle.py`) and check shapes and an identity MaxSim
(`tests/engine/test_colbert_zero.py`). The strongest check available is
gaius's own typing code beside this head (Verification, above). This
package's `tests/test_gaius_ref_tbox.py::test_extract_on_the_real_pinned_owl`
loads the real pinned OWL through gaius's loaders. It needs
`NAM_SDG_CORPORA_DIR` pointing at an sdg-corpora checkout at the pin, and
skips without one unless `NAM_GAIUS_REF_REQUIRED=1`.

### Coverage on the held-out records

The run used the 10,810 test records the other ontology heads are measured on
(`impls/ontology/gaius/tools/evalset.py`), with the ColBERT-Zero weights and
the SDG ontology active in the registry. It was offline, on an M3 Pro shared
with other work, on 2026-10-07. The report is `artifacts/coverage_test.json`
(`tools/evaluate.py`).

**Coverage here is a property of this harness, not of gaius.** gaius types
only admitted items (see Scope). The harness types every record.

| | gaius_ref |
|---|---|
| records typed (a class other than the empty-text abstain) | 10,810 (100%) |
| records whose top class is bfo:entity itself (keyed as unclassified) | 1 |
| records the bundler keys by that class | 100% |
| distinct classes used | 564 of 905 |
| share of records in the top 10 classes | 40.1% |
| top class an SDG class / a BFO-CCO class | 86.7% / 13.3% |
| single-thread ms/record | 131 mean, 193 p95 (1-minute load average 13.2) |
| process RSS after 50 records | 735 MB peak (707 MB for the head) |

Latency and RSS were measured on a loaded machine: other sessions held the
1-minute load average at 10-14. Per record, the work is one 39-token query
encode plus MaxSim over the 905 class texts (17,153 class tokens at 918
classes; the 13 punned classes held under 1% of them). The latency is
therefore load-bound here; at a load average of 8.0 the 918-class bank
measured 94 ms and 949 MB peak RSS. macOS RSS also moves with memory
pressure, and the 597 MB fp32 weights dominate it. Not yet re-measured on an
idle machine: run `tools/evaluate.py` at a load average under 2 and update
these two rows.

**What the 13 punned properties would change.** A loader that keeps them as
classes, giving 918, scores every other class identically. On these records
it differs only where a punned property wins:
- 217 records (2.0%) take a punned property as their top class. 171 of them
  take `sdg:publishes_journal`.
- With gaius's loader, those 217 go mostly to `sdg:AcademicJournal` (37),
  `sdg:CredentialedReviewer` (25), `sdg:ScientificOrganization` (18) and
  Person (18).
- Every other record has the same top class under both loaders.
- 3 records swap two adjacent top-8 entries whose scores are exactly equal,
  because the class order breaks such ties (see Order, above).

Top 20 classes (count of 10,810):

| class | label | records |
|---|---|---|
| `cco:ont00001262` | Person (gaius's text: "person") | 657 |
| `sdg:SacredArtifact` | sacred artifact | 531 |
| `sdg:GaulishTribe` | gaulish tribe | 512 |
| `sdg:GeographicRegion` | geographic region | 486 |
| `sdg:AccessionedRecord` | accessioned record | 465 |
| `sdg:StationLocation` | station location | 439 |
| `sdg:Child` | child | 367 |
| `sdg:AncestralSpiritCommunity` | ancestral spirit community | 317 |
| `sdg:ArcticRegion` | arctic region | 305 |
| `sdg:AttendanceRecord` | attendance record | 256 |
| `sdg:DomesticViolenceSurvivor` | domestic violence survivor | 249 |
| `cco:ont00000069` | Material Copy of a Transcript | 233 |
| `sdg:ExpeditionParticipant` | expedition participant | 164 |
| `sdg:CredentialedReviewer` | credentialed reviewer | 153 |
| `sdg:SlavicLinguisticRoot` | slavic linguistic root | 143 |
| `sdg:AnimationSequence` | animation sequence | 142 |
| `sdg:ResidentialStreet` | residential street | 138 |
| `sdg:MinorPerson` | minor person | 137 |
| `sdg:TrafficPath` | traffic path | 108 |
| `sdg:YoungPhysicianAmbassadorRole` | YoungPhysicianAmbassadorRole | 107 |

Against the BFO/CCO gold labels the codebook head is scored on, gaius_ref is
5.7% exact. That number does not measure an SDG typer and is given for
reference only.

### 30 records, checked by hand

The records are a seeded random sample (seed 20261007) of the run above; all
30 have the same class as in the 918-class run. The verdict is whether the
class is a reasonable type for the subject: 9 yes, 4 partly (the right kind
of thing with the wrong specifics), 17 no.

| # | record | subject | gaius class | sensible? |
|---|---|---|---|---|
| 1 | 87054 Laura Skandera Trombley | academic administrator | `sdg:AcademicAgent` academic agent | yes |
| 2 | 13573 A.C.C.D. Mineros de Guayana | football club | `sdg:CollegeCommunityMember` college community member | no |
| 3 | 25990 1991 Mens Hockey Champions Trophy | hockey tournament | `sdg:AttendanceRecord` attendance record | no |
| 4 | 113167 Rocky McIntosh | American football player | `cco:ont00001262` Person | yes |
| 5 | 84148 Agodorp | hamlet | `sdg:GeographicRegion` geographic region | yes |
| 6 | 64767 Faculty of Philosophy, University of Cambridge | university faculty | `sdg:AnalyticMethod` analytic method | no |
| 7 | 77043 Paxton Schulte | ice hockey player | `cco:ont00001262` Person | yes |
| 8 | 13824 MeruPuri | manga series | `sdg:SacredArtifact` sacred artifact | no |
| 9 | 95191 Chris Brink | mathematician | `sdg:QualificationLevel` qualification level | no |
| 10 | 37572 J. P. Kepka | speed skater (adult) | `sdg:MinorPerson` minor person | partly |
| 11 | 60336 Martinstown, County Antrim | village | `sdg:StationLocation` station location | partly |
| 12 | 34860 Sportsmans Association | shooting organisation | `sdg:Membership` membership | no |
| 13 | 19120 Charles de Lannoy, 1st Prince of Sulmona | soldier and statesman | `sdg:GaulishTribe` gaulish tribe | no |
| 14 | 3651 Bae Yong-kyun | film director | `cco:ont00001262` Person | yes |
| 15 | 93412 Whats Left of Me (album) | music album | `cco:ont00000069` Material Copy of a Transcript | no |
| 16 | 87635 Alexa Chung | television presenter | `cco:ont00001262` Person | yes |
| 17 | 96095 National Register of Historic Places listings in | list of NRHP listings | `sdg:AccessionedRecord` accessioned record | partly |
| 18 | 34027 Ossie Vitt | baseball player | `sdg:AccessionedRecord` accessioned record | no |
| 19 | 59582 Ileocolic artery | artery | `sdg:CardiothoracicRadiologistRole` CardiothoracicRadiologistRole | no |
| 20 | 33801 Vefa S.K. | football club | `sdg:AttendanceRecord` attendance record | no |
| 21 | 34872 Hrvoje Bartolović | chess problemist | `sdg:SlavicLinguisticRoot` slavic linguistic root | no |
| 22 | 25452 Flag of Santa Barbara, California | city flag | `sdg:LawEnforcementAgency` law enforcement agency | no |
| 23 | 98110 Laila (actress) | actress (adult) | `sdg:Child` child | partly |
| 24 | 81147 Mojmir I of Moravia | medieval ruler | `sdg:GaulishTribe` gaulish tribe | no |
| 25 | 107193 Gustaaf Adolf Frederik Molengraaff | geologist | `sdg:GaulishTribe` gaulish tribe | no |
| 26 | 114747 The Red & the Black | rock album | `sdg:Agent` agent | no |
| 27 | 59789 New Alipore | city locality | `sdg:GeographicRegion` geographic region | yes |
| 28 | 89154 Doğuş Group | business conglomerate | `sdg:InvestorRelationsActivity` investor relations activity | no |
| 29 | 20907 Pfaueninsel | island | `sdg:GeographicRegion` geographic region | yes |
| 30 | 52415 Aldrington railway station | railway station | `sdg:StationLocation` station location | yes |

The failures share one pattern: a topical word pulls the record to the SDG
class named after it. "Analytic" gives analytic method, "Agent 51" gives
agent, "Croatian" gives slavic linguistic root. The SDG ontology is a data
governance catalog (sdg-corpora README), so it has no class for an album, a
sports club, an artery or most occupations. People, places and stations land
well.

## Running it

```bash
# 1. the TBox and ontology artifacts (dev time; owlready2 + rdflib; gaius pins 0.49 / 7.5.0),
#    from sdg-corpora at c4138b4d; gaius's loaders write their .owlready/ cache beside the OWL
python -m nam_heads.impls.ontology.gaius_ref.tbox \
  --owl <sdg-corpora>/ontology/sdg-ontology.owl
# 2. the ColBERT-Zero export (dev time; torch, transformers, pylate for --check),
#    from a lightonai/ColBERT-Zero snapshot at 43aefda07cf4
python -m nam_heads.impls.ontology.gaius.tools.export_onnx \
  --src <ColBERT-Zero snapshot> --out $NAM_DATA/models/colbert-zero-ref-onnx --check
export NAM_GAIUS_REF_MODEL_DIR=$NAM_DATA/models/colbert-zero-ref-onnx  # images: /app/colbert-zero-ref-onnx
# 3. tests (NAM_SDG_CORPORA_DIR adds the real-OWL load; NAM_GAIUS_REF_REQUIRED=1 turns a
#    missing model or checkout into a failure)
NAM_SDG_CORPORA_DIR=<sdg-corpora> python -m pytest src/nam_heads/impls/ontology/gaius_ref -q
# 4. coverage on the held-out records
python -m nam_heads.impls.ontology.gaius_ref.tools.evaluate --procs 4
```

**What an image needs** (no image ships it today):
- the export directory, 573 MiB (`model.onnx` 596,991,519 bytes, sha256
  `1dae5989e922fc16bb8ee1a77fef2d47dafc02348f9617fee75507348a13d55d`, plus
  the tokenizer and configs), at `/app/colbert-zero-ref-onnx` or
  `$NAM_GAIUS_REF_MODEL_DIR`;
- optionally, the class bank the head builds on first start: 8.9 MB, file
  `gaius_classbank_412a19e294d28b0e.npz` (sha256 `b12621f6...`), in
  `$NAM_GAIUS_CACHE_DIR`. It is read if present and written if absent and
  the directory is writable. The dev Mac has a copy at
  `$NAM_DATA/models/gaius-ref-classbank/`. Without it, the head
  encodes the 905 class texts at start-up. The 918 texts took 51 s on one
  thread (M3 Pro, 2026-10-07); 905 was not re-timed.

Config keys:
- `model_dir`;
- `cache_dir` (the class-bank disk cache, or `$NAM_GAIUS_CACHE_DIR`);
- `max_threads`;
- `top_k` (8);
- `require_colbert_zero` (True);
- `text_cache_size`.
