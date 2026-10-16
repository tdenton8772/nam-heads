# gaius fidelity audit: the `gaius` and `gaius_ref` ontology implementations

This audit asks whether nam's `gaius` ontology implementation
(`nam/src/encoders/impls/ontology/gaius/`) types text as gaius does. gaius is
github.com/zndx/gaius, by Ryan. It also describes `gaius_ref`
(`nam/src/encoders/impls/ontology/gaius_ref/`), the variant built to follow
gaius stage by stage.

## Pins

| What | Pin | How it was established |
|---|---|---|
| gaius, for the deviation table | `73722dce27adcceb6c3bb3697b7efdc4a3c9d7bc` (2026-09-21), the commit nam `gaius` was built from | Cloned 2026-10-07. |
| gaius, for `gaius_ref` | `06cbf95dbba33d5e865bc76b80bb6d275bc67714` (trunk, 2026-10-07). Trunk `177e419` changes nothing under `agents/theta`, `engine/embeddings` or `external`. | Cloned 2026-10-07; `git diff 06cbf95d 177e419` on those paths is empty. |
| gaius's TBox | `external/sdg-corpora/ontology/sdg-ontology.owl`, submodule at `c2079e67` (73722dce) and `c4138b4d` (06cbf95d). The OWL and `imports/cco-module.ttl` are byte-identical at both. | gaius `git submodule status`; sha256 of both files at both commits. github.com/zndx/sdg-corpora is public over HTTPS. |
| gaius's model stack | pylate 1.6.0, sentence-transformers 5.3.0, transformers 5.15.0, torch 2.13.0, owlready2 0.49, qdrant-client 1.12.1 | gaius `uv.lock`, lines 4828, 5505, 5825, 6181, 6739, 6864 |
| gaius's model | `lightonai/ColBERT-Zero` at HF revision `43aefda07cf4` (main, 2026-08-31): `model.safetensors` sha256 `57781fbe...`, `1_Dense/model.safetensors` `4402f4e6...`, `tokenizer.json` `898c16d2...` | `engine/embeddings/colbert.py:27`; the HF API's LFS sha256s, matched by the downloaded files (2026-10-07) |
| nam `gaius` | as of nam commit `50a1324b`. `git diff 50a1324b 7348abe2` is empty for that package. | Ran 2026-10-07. |

In the tables, the gaius column cites files of gaius at the pin, and the nam
column cites files of `nam/src/encoders/impls/ontology/gaius/`. `sdg:` marks a
file of sdg-corpora at its pin.

## Verdict

**nam's `gaius` implementation does not reproduce gaius.** It is a different
classifier that shares two of gaius's ideas: typing as zero-shot
late-interaction retrieval against class texts, and MaxSim divided by the
query token count. Every other stage that decides the class differs:

- the ontology;
- the class set;
- the class texts;
- the model weights;
- the text that is embedded;
- the output (gaius returns raw top-8 scores; ours a calibrated top class).

What it does reproduce is the encoding detail: the prompts, the [Q]/[D]
markers, the lengths, the skiplist, the normalisation and the scoring
arithmetic. All of these match gaius's.

The 32.2% exact (nam `gaius`) is therefore not a measurement of gaius's
method. Nor is the TBox nam `gaius` searches gaius's. When the head was
specified, the class set was described as "Full certified TBox: All ~1,596
BFO+CCO classes, as gaius does" (2026-10-04). gaius's certified TBox is the
SDG ontology, not BFO+CCO. The message before that offer (line 1129) said
the file "sits in a private submodule I couldn't open". The submodule's
repository, github.com/zndx/sdg-corpora, is public over HTTPS.

gaius's own loaders read its TBox as 905 classes (next section). `gaius_ref`
reproduces gaius at `06cbf95d`:
- its class side exactly: the same 905 (IRI, label, text) rows as gaius's
  `class_verbalizations()`;
- its weights: lightonai/ColBERT-Zero;
- its query and scoring: gaius's own `tbox.py`, `tbox_maxsim.py` and
  `colbert.py`, run end to end, return the same top 8 as `gaius_ref` for 34
  of 34 inputs (see "gaius_ref").

## Loading gaius's TBox

The OWL imports `https://signals.zndx.org/sdg/imports/cco-module`, which does
not resolve (`host`: NXDOMAIN). It also types 13 IRIs both `owl:Class` and an
object or datatype property, and owlready2 refuses such puns
(`Ontology._load_properties`).

- **At `73722dce`**, `tbox.py:78` calls `get_ontology(<sdg-ontology.owl>).load()`
  directly. Run unmodified on 2026-10-07 (owlready2 0.49 and 0.51), it raised
  `OwlReadyOntologyParsingError: Cannot download`. With the import supplied,
  it raised `TypeError` on `sdg#bindsProvider`.
- **At `06cbf95d`**, `tbox_relations` loads through two loaders
  (`tbox.py:198-200`):
  - `load_pinned_imports` (`tbox.py:128-149`) writes the pinned
    `imports/cco-module.ttl` as RDF/XML to `ontology/.owlready/cco-module`
    and puts that folder on owlready2's `onto_path`;
  - `ontology_for_owlready` (`tbox.py:35-82`) writes
    `.owlready/sdg-ontology.nopun.owl`, which drops the `owl:Class` typing and
    `rdfs:subClassOf` of every object and datatype property.

  Run unmodified on its own submodule checkout on 2026-10-07 (owlready2
  0.49, rdflib 7.5.0, gaius's `uv.lock`): `class_verbalizations()` returns
  905 classes, all 905 indexable, with 1,875 entailed subsumption pairs. The
  13 puns are no longer classes. Ryan's review of this audit
  (`docs/scratch/2026-10-07/162158_nam_tbox_head_review.md` in gaius) gives
  the same count.
- **Order.** gaius's class order at `06cbf95d` is not stable. rdflib writes
  the `.nopun.owl` cache in hash-seed order: 6 fresh builds gave 6 orders of
  the same 905 rows. Order only breaks exact score ties.

gaius's tests load a 3-class fixture OWL (`tests/engine/test_theta_cycle.py`),
never the real one. `gaius_ref`'s `test_extract_on_the_real_pinned_owl` loads
the real OWL through gaius's loaders when `NAM_SDG_CORPORA_DIR` names a
checkout.

## Deviation table

"Changes the class?" asks whether the difference can change the top
predicted class for the same input record.

| Stage | gaius (file:line) | nam `gaius` (file:line) | Identical | Changes the class? |
|---|---|---|---|---|
| TBox file | `sdg-ontology.owl` (`tbox.py:18-19`). It holds 918 named classes (`sdg:ontology/HERMIT_CERTIFICATE.md`) and imports a CCO module; gaius's loader at `06cbf95d` reads 905 of them as classes. | BFO 2020 + CCO v2.2 merged: the nam registry, 1,437 classes (`verbalize.py:3-7,62-100`) | no | yes |
| Classes retrieved | every class `onto.classes()` lists, minus Thing/Nothing (`tbox.py:21,45-47,79`). At `06cbf95d` that is 905: 735 `sdg:`, 154 CCO, 14 BFO, 2 XSD datatypes. | 69: the nam active set minus bfo:entity, continuant, occurrent and the 4 BFO branches (`tools/calibrate.py:38,48-53`; `artifacts/calibration.json`) | no. 27 of our 69 are in gaius's 905; 168 of gaius's 905 are nam registry classes. | yes |
| Mapping back | none. It returns the class IRI (`tbox_maxsim.py:110-116`). | registry CURIE; back-off to an ancestor when below `accept` (`scoring.py:120-136`) | no | no for the top class (`accept` = 0, so back-off never fires) |
| Label | first `rdfs:label` owlready2 returns, else the IRI local name (`tbox.py:50-54`). SDG's own labels come first: CCO Person is "person", Material Artifact is "artifact", Document Content Entity is "ont00002039". | registry label: "Person", "Material Artifact" (`verbalize.py:84,124`) | no | yes |
| Definition | every `IAO_0000115` value joined by a space, else the first non-blank `rdfs:comment` (`tbox.py:57-64`). The CCO module carries `skos:definition` only, so the 155 CCO classes get label-only texts. | CCO `skos:definition`, with BFO's "(elucidation)" prefix stripped (`verbalize.py:45,52-54,87-96`) | no. 0 of the 27 shared classes have the same text. | yes |
| Template | `f"{label}. {extra}".strip()`, else the label; no period added (`tbox.py:65`) | `f"{label}. {_sentence(definition)}"`: adds a final period, collapses whitespace (`verbalize.py:57-59,127`) | no | yes (small) |
| Input text | an admitted 512-token window of the source's plain text, passed raw (`cycle.py:165,169`; `clt_incidence.py:72`; `clt_skos_admit.py:95-112`; `db/migrations/20260818000002_clt_skos_ledger.sql:2`) | the defining noun phrase of the lead sentence from the rule parse, else title plus lead (`text.py:86-124,146-159`; `head.py:372-375`) | no | yes, the largest input change |
| Admission | only windows the SDG aperture admits: 33 SDG anchors, `domain_tau` 0.1 (`clt_skos_admit.py:95-112`; sdg-strategy `components/lens/admission_filter.json`) | none | no | no: it decides whether a record is typed, not as what |
| Windows | each 512-token window at stride 256 is typed separately (`flows/prospects/windows.py:12-13,57-63`) | one record class, plus up to 4 per-sentence phrases (`head.py:376-387`) | no | no for window 0 |
| Model | `lightonai/ColBERT-Zero` through PyLate (`engine/embeddings/colbert.py:27,119-122`; `tbox_maxsim.py:19`). HF main `43aefda0`, `model.safetensors` sha256 `57781fbe...`. | `lightonai/ModernColBERT-embed-base` @ `0201e575` (`docker/assets/colbert-model`; `model.safetensors` sha256 `a4217337...`). `export.json` calls it ColBERT-Zero because the name is a constant (`encoder.py:40`; `tools/export_onnx.py:72`). | no | yes: different weights change every score |
| Runtime | PyLate on torch | onnxruntime fp32 graph of the same layers (`encoder.py`; `tools/export_onnx.py:46-58`) | numerically, on the same weights (see "Parity") | no |
| Query prompt and marker | `prefix="search_query: "` selects `prompt_name="query"` (`tbox_maxsim.py:101`; `colbert.py:128-137`). PyLate prepends the prompt and inserts `[Q]` after `[CLS]`. | the prompt is prepended and `[Q]` inserted after `[CLS]` (`encoder.py:102-110`) | yes (token ids equal) | no |
| Document prompt and marker | `prefix="search_document: "`, text stripped, empty skipped (`tbox_maxsim.py:66-70`) | the same, with `[D]` (`encoder.py:102-110,137-142`) | yes | no |
| Lengths | query 39, document 519 (model `config_sentence_transformers.json`; the same in both checkpoints) | read from the same config (`encoder.py:83-84,106`) | yes. gaius's 512-token window is cut to its first ~33 text tokens; our phrase is shorter. | no |
| Query expansion | off (`do_query_expansion: false`) | off; a model with it on is refused (`encoder.py:80-81`) | yes | no |
| Lowercasing and stripping | none; `do_lower_case: false`. Sentence-transformers strips the text (`Transformer.py:383`, 5.3.0). | none; the text is stripped (`encoder.py:107`) | yes | no |
| Skiplist | punctuation dropped from documents only (PyLate `colbert.py:750-757`) | the same (`encoder.py:95-96,141`) | yes | no |
| Normalisation | PyLate L2-normalises token vectors; Qdrant COSINE normalises again (`tbox_maxsim.py:58-63`) | L2 normalisation inside the ONNX graph (`tools/export_onnx.py:56-58`) | yes | no |
| MaxSim | Qdrant MAX_SIM: the sum over query vectors of the best cosine, divided by the query vector count, `[CLS]`/`[Q]`/prompt/`[SEP]` included (`tbox_maxsim.py:55-63,108-115`) | the mean over all query tokens of the best class-token cosine (`scoring.py:78-84`) | yes (same arithmetic) | no |
| Search | Qdrant `query_points`, `limit=k` (`tbox_maxsim.py:102-107`). Exact or HNSW is not set in the code. | exact over all classes | exact in Qdrant's local mode (34 of 34 top-8 lists equal, 2026-10-07); a Qdrant server's index is unverified | only if a server's search is approximate |
| Scores to output | raw score, top k = 8, no threshold, no softmax (`tbox_maxsim.py:87-90,109-116`) | temperature softmax (T = 0.00418, fitted on 10,960 labelled validation records), `accept` 0, runner-ups with p >= 0.05 up to 5 (`scoring.py:87-136`; `head.py:113-121,337-349`; `calibrate.py:79-88`) | no | no for the top class (softmax is monotonic); the runner-up list differs |
| Hierarchy | not used in typing; entailed subsumption only excludes BERTSubs pairs afterwards (`clt_incidence.py:50`; `tbox.py:83-93`) | back-off, and ancestors are emittable (`scoring.py:104-136`; `head.py:304-311`) | no | no at `accept` 0 |
| Abstention | an empty text or `k < 1` gives `[]` (`tbox_maxsim.py:94-95`); otherwise always k classes | `BFO:0000001` with no parse, no text, or (with `accept` > 0) low mass (`head.py:321-324,363-371`) | no | rarely: ours abstains without a parse |
| Per-sentence classes | none | `sentence_ontologies` for up to 4 sentences (`head.py:376-387`) | no | sentence votes only |

The material rows are, in order of effect:

1. the input text;
2. the TBox, class set and class texts;
3. the model weights;
4. the scoring arithmetic, prompts, markers and lengths. These are the
   same in both, as are the effective (top-class) results of the
   calibration.

## Parity: onnxruntime against PyLate

**On the real weights.** These are `lightonai/ColBERT-Zero` at HF revision
`43aefda0`; every downloaded file matched the repository's LFS sha256. The
export is the `colbert-zero-ref-onnx` directory (`model.onnx`
sha256 `1dae5989e922fc16bb8ee1a77fef2d47dafc02348f9617fee75507348a13d55d`).

gaius's encode call (`ColBERT.encode([text], batch_size=1, is_query,
prompt_name)`, `colbert.py:128-137`) ran on gaius's model stack in a
throwaway venv:
- pylate 1.6.0, sentence-transformers 5.3.0, transformers 5.15.0,
  torch 2.13.0;
- pylate was installed without its `transformers<=5.3.0` bound, as gaius's
  `override-dependencies` do (`pyproject.toml`).

It was compared with nam's onnxruntime encoder over all 905 class texts and
34 queries (30 held-out records and 4 edge cases), on 2026-10-07:
- token ids are identical for every text;
- the largest vector difference is 8.1e-7;
- the largest score difference is 5.4e-7;
- gaius's top 8 is identical for 34 of 34 queries.

The report is `gaius_ref/artifacts/parity_pylate.json`. The exporter's own
check, run in nam's env (pylate 1.6.0, transformers 4.57.6, torch 2.11.0),
gave 6.6e-7.

**On the ModernColBERT-embed-base weights** (nam `gaius`'s export), the same
check over 10 texts gave identical token ids and a largest difference of
4.5e-7. The runtime is faithful; the weights were the deviation.

## gaius_ref

`nam/src/encoders/impls/ontology/gaius_ref/` (README there). It differs from
nam `gaius` in every material row above, and matches gaius in each:

- **Class side.** `artifacts/tbox_classes.json` holds gaius's 905 classes
  and texts. It is extracted with gaius's own loaders and verbalization
  functions, copied line for line. It equals gaius's
  `class_verbalizations()` at `06cbf95d` in every (IRI, label, text) row, and
  the entailed pairs are the same too. The artifact keeps one fixed order,
  by IRI, where gaius's varies. Runtime-verified 2026-10-07.
- **Query.** The record text the ontology stage passes in, unparsed. Typing
  the record text gives the same token ids as gaius's window 0, because the
  query keeps only the first 39 tokens.
- **Model.** lightonai/ColBERT-Zero. It pins the source hashes and refuses
  another export.
- **Score and output.** gaius's top 8, by MaxSim divided by the query token
  count. There is no softmax, threshold or back-off.
- **Contract.** gaius uses its top 8 as a grounding set for CLT-incidence
  subsumption pairs (`clt_incidence.py:26-56`), not as a record type. The
  slot wants one record class, so `addresses` holds gaius's rank-1 hit, and
  `metadata["top_k"]` carries the whole top 8 with gaius's scores.
- **Scope.** gaius types only `admitted_item` rows, after SDG aperture
  admission (`clt_incidence.py:92,153`; `clt_skos_admit.py:95-112`). This head
  types every record it is given, so its coverage is a property of nam's
  pipeline, not of gaius.
- **Ontology.** The SDG classes are first-class in nam while gaius_ref is
  selected: `IMPL_META["ontology"]` brings them, and every manifest reader
  adds them to the ontology registry. Design: NAM's `docs/design-head-slots.md`,
  "An ontology implementation's own ontology".

**gaius's own code against gaius_ref (2026-10-07).** What ran:
- gaius's unmodified `tbox.py`, `tbox_maxsim.py` and `colbert.py` at
  `06cbf95d`, end to end, loading its own submodule checkout;
- gaius's stack (above) plus qdrant-client 1.12.1 in local mode;
- seams replaced: the Qdrant client and the CPU device only.

`materialize_tbox()` built gaius's `sdg_tbox` collection: 905 points,
MAX_SIM over cosine, upserted in batches of 64. `maxsim_tbox_topk(text, k=8)`
then gave the same top 8 as gaius_ref for 34 of 34 inputs, with a largest
score difference of 3.0e-7 (`gaius_ref/artifacts/gaius_native_compare.json`).

Not reproduced: SDG aperture admission, and the typing of windows after the
first. 56 of the 10,810 held-out records are longer than one 512-token
window.

## Measurements

The run used the 10,810 held-out test records and the same production text
as the existing report (`impls/ontology/gaius/artifacts/eval_test.json`). It
was offline, on an M3 Pro shared with other work, on 2026-10-07.

gaius_ref's classes are the SDG ontology's, so accuracy against the BFO/CCO
gold labels does not measure it. That column is for reference only, and the
coverage numbers are in gaius_ref's README.

| 10,810 held-out records | codebook | nam `gaius` | gaius_ref |
|---|---|---|---|
| ontology | BFO/CCO | BFO/CCO (69 classes) | SDG (905 classes) |
| exact against BFO/CCO gold (reference only) | 77.5% | 32.2% | 5.7% |
| records typed | 99.7% | 100% | 100% |
| distinct classes used | 60 | 69 | 564 |
| top class an SDG class | 0 | 0 | 86.7% |
| CPU ms/record, 1 thread (1-minute load average) | 2.1 (12.1) | 146 (12.1) | 131 (13.2) |
| process RSS after 50 records | 75 MB | 745 MB | 735 MB |

The codebook and nam `gaius` columns come from
`impls/ontology/gaius/artifacts/eval_test.json`. The gaius_ref column comes
from `gaius_ref/artifacts/coverage_test.json` (`gaius_ref/tools/evaluate.py`,
4 processes). Its latency and RSS were re-taken alone on 300 and 50 records
at a load average of 13.2, so they are only roughly comparable. The same
head on the 918-class bank measured 94 ms and 949 MB at a load average of
8.0.

gaius_ref's coverage is a property of this harness: gaius types only
admitted items. Hand check of 30 random records (gaius_ref README): 9
sensible, 4 partly, 17 not. People, places and stations land well. Records
with no matching SDG class are pulled to an SDG class by a topical word; the
misses include albums, clubs, an artery and most occupations.

## What would settle the rest

- **Qdrant search exactness on a server.** It is not set in gaius's code.
  Local mode is exact; a server's index was not measured.
- **gaius's own examples.** gaius at `06cbf95d` publishes no evaluation and
  no expected classes for TBox typing. `tests/engine/test_colbert_zero.py`
  checks shapes and the identity MaxSim only, and `tests/engine/test_clt_skos.py`
  mocks MaxSim. The strongest available evidence is running gaius's own
  typing code beside gaius_ref (above).
- **Latency on an idle machine.** Run `gaius_ref/tools/evaluate.py` at a
  1-minute load average under 2.
