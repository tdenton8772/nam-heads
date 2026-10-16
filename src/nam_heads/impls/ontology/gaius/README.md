# Gaius-style ontology head (slot `ontology`, implementation `gaius`)

This is the second implementation of the ontology slot. It is independent of the
codebook head (`nam_heads/impls/ontology/codebook`), and the two are swapped in the
same slot. It meets the same contract (`ontology-v1`), so downstream heads
cannot tell which head ran, except by the quality of the classes.

## How close this is to gaius

**It is gaius-inspired, not a reproduction.** The fidelity audit
(`docs/research/gaius-fidelity-audit.md`, gaius pinned at `73722dce`) found
that this head shares two of gaius's ideas: zero-shot late-interaction
retrieval against class texts, and MaxSim divided by the query token count.
It differs on everything else:
- **Ontology:** gaius uses the SDG ontology (918 classes, from its
  `sdg-corpora` submodule); this head uses BFO 2020 + CCO v2.2.
- **Classes:** 69 active classes here; only 27 of them are in gaius's set.
- **Class texts:** CCO definitions with a period added; 0 of the 27 shared
  classes match gaius's text.
- **Input:** a noun phrase from the rule parse here; gaius embeds the raw text.
- **Output:** a calibrated softmax with one top class here; gaius returns its
  top 8 raw scores.
- **Weights:** the export in `docker/assets/colbert-model` is
  **lightonai/ModernColBERT-embed-base**, not ColBERT-Zero; every file hash
  matches that repository. `encoder.py` `MODEL_NAME` and the `colbert-zero-onnx`
  asset path keep the old name.

The faithful reproduction is the separate implementation
`impls/ontology/gaius_ref`. Its 918 class texts match gaius's own `tbox.py`
output exactly.

## Origin

The brief (2026-10-04) was to take gaius's ontology logic
(github.com/zndx/gaius) into a separate head, with the ColBERT-Zero MaxSim
encoder and the full certified TBox as the class set. The gaius code (commit `73722dce`):

- `agents/theta/tbox.py`: one certified OWL TBox, and no class is ever minted
  from text. Each class becomes `label + ". " + definition`, where the
  definition is the IAO_0000115 annotation (here, CCO's `skos:definition`).
- `agents/theta/tbox_maxsim.py`: every class text is embedded with
  `lightonai/ColBERT-Zero`, the documents use the prompt `search_document:`,
  and the vectors go into a MaxSim collection. To type a text, gaius encodes
  it as a query (`search_query:`) and takes the top classes by MaxSim score
  divided by the number of query tokens. Typing is retrieval. Nothing is
  trained.

## How this head applies it

| Piece | File |
|---|---|
| TBox annotations (label, alternative labels, definitions, examples, scope notes) for all 1,437 classes, extracted from the pinned BFO 2020 + CCO v2.2 files | `verbalize.py`, `artifacts/tbox_annotations.json` |
| A ColBERT model on onnxruntime, fp32: ModernColBERT-embed-base weights (see above). Tokenisation matches PyLate token for token. | `encoder.py`, `tools/export_onnx.py` |
| Class bank (the fixed class side, built once per process), MaxSim, softmax, resolution | `scoring.py` |
| Selection of the text to embed from the rule-backend parse | `text.py` |
| Head (contract, diagnostics, `IMPL_META`, `check_available`) | `head.py` |
| Calibration (temperature, acceptance threshold) on validation records | `tools/calibrate.py`, `artifacts/calibration.json` |
| Comparison with the codebook head on held-out records | `tools/evaluate.py` |

- **Classes.** The head retrieves among the active class set
  (`nam_heads/ontology/active.py`), minus bfo:entity and the BFO top categories
  (continuant, occurrent and the four branch classes). That leaves 69 classes.
  The top categories are reachable only by back-off: when they were
  retrievable, "continuant" won 6-7% of validation records. The full TBox
  (1,436 classes, mapped back to the active set) did worse, at 24% exact
  against 29%, because CCO's 340 Material Artifact subclasses absorbed
  records (offline, 2026-10-07).
- **Verbalization.** The style is gaius's own, `label. definition`. The
  richer `tbox` style (alternative labels, parents, examples) and label-only
  texts both scored lower on validation.
- **Text.** The head embeds the defining phrase of the lead sentence, the
  noun phrase after the first "be", extended through "of"; for example, "was
  a German-born theoretical *physicist*" gives "a German-born theoretical
  physicist". Without such a phrase it embeds the title plus the lead. Each
  further real sentence, up to `max_sentences` (default 4), is typed from its
  own phrase. Focusing on the phrase gave 29% exact on validation, against 24%
  for the whole lead. Longer query windows (64 or 128 tokens) did not help.
- **Score.** MaxSim, averaged over all query tokens, prompt tokens included
  (gaius divides by the query token count). Scoring only the phrase tokens or
  the head noun was worse.
- **Calibration.** The softmax temperature is fitted by NLL on the 10,960
  validation records. The acceptance threshold is 0, so the head commits to
  its top class. Every threshold above 0 traded exact classes for back-offs to
  material entity or continuant. The curve is in `calibration.json`.
- **Abstain.** The head returns `BFO:0000001` when there is no parse
  (`abstained: "no_parse"`) or no text to type. With a configured `accept`, it
  also abstains when not even an ancestor holds that much probability.
- **Pins.** The head refuses to start on any of these: a different TBox, a
  different verbalization digest, different source weights, or a model whose
  three probe scores drift by more than 2e-3.

## Measured

Both heads were run on the codebook's held-out test split (10,810 records),
with the same production text and the same rule parse. Labels come from
Wikidata P31 through `cco_anchors.py`. These numbers are offline
measurements from 2026-10-07; the full report is `artifacts/eval_test.json`
(`tools/evaluate.py`).

| held-out records | codebook | gaius |
|---|---|---|
| exact class | 77.5% | 32.2% |
| exact or correct-but-coarse | 82.4% | 34.3% |
| abstained (bfo:entity) | 0.34% | 0.0% |
| typed Person (29.6% of the gold labels are Person) | 30.5% | 23.7% |
| distinct classes emitted | 60 | 69 |
| CPU ms/record, 1 thread, M3 Pro, machine shared | 2.1 (0.8 unloaded) | 146 (120 unloaded); 50 with `max_sentences=1` |
| process RSS after 50 records | 75 MB | 745 MB |

The two heads give the same class on 32.0% of records. On 48.6%, one head's
class is the other's, or subsumes it.

The main confusions, with counts out of 10,810:

| gold class | gaius class | records |
|---|---|---|
| ICE | Person | 356 |
| Person | Election | 237 |
| Person | Armed Force | 144 |
| Geospatial Region | Village | 131 |
| Person | Video | 108 |
| Document | Person | 108 |

The pattern is that a topical word in the phrase pulls the record to its
topic's class: "a politician" goes to Election, and "an officer" goes to
Armed Force.

Two things explain most of the gap. First, the CCO TBox has no vocabulary
for most Wikipedia subjects: occupations, creative works and taxa. Second,
the labels encode the codebook's own judgment calls (a fictional character
is ICE, a taxon is Organism), and zero-shot retrieval cannot know them.

Calibrated confidences are low: the mean top-class probability is 0.16. The
slot's `ontology_unclassified` diagnostic (top probability below 0.25)
therefore fires on most records, and that is an accurate signal.

## Running it

It needs `numpy`, `onnxruntime` and `tokenizers` (`IMPL_META["requires"]`),
plus the exported model directory (`IMPL_META["assets"]`). nam-nats-worker
and nam-query install all three packages, but neither image ships the model
directory today. Keep a local export outside `docker/assets/`: the build
contexts do not exclude `colbert-zero-onnx`, so 600 MB there would be uploaded
with every image build. The dev Mac's copy is
`$NAM_DATA/models/colbert-zero-onnx`.

```bash
# 1. export (dev time: torch, transformers, pylate for --check)
python -m nam_heads.impls.ontology.gaius.tools.export_onnx \
  --src docker/assets/colbert-model --out <dir>/colbert-zero-onnx --check
# 2. point the head at it
export NAM_GAIUS_MODEL_DIR=<dir>/colbert-zero-onnx      # images: /app/colbert-zero-onnx
# 3. tests (NAM_GAIUS_REQUIRED=1 turns a missing model into a failure)
python -m pytest src/nam_heads/impls/ontology/gaius -q
# re-calibrate after any change to the model, the verbalizations or the class set
python -m nam_heads.impls.ontology.gaius.tools.calibrate --procs 5
```

Config keys: `model_dir`, `cache_dir` (class-bank disk cache, or
`$NAM_GAIUS_CACHE_DIR`), `max_threads`, `max_sentences`, `max_addresses`,
`min_address_confidence`, `accept`, `text_cache_size`. Diagnostics use the
slot's `NAM_ONTOLOGY_*` knobs and the codebook head's snapshot schema
(`heads/ontology/diagnostics.py`), so the dashboard merges either head the
same way.
