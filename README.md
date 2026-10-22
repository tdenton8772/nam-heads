# nam-heads

Pluggable head implementations for NAM, and the SDK they are written against.

NAM gives every record an address. Each dimension of that address is a **slot**
(`ontology`, `entity`, `attribute`, `affordance`, `context`, plus any slot added
later), and a **head** fills a slot; the NLP parser that every head reads is
loaded the same way (slot `nlp`). One implementation is selected per slot, and
an operator can swap the selection on a running cluster without rebuilding
anything. Every implementation NAM runs lives here, its defaults included; NAM
itself is the host: a factory that loads heads, plus the pipeline around them.

| Implementation | What it does |
|---|---|
| `nlp.rule` | NAM's rule parser (the default): tokenizer, POS tagger, lemmatizer, NER and dependency parser, plus the sentence sketch. Contract `parser-v1`. |
| `ontology.codebook` | The default ontology head. A trained codebook over sentence cues from the host's parse picks a BFO 2020 / CCO v2.2 class. It backs off to an ancestor class, or abstains to `bfo:entity`. |
| `ontology.abstain` | The control. It types every record `bfo:entity`, which shows what the ontology head contributes. |
| `ontology.gaius` | Inspired by [gaius](https://github.com/zndx/gaius): ColBERT MaxSim of the record against verbalized BFO/CCO classes. It is not a reproduction (see its README). |
| `ontology.gaius_ref` | gaius's TBox typing, reproduced. It uses the 905 SDG classes that gaius's own loaders read, scores with ColBERT-Zero MaxSim, and matches gaius's top 8 on 34 of 34 test inputs. |
| `entity.default` | Entity resolution: mentions to canonical entity ids, with comparators and a session-backed entity store. |
| `attribute.default` | Attributes from the parse (adjectives, predications), coded by its own CharCNN axis encoder. |
| `affordance.default` | Verbs and their senses, coded from a verb sense table; records with an unknown verb are parked for its learner part, which decides them offline on a GPU. |
| `context.default` | Particular context: dates, places, institutions and fields (its extractor is `context_axis/`). |

## Layout

```
src/nam_heads/
  sdk/                     the contract
    head.py                EncoderHead, EncodingResult, SemanticAddress
    impl_meta.py           IMPL_META: read, checked, discovered (entry points)
    host.py                what a head borrows from the host: parser, axis encoder, CPU budget, stores
    store.py, merge.py     head stores: state a head keeps between records, and their merges
    learner.py             head learning: parking a record, and the learner part that decides its values
    sentence_ontology.py   how the host derives per-sentence classes
    testing.py             conformance checks for any implementation
  ontology/                the class registry: BFO 2020 + CCO v2.2 (classes.json),
                           plus an extension for an implementation's own classes
    package.py             plugin packages: pack, verify, unpack (the `nam-heads` command)
    parser.py              the NLP plugin contract (parser-v1)
  impls/<slot>/<name>/     one package per implementation: head.py, its data and models, tests/, tools/
tests/                     conformance over every installed implementation
```

## Setup

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"          # numpy, onnxruntime, tokenizers, scipy, pytest
pytest
```

Without NAM, the tests that need NAM, or the model files, skip:

- **Models.** The MaxSim heads load ONNX exports that are not in this repo.
  Each head's README says how to export one; point to them with
  `NAM_GAIUS_MODEL_DIR` (gaius) or `NAM_GAIUS_REF_MODEL_DIR` (gaius_ref). Set
  `NAM_GAIUS_CACHE_DIR` to keep the class bank between runs.
- **NAM as the host.** Heads never parse: the host passes its parse in
  `context["spacy_doc"]`. The default heads (`nlp.rule` and the entity,
  attribute, affordance and context defaults) are also written against NAM's
  host modules (its session-store clients and node cache, `src.nlp`, record
  parsing, the sentence sketch, the axis encoder), so their packages import
  only inside NAM. With NAM checked out next to this repo, everything runs:
  `NAM_SRC=../nam-0.0.1/nam pytest`.

## Plugin packages

A head ships as one versioned package (`.nhp`): its directory, the sha256 of every
file, its `IMPL_META` as data, and its own dependencies as wheels.

```bash
nam-heads pack src/nam_heads/impls/affordance/default -o build/    # one implementation
nam-heads pack-all -o build/                                        # every implementation here
nam-heads verify build/affordance.default-1.0.1.nhp
```

A head's version is the one its `IMPL_META` declares; NAM's head-slot manifest
names the exact version each slot runs. A package is installed into a running
cluster with `POST /v1/admin/plugins` (the raw archive), and runs once a slot
selects it. A new version that writes the same `output` swaps in without
re-ingesting; a different `output` needs re-ingestion.

## How NAM uses this package

With heads as plugins, NAM loads each selected implementation from its package
(installed as above) into a head host in each ingest pod and into the query
service, one worker process per head, with the CPU the slot is given. NAM also
installs this package at a pinned tag: the SDK and the BFO/CCO registry are the
contract, and its `nam.heads` entry points are what a NAM without plugins runs
(the defaults are its built-ins). Selecting an implementation swaps it into the
ingest workers and the query service between batches. An implementation runs
only while its slot selects it.

To ship a change: tag a release here, then bump the pin in NAM. See
[CONTRIBUTING.md](CONTRIBUTING.md) for how to write a new implementation.

## License

Apache-2.0. Third-party data (BFO, CCO, the SDG ontology, Wikipedia probe
texts) is listed in [NOTICE](NOTICE).
