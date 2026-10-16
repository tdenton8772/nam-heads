# nam-heads

Pluggable head implementations for NAM, and the SDK they are written against.

NAM gives every record an address. Each dimension of that address is a **slot**
(`ontology`, `entity`, `attribute`, `affordance`, `context`, plus any slot added
later), and a **head** fills a slot. One implementation is selected per slot,
and an operator can swap the selection on a running cluster without rebuilding
anything. That holds today for heads that keep no state, such as the ontology
heads; see [State](CONTRIBUTING.md#state-not-yet-provided-by-the-host) for the
others. This repository holds implementations that are developed outside NAM's
own tree, starting with the ontology slot:

| Implementation | What it does |
|---|---|
| `ontology.codebook` | NAM's built-in ontology head (the default). A trained codebook over sentence cues from the host's parse picks a BFO 2020 / CCO v2.2 class. It backs off to an ancestor class, or abstains to `bfo:entity`. |
| `ontology.abstain` | The control. It types every record `bfo:entity`, which shows what the ontology head contributes. |
| `ontology.gaius` | Inspired by [gaius](https://github.com/zndx/gaius): ColBERT MaxSim of the record against verbalized BFO/CCO classes. It is not a reproduction (see its README). |
| `ontology.gaius_ref` | gaius's TBox typing, reproduced. It uses the 905 SDG classes that gaius's own loaders read, scores with ColBERT-Zero MaxSim, and matches gaius's top 8 on 34 of 34 test inputs. |

## Layout

```
src/nam_heads/
  sdk/                     the contract
    head.py                EncoderHead, EncodingResult, SemanticAddress
    impl_meta.py           IMPL_META: read, checked, discovered (entry points)
    host.py                what a head borrows from the host: parser, axis encoder, CPU budget
    sentence_ontology.py   how the host derives per-sentence classes
    testing.py             conformance checks for any implementation
  ontology/                the class registry: BFO 2020 + CCO v2.2 (classes.json),
                           plus an extension for an implementation's own classes
  impls/<slot>/<name>/     one package per implementation: head.py, artifacts/, tests/, tools/
tests/                     conformance over every installed implementation
```

## Setup

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"          # numpy, onnxruntime, tokenizers, scipy, pytest
pytest
```

Without NAM, the tests that need NAM's parser, or the model files, skip:

- **Models.** The MaxSim heads load ONNX exports that are not in this repo.
  Each head's README says how to export one; point to them with
  `NAM_GAIUS_MODEL_DIR` (gaius) or `NAM_GAIUS_REF_MODEL_DIR` (gaius_ref). Set
  `NAM_GAIUS_CACHE_DIR` to keep the class bank between runs.
- **Parser.** Heads never parse. The host passes its parse in
  `context["spacy_doc"]`. Tests that need a real parse run when NAM is checked
  out next to this repo: `NAM_SRC=../nam-0.0.1/nam pytest`.

## How NAM uses this package

NAM installs it at a pinned tag. NAM finds implementations through the
`nam.heads` entry-point group, and lists them in its head-slot manifest and its
`/admin/heads` dashboard. Selecting one swaps it into the ingest workers and the
query service between batches. An implementation runs only while its slot
selects it.

To ship a change: tag a release here, then bump the pin in NAM. See
[CONTRIBUTING.md](CONTRIBUTING.md) for how to write a new implementation.

## License

Apache-2.0. Third-party data (BFO, CCO, the SDG ontology, Wikipedia probe
texts) is listed in [NOTICE](NOTICE).
