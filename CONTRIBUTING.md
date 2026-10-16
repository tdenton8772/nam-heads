# Writing a head implementation

## 1. The package

Create `src/nam_heads/impls/<slot>/<name>/` with an `__init__.py` and a
`head.py`. `<name>` matches `[a-z][a-z0-9_]{1,31}`. Then register the package in
`pyproject.toml`:

```toml
[project.entry-points."nam.heads"]
"ontology.myhead" = "nam_heads.impls.ontology.myhead"
```

Run `pip install -e .` again after adding an entry point.

## 2. `head.py`

`head.py` defines exactly one `EncoderHead` subclass, and a module-level
**literal** `IMPL_META`:

```python
from nam_heads.sdk import EncoderHead, EncodingResult, SemanticAddress
from nam_heads.ontology.bindings import UNCLASSIFIED   # bfo:entity

IMPL_META = {
    "slot": "ontology",
    "name": "myhead",
    "version": "0.1.0",
    "contract": "ontology-v1",
    "description": "One line for the dashboard.",
    "requires": ["numpy"],                      # top-level imports beyond numpy
    "assets": [{"name": "my-model", "env": "MYHEAD_MODEL_DIR",
                "default": "/app/my-model", "files": ["model.onnx"]}],
    # optional, ontology slot only: classes of your own (see section 4)
    # "ontology": {"name": "myonto", "artifact": "artifacts/myonto.json"},
}


class MyOntologyEncoder(EncoderHead):
    def _initialize(self): ...                  # load models; self.config is the host's dict
    def encode(self, text, context=None) -> EncodingResult: ...
    def encode_query(self, query, context=None) -> EncodingResult: ...
    def classify_nearest(self, text, spacy_doc=None): ...   # a class or None
    def emittable_classes(self): ...            # every class encode can return
```

The host reads `IMPL_META` with `ast`, without importing the module. Values must
be literals, or names bound to earlier module-level literals. A requirement or
asset that is missing makes the implementation **unavailable**: the host lists
it but refuses to select it. It is never half-loaded.

## 3. The `ontology-v1` contract

`nam_heads.sdk.testing.check_ontology_head` checks all of this, and
`tests/test_conformance.py` runs it on every installed implementation.

- `encode` / `encode_query` return an `EncodingResult` with `head_type ==
  "ontology"` and at least one address. `addresses[0].concept` is the record's
  class: it becomes the address key's type segment, so every record, and the
  query that should find it, is partitioned by it.
- Every class must be in the registry (`nam_heads.ontology.registry`). When
  nothing fits, return `bfo:entity` and say why in `metadata["abstained"]`.
  Never return an empty list.
- `confidence` is in [0, 1].
- `metadata["sentence_ontologies"]` (optional) maps sentence index to class,
  with `sentence_confidences` alongside. The other heads steer by it. If you
  leave it out, the host fills it by calling `classify_nearest` on each
  sentence (`sdk/sentence_ontology.py`).
- `context["spacy_doc"]` is the host's parse. It may be absent: answer anyway,
  even if only by abstaining. Do not parse inside the head.
- Ingest and query must classify the same text the same way. Retrieval looks
  in the partition the query's class names.
- Keep class CURIEs short. The class is part of the address key and the graph
  index keys, and every key must fit the data service's 250-byte cap. NAM's
  `test_ontology_key_cap.py` builds the longest key for every class of every
  installed implementation's ontology.

## 4. Classes of your own

An ontology implementation may bring classes that are not in BFO/CCO, as
`gaius_ref` does with the SDG ontology. Put them in an extension artifact
(schema `nam-ontology-extension/1`; see
`impls/ontology/gaius_ref/artifacts/sdg_ontology.json` and
`registry.OntologyRegistry.extended`) and name it in `IMPL_META["ontology"]`.
While your implementation is selected, every NAM process adds those classes to
its registry: the bundler, the address builder, the other heads and the query
planner.

## State: not yet provided by the host

NAM's built-in entity, attribute, affordance and context heads keep learned
state in NAM's session store, read through a per-node LMDB cache. The host does
not yet hand that store to an implementation: `EncoderHead` receives only
`head_type` and `config`. Its `kv_on_cache_miss` flag (config key, or
`NAM_<SLOT>_KV_ON_MISS`) is the host's miss policy for that store, and has no
effect on an implementation that keeps no state.

So, today:

- an implementation must be **stateless** between records (it may load fixed
  artifacts from its own package or an asset directory, like the ontology
  heads);
- a state store declared in `IMPL_META` and given to the head by the host is
  planned. It will be an addition, so implementations written now keep working.

## 5. Tests

- Put unit tests in `impls/<slot>/<name>/tests/`. Test the logic you wrote, not
  just the happy path.
- Tests that need a model: skip when the asset is missing (see
  `gaius_ref/tests/conftest.py`).
- Tests that need a real parse: use `nam_heads.sdk.testing.parser_or_skip()`.
- `pytest` must pass with no models and no NAM. CI runs it that way.

## 6. Getting it into NAM

1. Open a PR here. Once it is merged, the maintainers tag a release.
2. NAM bumps its pin to that tag and builds images. A model the head loads must
   also be placed in the images (or a volume) at the asset's path.
3. On a cluster, select it in the head-slot manifest (`/admin/heads`, or
   `PUT /v1/admin/head-slots/<slot>`). Workers swap within seconds.

Changing which class a record gets moves it to another partition. Records
loaded before a swap keep their old keys until they are re-ingested.
