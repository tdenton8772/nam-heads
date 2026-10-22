# Context Encoder Head

Answers: **"Under what circumstances does this memory apply?"**

One head type (`context`), two extractions, chosen by `NAM_CONTEXT_HEAD` (or config `head`):

- **`particular`** (default) -- the particular-evidence extractor, `src/context_axis/`
  (`ParticularContextEncoder`, `context_axis/encoder.py`): temporal paths (`t:C19/D1860/Y1865`),
  gazetteer places (`g:NA/US/US-IL/...`) with their containers, institutions (`i:<org>`) and
  fields (`d:<field>`). Each value carries the sentences it came from; the bundler scopes values
  to entities. It reads and grows the context store, relayed to LMDB by nats_worker.
- **`legacy`** -- the NER-driven extraction in `head.py`, kept for rollback: four fixed
  dimensions (`temporal`, `geographic`, `domain`, `institutional`) normalized through
  store-backed lookups, one address per dimension with signal, else `__null__`.

Both read the record's parse, the rule backend's ParsedDocument in `context["spacy_doc"]` (the
head SDK's key name). A record that arrives without one is parsed with the same rule parser
(`src/encoders/record_parse.py` for legacy; `ParticularContextEncoder._doc` for particular).

## Configuration

- `head`: `particular` or `legacy` (env `NAM_CONTEXT_HEAD`, default `particular`).
- `context_store_backend` (env `NAM_CONTEXT_STORE_BACKEND`, default `couchbase_kv`),
  `context_store_fallback` (`memory`), `context_store_seed`.
- `dimensions`, `max_per_dimension` (default 1), `split_fragments` (default true),
  `default_null` (default `__null__`).
- particular only: `sentence_steering` (default off), `learn` (default on), `min_witnesses`.

```python
from src.encoders.heads.context.head import ContextEncoder

enc = ContextEncoder(head_type="context", config={"context_store_backend": "memory"})
result = enc.encode("Lincoln was born in Kentucky in 1809.")
for addr in result.addresses:
    print(addr.concept, addr.metadata["dimension"], addr.metadata.get("sentences"))
```

## Output

Each `SemanticAddress` has `head_type="context"`, `concept` (the value or `__null__`) and
`metadata`:

- particular: `dimension`, `raw` (surface), `source`, `sentences`, `spans`,
  `max_per_dimension`, and when present `kind`, `containers`, `demonym_spans`.
- legacy: `dimension`, `raw`, `source` (`ner_<label>`, `ontology_ner`, `keyword_fallback` or `null`).

Tests: `src/context_axis/tests/` (particular), `heads/context/tests/` (head selection, the
legacy extraction, runtime shape).
