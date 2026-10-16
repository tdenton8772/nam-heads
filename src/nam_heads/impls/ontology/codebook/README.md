# Ontology head (BFO 2020 / CCO v2.2)

Types text with a class from the certified BFO/CCO TBox; that class is the
address key's type segment. Design, training and measurements:
`docs/plans/ontology-bfo-cco.md`.

- `head.py` -- `OntologyEncoder` (encode, encode_query, classify_nearest, emittable_classes)
- `features.py` -- sentence cues -> hashed, L2-normalised vector
- `codebook.py` -- artifact load (strict), scoring, back-off, abstain
- `diagnostics.py` -- per-encode entropy/alerts and the rolling readout in `get_metrics()`
- `artifacts/cco_codebook.{json,npz}` -- the trained codebook
- `training/` -- labels (`cco_anchors.py`, `cco_dataset.py`), `cco_train.py`, `cco_baseline.py`

## Diagnostics

Every `encode` / `encode_query` result carries `metadata["diagnostics"]`
(`entropy`, `max_prob`, `active_count`, `alerts`), computed from the calibrated
softmax over the K trained classes that decided the record. Entropy is
normalized by log2(K) (0 = one-hot, 1 = uniform). Alerts:
`ontology_unclassified` (emitted bfo:entity, or top probability <
`unknown_threshold`) and `ontology_overactivated` (entropy >
`collapse_entropy_threshold` with >= `min_active_for_collapse` classes at
>= `active_floor`). They never change the emitted class (tested byte-for-byte
with diagnostics on and off).

| Config key | Env override | Default |
|---|---|---|
| `enable_diagnostics` | `NAM_ONTOLOGY_DIAGNOSTICS` (`0` = off) | on |
| `unknown_threshold` | `NAM_ONTOLOGY_UNKNOWN_THRESHOLD` | 0.25 |
| `collapse_entropy_threshold` | `NAM_ONTOLOGY_COLLAPSE_ENTROPY` | 0.80 |
| `min_active_for_collapse` | `NAM_ONTOLOGY_MIN_ACTIVE` | 5 |
| `active_floor` | `NAM_ONTOLOGY_ACTIVE_FLOOR` | 0.02 |
| `diagnostics_window` | `NAM_ONTOLOGY_DIAG_WINDOW` | 1000 |
| `diagnostics_window_seconds` | `NAM_ONTOLOGY_DIAG_WINDOW_SECONDS` | 300 (window = last N encodes or last S seconds, whichever holds fewer; 0 = count only) |

Head status comes from the window's aggregate alert RATES (share of recent encodes that raised
each alert), not from single records: `NAM_ONTOLOGY_RATE_UNCLASSIFIED_AMBER` / `_RED` and
`NAM_ONTOLOGY_RATE_OVERACTIVATED_AMBER` / `_RED` (defaults 0.25 / 0.50). The snapshot carries
`health: {status, rates: {unclassified|overactivated: {rate, amber, red, status}}}`; merged
worker snapshots recompute it from the merged rates.

`get_metrics()["diagnostics"]` is the live readout: lifetime `count` and
`alerts_total`, and over the last `diagnostics_window` encodes `window`:
`entropy_mean`, `max_prob_mean`, `entropy_hist` (10 bins over [0, 1]),
`unclassified_rate`, `overactivated_rate`, `alert_counts`, `class_counts`,
`distinct_classes`, `class_entropy` (emitted-class spread over log2 of the
emittable class count), `top1_class`, `top1_share`. Flat
`ontology_entropy_mean`, `ontology_unclassified_rate`,
`ontology_overactivated_rate`, `ontology_class_entropy`, `ontology_top1_share`
repeat the headline numbers so the heartbeat gauges give them history.

Path to the dashboard: the ontology NATS stage's heartbeat writes
`get_metrics()` into `head_reg::<pod>::ontology` in the metrics bucket
(NAM: `nam/services/nats_worker.py`, `_heartbeat_loop`); nam-query merges the
workers' readouts (`diagnostics.merge_snapshots`) into
`encoder_heads.ontology.diagnostics` of `GET /v1/admin/system-status` and
`heads.ontology.diagnostics` of `GET /v1/admin/encoder-heads`, whose
`instances[].metrics.diagnostics` keeps each worker's own.

## Retrain

```bash
python -m nam_heads.impls.ontology.codebook.training.cco_dataset \
  --corpus $NAM_DATA/wikipedia_abstracts.jsonl \
  --types $NAM_DATA/ontology/wikidata_p31.jsonl \
  --classes $NAM_DATA/ontology/wikidata_classes.jsonl \
  --output $NAM_DATA/ontology/cco_sentences.jsonl
python -m nam_heads.impls.ontology.codebook.training.cco_train \
  --dataset $NAM_DATA/ontology/cco_sentences.jsonl \
  --out nam/nam_heads/impls/ontology/codebook/artifacts
```
