"""Per-sentence ontology, computed the same way at ingest and at query.

The ontology head publishes each sentence's class in
``ontology_result.metadata["sentence_ontologies"]`` (with
``sentence_confidences``; abstaining sentences are absent) on both its
ingest and query paths. Heads steer by it (affordance per-sentence profiles
and category scope, attribute evidence, context dimensions). The ontology
stage (NAM: services/nats_worker.py run_ontology) and the query path
(NAM: MultiHeadEncoder.encode_query) call ``attach_sentence_ontologies`` so an
ontology encoder that does not publish the map itself still gets one from
``classify_nearest``, identically at ingest and at query.
"""

from typing import Any, Dict


def attach_sentence_ontologies(ont_encoder: Any, doc: Any, result: Any) -> Dict[int, str]:
    """Ensure ``result`` carries the sentence index -> class map.

    Keeps a map the head already published (it pairs with
    ``sentence_confidences``); otherwise classifies each sentence of ``doc``
    with ``classify_nearest``. Returns the map. No-op when the encoder cannot
    classify sentences, the doc has no sentences, or the result has no
    metadata dict.
    """
    if result is None or getattr(result, "metadata", None) is None:
        return {}
    if "sentence_ontologies" in result.metadata:
        return result.metadata["sentence_ontologies"]
    if doc is None:
        return {}
    classify = getattr(ont_encoder, "classify_nearest", None)
    sents = getattr(doc, "sents", None)
    if classify is None or not sents:
        return {}
    out: Dict[int, str] = {}
    for si, span in enumerate(sents):
        nearest = classify(span.text, spacy_doc=span)
        if nearest:
            out[si] = nearest
    if out:
        result.metadata["sentence_ontologies"] = out
    return out


# Short name shared with the attribute head.
attach = attach_sentence_ontologies

# Older name, kept as an alias (same function, same arguments).
annotate_sentence_ontologies = attach_sentence_ontologies
