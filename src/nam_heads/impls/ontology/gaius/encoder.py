"""ColBERT-Zero late-interaction encoder (onnxruntime, fp32) and MaxSim.

ColBERT-Zero (``lightonai/ColBERT-Zero``, ModernBERT-base + a 768->128
projection; the copy in ``docker/assets/colbert-model``) is the encoder gaius
uses (``engine/embeddings/colbert.py``): one L2-normalised 128-d vector per
token, and a text's similarity to a class is MaxSim -- each query token's best
match among the class's tokens, averaged over the query tokens
(``tbox_maxsim.maxsim_tbox_topk`` divides the summed MaxSim by the query
token count).

Runtime is onnxruntime, not torch: the nam worker images carry onnxruntime
and no torch. ``export`` (dev time, needs torch + transformers) writes the
fp32 graph and its provenance to a model directory; the head only reads it.
Precision is pinned: the graph is exported fp32 and nothing quantises it.

Tokenisation reproduces PyLate's ``ColBERT.tokenize`` for this model's
config (``config_sentence_transformers.json``): the prompt
(``search_query: `` / ``search_document: ``) is prepended to the text,
the text is truncated to ``length - 1`` tokens including [CLS]/[SEP], the
marker token ([Q] / [D]) is inserted after [CLS], queries keep every token
(the model disables query expansion), documents drop punctuation tokens
(the skiplist). ``tools/export_onnx.py --check`` compares the result with
PyLate token by token.

Each text runs alone (batch 1, no padding), so a text's vectors do not depend
on what else is encoded: same text, same vectors.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

MODEL_NAME = "lightonai/ColBERT-Zero"
ONNX_FILE = "model.onnx"
EXPORT_META = "export.json"
ENV_MODEL_DIR = "NAM_GAIUS_MODEL_DIR"
# Where the exported model is looked for after the env/config (NAM's images).
DEFAULT_DIRS = (
    Path("/app/colbert-zero-onnx"),
)
PRECISION = "fp32"


class EncoderError(RuntimeError):
    """The exported model is missing or not the one this head was calibrated on."""


def resolve_model_dir(explicit: Optional[str] = None) -> Path:
    for cand in ([explicit] if explicit else []) + [os.environ.get(ENV_MODEL_DIR)] + list(DEFAULT_DIRS):
        if cand and (Path(cand) / ONNX_FILE).is_file():
            return Path(cand)
    raise EncoderError(
        f"ColBERT-Zero ONNX model not found (config model_dir, ${ENV_MODEL_DIR}, "
        f"{', '.join(str(d) for d in DEFAULT_DIRS)}). Build it with "
        "`python -m nam_heads.impls.ontology.gaius.tools.export_onnx`.")


class LateInteractionEncoder:
    """Token vectors for queries and documents from the exported model."""

    def __init__(self, model_dir: Optional[str] = None, max_threads: Optional[int] = None):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        from nam_heads.sdk.host import ort_session_options

        self.model_dir = resolve_model_dir(model_dir)
        self.meta: Dict = json.loads((self.model_dir / EXPORT_META).read_text())
        if self.meta.get("precision") != PRECISION:
            raise EncoderError(f"model precision {self.meta.get('precision')!r} != pinned {PRECISION!r}")
        st = json.loads((self.model_dir / "config_sentence_transformers.json").read_text())
        if st.get("do_query_expansion"):
            raise EncoderError("query expansion is not implemented; this model must disable it")
        self.prompts = st["prompts"]
        self.query_length = int(st["query_length"])
        self.document_length = int(st["document_length"])
        tok = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        tok.no_padding()
        self._tok = tok
        # truncation is tokenizer state: one caller at a time (the query
        # server encodes from several threads)
        self._tok_lock = threading.Lock()
        self.q_marker = tok.token_to_id(st["query_prefix"])
        self.d_marker = tok.token_to_id(st["document_prefix"])
        if self.q_marker is None or self.d_marker is None:
            raise EncoderError("query/document marker tokens missing from the tokenizer")
        self.skiplist = frozenset(i for i in (tok.token_to_id(w) for w in st["skiplist_words"])
                                  if i is not None)
        opts = ort_session_options(max_threads=max_threads)
        self._session = ort.InferenceSession(str(self.model_dir / ONNX_FILE), sess_options=opts,
                                             providers=["CPUExecutionProvider"])

    # ---- tokenisation -----------------------------------------------------
    def _encode_ids(self, text: str, is_query: bool, max_length: Optional[int]):
        length = int(max_length or (self.query_length if is_query else self.document_length))
        prompt = self.prompts["query" if is_query else "document"]
        with self._tok_lock:
            self._tok.enable_truncation(max_length=length - 1)
            enc = self._tok.encode((prompt + text).strip())
        marker = self.q_marker if is_query else self.d_marker
        ids = list(enc.ids)
        return ids[:1] + [marker] + ids[1:], prompt, enc

    def token_ids(self, text: str, is_query: bool, max_length: Optional[int] = None) -> List[int]:
        return self._encode_ids(text, is_query, max_length)[0]

    def query_char_spans(self, text: str, max_length: Optional[int] = None) -> List[tuple]:
        """Per query token, its (start, end) characters in ``text``; (0, 0) for
        [CLS], the marker, the prompt and [SEP]."""
        text = text.strip()
        ids, prompt, enc = self._encode_ids(text, True, max_length)
        n = len(prompt)
        # byte-level offsets include the token's leading space, so a text
        # token can start inside the prompt's trailing space
        spans = [(max(0, a - n), b - n) if b > n else (0, 0) for a, b in enc.offsets]
        spans = [(0, 0) if (enc.special_tokens_mask[i]) else s for i, s in enumerate(spans)]
        return spans[:1] + [(0, 0)] + spans[1:]

    # ---- encoding ---------------------------------------------------------
    def _run(self, ids: List[int]) -> np.ndarray:
        a = np.asarray([ids], dtype=np.int64)
        out = self._session.run(None, {"input_ids": a, "attention_mask": np.ones_like(a)})[0][0]
        return np.ascontiguousarray(out, dtype=np.float32)

    def encode_query(self, text: str, max_length: Optional[int] = None) -> np.ndarray:
        """(n, 128) unit token vectors of ``text`` as a query (all tokens)."""
        return self._run(self.token_ids(text, True, max_length))

    def encode_document(self, text: str) -> np.ndarray:
        """(m, 128) unit token vectors of ``text`` as a document (punctuation dropped)."""
        ids = self.token_ids(text, False)
        vecs = self._run(ids)
        keep = [i for i, t in enumerate(ids) if t not in self.skiplist]
        return vecs[keep]


def maxsim(q: np.ndarray, d: np.ndarray) -> float:
    """Mean over query tokens of the best document-token cosine."""
    return float((q @ d.T).max(axis=1).mean())


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()
