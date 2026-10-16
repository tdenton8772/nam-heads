"""Export ColBERT-Zero to the fp32 ONNX model the gaius head runs.

    python -m nam_heads.impls.ontology.gaius.tools.export_onnx \\
        --src docker/assets/colbert-model --out docker/assets/colbert-zero-onnx [--check]

Dev time only (needs torch, transformers, safetensors; ``--check`` also
pylate). Writes ``model.onnx`` (ModernBERT + the 768->128 projection + L2
normalisation, fp32), copies the tokenizer and the sentence-transformers
config, and records provenance in ``export.json``. ``--check`` encodes probe
texts with PyLate and with the exported model through the head's own
tokenisation and fails on any token-id mismatch or a vector difference above
1e-4.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from ..encoder import EXPORT_META, MODEL_NAME, ONNX_FILE, PRECISION, LateInteractionEncoder, file_sha256

COPY = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "config_sentence_transformers.json")
CHECK_TEXTS = (
    ("Albert Einstein. Albert Einstein was a German-born theoretical physicist who is widely held "
     "to be one of the greatest scientists of all time.", True),
    ("List of Polydor Records artists. This is a list of artists who have recorded for Polydor Records.", True),
    ("who won the nobel prize in physics", True),
    ("Person (Human). An Animal that is a member of the species Homo sapiens. A kind of Animal.", False),
    ("process (event). p is a process means p is an occurrent that has some temporal proper part. "
     "Examples: An act of selling; the life of an organism.", False),
)


def export(src: Path, out: Path) -> dict:
    import torch
    import transformers
    from safetensors.torch import load_file
    from transformers import AutoModel

    torch.set_num_threads(1)

    class ColbertZero(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bert = AutoModel.from_pretrained(str(src), attn_implementation="eager",
                                                  dtype=torch.float32)
            w = load_file(str(src / "1_Dense" / "model.safetensors"))
            (weight,) = w.values()
            self.dense = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
            self.dense.weight.data = weight.float()

        def forward(self, input_ids, attention_mask):
            h = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            return torch.nn.functional.normalize(self.dense(h), p=2, dim=-1)

    out.mkdir(parents=True, exist_ok=True)
    model = ColbertZero().eval()
    ids = torch.tensor([[50281, 50368, 8716, 64, 7267, 27, 18252, 21003, 15, 50282]])
    torch.onnx.export(model, (ids, torch.ones_like(ids)), str(out / ONNX_FILE),
                      input_names=["input_ids", "attention_mask"], output_names=["token_embeddings"],
                      dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                                    "attention_mask": {0: "batch", 1: "seq"},
                                    "token_embeddings": {0: "batch", 1: "seq"}},
                      opset_version=17, do_constant_folding=True, dynamo=False)
    for name in COPY:
        shutil.copy2(src / name, out / name)
    meta = {
        "model": MODEL_NAME,
        "precision": PRECISION,
        "source": {"model.safetensors": file_sha256(src / "model.safetensors"),
                   "1_Dense/model.safetensors": file_sha256(src / "1_Dense" / "model.safetensors"),
                   "tokenizer.json": file_sha256(src / "tokenizer.json")},
        "onnx_sha256": file_sha256(out / ONNX_FILE),
        "opset": 17,
        "exporter": {"torch": torch.__version__, "transformers": transformers.__version__},
    }
    (out / EXPORT_META).write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    return meta


def check(src: Path, out: Path) -> float:
    """Max |vector difference| between PyLate and the exported model."""
    import numpy as np
    import torch
    from pylate import models

    torch.set_num_threads(1)
    ref = models.ColBERT(model_name_or_path=str(src), device="cpu")
    enc = LateInteractionEncoder(str(out), max_threads=1)
    worst = 0.0
    for text, is_query in CHECK_TEXTS:
        feats = ref.tokenize([ref.prompts["query" if is_query else "document"] + text], is_query=is_query)
        want_ids = feats["input_ids"][0].tolist()
        got_ids = enc.token_ids(text, is_query)
        if want_ids != got_ids:
            raise SystemExit(f"token ids differ for {text[:40]!r}:\n {want_ids}\n {got_ids}")
        want = ref.encode([text], is_query=is_query, prompt_name="query" if is_query else "document",
                          show_progress_bar=False)[0]
        got = enc.encode_query(text) if is_query else enc.encode_document(text)
        if want.shape != got.shape:
            raise SystemExit(f"shape {got.shape} != pylate {want.shape} for {text[:40]!r}")
        worst = max(worst, float(np.abs(np.asarray(want) - got).max()))
    if worst > 1e-4:
        raise SystemExit(f"exported model differs from PyLate by {worst}")
    return worst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="ColBERT-Zero checkout (docker/assets/colbert-model)")
    ap.add_argument("--out", required=True, help="output model dir (docker/assets/colbert-zero-onnx)")
    ap.add_argument("--check", action="store_true", help="compare with PyLate after export")
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args(argv)
    src, out = Path(a.src), Path(a.out)
    if not a.check_only:
        print(json.dumps(export(src, out), indent=1))
    if a.check or a.check_only:
        print(f"pylate parity: max |diff| = {check(src, out):.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
