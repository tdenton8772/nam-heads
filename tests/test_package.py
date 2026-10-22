"""Plugin packages: pack, verify, unpack, reproducibility, tamper and path checks, the CLI; the parser-v1
contract and the plugin fields of IMPL_META."""

import io
import json
import tarfile
import textwrap
import zipfile

import pytest

from nam_heads.sdk import package as P
from nam_heads.sdk.impl_meta import SLOT_CONTRACTS, plugin_meta_problems
from nam_heads.sdk.parser import Parser

META = '''IMPL_META = {"slot": "sentiment", "name": "lexicon", "version": "0.3.0", "contract": "facet-v1",
             "description": "toy", "output": "sentiment-v1",
             "resources": {"cpu": 0.5, "memory_mb": 256, "workers": 1},
             "dependencies": ["tinydep==1.0"], "learner": {}}
'''


def _src(tmp_path, meta=META, learner=True):
    src = tmp_path / "lexicon"
    (src / "data").mkdir(parents=True)
    (src / "tests").mkdir()
    (src / "__pycache__").mkdir()
    (src / "head.py").write_text(meta + "from .helpers import X\n")
    (src / "helpers.py").write_text("X = 1\n")
    (src / "data" / "lexicon.json").write_text('{"good": 1}')
    (src / "tests" / "test_x.py").write_text("def test(): pass\n")
    (src / "__pycache__" / "head.cpython-311.pyc").write_bytes(b"\0")
    if learner:
        (src / "learner.py").write_text("")
    return src


def _wheel(tmp_path):
    w = tmp_path / "tinydep-1.0-py3-none-any.whl"
    with zipfile.ZipFile(w, "w") as z:
        z.writestr("tinydep/__init__.py", "VALUE = 42\n")
    return w


def test_pack_verify_unpack(tmp_path):
    out = P.pack(_src(tmp_path), tmp_path / "out", wheels=[_wheel(tmp_path)])
    assert out.name == "sentiment.lexicon-0.3.0.nhp"
    m = P.verify(out)
    assert m["id"] == "sentiment.lexicon@0.3.0" and m["package"] == "lexicon"
    assert sorted(m["files"]) == ["deps/tinydep-1.0-py3-none-any.whl", "pkg/data/lexicon.json",
                                  "pkg/head.py", "pkg/helpers.py", "pkg/learner.py"]   # no tests, no caches
    assert m["meta"]["output"] == "sentiment-v1" and m["meta"]["resources"]["workers"] == 1
    dest = P.unpack(out, tmp_path / "cache")
    assert (dest / "pkg" / "data" / "lexicon.json").read_text() == '{"good": 1}'
    assert (dest / "deps" / "tinydep-1.0-py3-none-any.whl").is_file()
    assert json.loads((dest / "plugin.json").read_text())["id"] == "sentiment.lexicon@0.3.0"


def test_packing_is_reproducible(tmp_path):
    src = _src(tmp_path)
    a = P.pack(src, tmp_path / "a")
    b = P.pack(src, tmp_path / "b")
    assert P.sha256_file(a) == P.sha256_file(b)
    (src / "helpers.py").write_text("X = 2\n")
    assert P.sha256_file(P.pack(src, tmp_path / "c")) != P.sha256_file(a)


def _rewrite(archive, edit):
    """Rebuild ``archive`` with edit(name, data) -> (name, data) or None (drop) applied to each member."""
    members = []
    with tarfile.open(archive, "r:gz") as t:
        for m in t.getmembers():
            members.append((m.name, t.extractfile(m).read()))
    blob = io.BytesIO()
    with tarfile.open(fileobj=blob, mode="w:gz") as t:
        for name, data in members:
            r = edit(name, data)
            if r is None:
                continue
            info = tarfile.TarInfo(r[0])
            info.size = len(r[1])
            t.addfile(info, io.BytesIO(r[1]))
    archive.write_bytes(blob.getvalue())


@pytest.mark.parametrize("edit, why", [
    (lambda n, d: (n, b"X = 666\n") if n == "pkg/helpers.py" else (n, d), "checksum"),
    (lambda n, d: None if n == "pkg/helpers.py" else (n, d), "missing"),
    (lambda n, d: ("../evil.py", d) if n == "pkg/helpers.py" else (n, d), "unsafe path"),
    (lambda n, d: (n, json.dumps(dict(json.loads(d), id="other.x@1")).encode()) if n == "plugin.json"
     else (n, d), "id"),
])
def test_a_tampered_archive_is_refused(tmp_path, edit, why):
    out = P.pack(_src(tmp_path), tmp_path / "out")
    _rewrite(out, edit)
    with pytest.raises(P.PackageError, match=why):
        P.verify(out)
    with pytest.raises(P.PackageError):
        P.unpack(out, tmp_path / "never")
    assert not (tmp_path / "never").exists()


def test_an_extra_member_is_refused(tmp_path):
    out = P.pack(_src(tmp_path), tmp_path / "out")
    _rewrite(out, lambda n, d: (n, d))
    with tarfile.open(out, "r:gz") as t:
        members = [(m.name, t.extractfile(m).read()) for m in t.getmembers()]
    members.append(("pkg/sneaky.py", b"import os\n"))
    blob = io.BytesIO()
    with tarfile.open(fileobj=blob, mode="w:gz") as t:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    out.write_bytes(blob.getvalue())
    with pytest.raises(P.PackageError, match="extra"):
        P.verify(out)


@pytest.mark.parametrize("meta, why", [
    (META.replace('"output": "sentiment-v1",', ""), "output is required"),
    (META.replace('"sentiment-v1"', '"Sentiment V1"'), "output must match"),
    (META.replace('"cpu": 0.5', '"cpu": 0'), "positive number"),
    (META.replace('"workers": 1', '"workers": 1, "gpus": 1'), "unknown key"),
    (META.replace('["tinydep==1.0"]', '"tinydep"'), "dependencies"),
    (META.replace('"facet-v1"', '"attribute-v1"'), "contract"),
])
def test_pack_refuses_a_bad_declaration(tmp_path, meta, why):
    with pytest.raises(P.PackageError, match=why):
        P.pack(_src(tmp_path, meta), tmp_path / "out")


def test_pack_refuses_a_declared_learner_without_its_module(tmp_path):
    with pytest.raises(P.PackageError, match="learner module learner.py"):
        P.pack(_src(tmp_path, learner=False), tmp_path / "out")


def test_cli(tmp_path, capsys):
    src = _src(tmp_path)
    assert P.main(["pack", str(src), "-o", str(tmp_path / "out")]) == 0
    archive = tmp_path / "out" / "sentiment.lexicon-0.3.0.nhp"
    assert P.main(["verify", str(archive)]) == 0
    assert P.main(["info", str(archive)]) == 0
    assert '"sentiment.lexicon@0.3.0"' in capsys.readouterr().out
    assert P.main(["verify", str(tmp_path / "missing.nhp")]) == 1


def test_plugin_fields_and_the_parser_contract():
    assert plugin_meta_problems({}) == [] and plugin_meta_problems({}, require_output=True)
    assert SLOT_CONTRACTS["nlp"] == "parser-v1"

    class Words(Parser):
        def parse(self, text):
            return text.split()

    p = Words({"x": 1})
    assert p.config == {"x": 1} and p.parse_batch(["a b", "c"]) == [["a", "b"], ["c"]] and p.fingerprint() == ""
    with pytest.raises(TypeError):
        Parser()
