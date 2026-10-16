"""The SDK's own behaviour: host hooks, IMPL_META reading, and that the
conformance checks catch a head that breaks the contract."""

import textwrap

import pytest

from nam_heads.ontology import bindings as B
from nam_heads.sdk import EncoderHead, EncodingResult, SemanticAddress, host
from nam_heads.sdk import testing as T
from nam_heads.sdk.impl_meta import read_impl_meta


# ---- host hooks ---------------------------------------------------------------------

@pytest.fixture
def no_host():
    saved = (host._parser_factory, host._axis_encoder_factory, host._ort_options)
    host.set_parser_factory(None)
    host.set_axis_encoder_factory(None)
    host.set_ort_session_options(None)
    yield
    host.set_parser_factory(saved[0])
    host.set_axis_encoder_factory(saved[1])
    host.set_ort_session_options(saved[2])


def test_parser_without_a_host_fails_closed(no_host):
    assert not host.has_parser()
    with pytest.raises(host.HostUnavailable):
        host.parser()


def test_parser_factory_is_built_once(no_host):
    built = []
    host.set_parser_factory(lambda: built.append(1) or object())
    assert host.parser() is host.parser()
    assert built == [1]


def test_axis_encoder_hook(no_host):
    assert host.axis_encoder("attribute") is None
    host.set_axis_encoder_factory(lambda slot: f"enc:{slot}")
    assert host.axis_encoder("attribute") == "enc:attribute"


def test_ort_options_fallback_caps_threads(no_host):
    pytest.importorskip("onnxruntime")
    assert host.ort_session_options(max_threads=1).intra_op_num_threads == 1
    host.set_ort_session_options(lambda max_threads=None: ("host", max_threads))
    assert host.ort_session_options(3) == ("host", 3)


# ---- IMPL_META ----------------------------------------------------------------------

def _head(tmp_path, slot, name, body):
    d = tmp_path / slot / name
    d.mkdir(parents=True)
    (d / "head.py").write_text(textwrap.dedent(body))
    return d / "head.py"


def test_impl_meta_well_formed(tmp_path):
    h = _head(tmp_path, "ontology", "toy", """
        V = "1.0"
        IMPL_META = {"slot": "ontology", "name": "toy", "version": V, "contract": "ontology-v1"}
    """)
    meta, problems = read_impl_meta(h)
    assert problems == [] and meta["version"] == "1.0"


@pytest.mark.parametrize("meta, why", [
    ('{"slot": "ontology", "name": "other", "version": "1", "contract": "ontology-v1"}', "directory"),
    ('{"slot": "ontology", "name": "toy", "version": "1", "contract": "entity-v1"}', "contract"),
    ('{"slot": "ontology", "name": "toy", "contract": "ontology-v1"}', "version"),
    ('{"slot": "ontology", "name": "toy", "version": "1", "contract": "ontology-v1", "requires": [f()]}',
     "not static"),
    ('{"slot": "ontology", "name": "toy", "version": "1", "contract": "ontology-v1",'
     ' "ontology": {"name": "x1", "artifact": "missing.json"}}', "not in the package"),
    ('{"slot": "ontology", "name": "toy", "version": "1", "contract": "ontology-v1",'
     ' "ontology": {"name": "x1", "artifact": "../x.json"}}', "path in the package"),
])
def test_impl_meta_problems(tmp_path, meta, why):
    h = _head(tmp_path, "ontology", "toy", f"IMPL_META = {meta}\n")
    _, problems = read_impl_meta(h)
    assert problems and why in " ".join(problems)


# ---- conformance checks fire --------------------------------------------------------

class _Head(EncoderHead):
    concept, emit, result_type = B.UNCLASSIFIED, (B.UNCLASSIFIED,), EncodingResult

    def _initialize(self):
        pass

    def _res(self):
        if self.result_type is not EncodingResult:
            return {"addresses": []}
        addrs = [SemanticAddress("ontology", self.concept, 0.5)] if self.concept else []
        return EncodingResult("ontology", addrs, {"sentence_ontologies": {}})

    def encode(self, text, context=None):
        return self._res()

    def encode_query(self, query, context=None):
        return self._res()

    def classify_nearest(self, text, spacy_doc=None):
        return None

    def emittable_classes(self):
        return list(self.emit)


def _problems(**attrs):
    head = type("H", (_Head,), attrs)(head_type="ontology")
    return T.check_ontology_head(head, texts=["x"], queries=["y"], parse=None)


def test_conforming_head_has_no_problems(no_host):
    assert _problems() == []


@pytest.mark.parametrize("attrs, why", [
    ({"concept": "ex:not_a_class"}, "not in the registry"),
    ({"concept": None}, "no addresses"),
    ({"emit": ("ex:nope",)}, "emittable class"),
    ({"emit": ()}, "is empty"),
    ({"result_type": dict}, "not EncodingResult"),
])
def test_contract_violations_are_reported(no_host, attrs, why):
    problems = _problems(**attrs)
    assert problems and any(why in p for p in problems), problems
