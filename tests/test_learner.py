"""Head learning: IMPL_META "learner", the parked result, the HeadLearner defaults, loading a learner
part from a package, and the conformance check."""

import json
import sys
import textwrap

import pytest

from nam_heads.sdk import EncoderHead, EncodingResult, HeadLearner, parked_result, parked_values
from nam_heads.sdk import testing as T
from nam_heads.sdk.impl_meta import InstalledImpl, read_impl_meta
from nam_heads.sdk.learner import (LEAVE_OUT, LearnerSpec, LearnerSpecError, load_learner_class,
                                   parse_learner)
from nam_heads.sdk.store import MemoryStore, StoreSpec

GPU = {"image": "me/learner", "model_id": "toy-v1", "port": 50051, "disk_gb": 20,
       "gpu_types": ["NVIDIA GeForce RTX 4090"]}


# ---- IMPL_META "learner" ---------------------------------------------------------------------

def test_parse_learner_defaults_and_absence():
    assert parse_learner(None) is None
    assert parse_learner({}) == LearnerSpec(module="learner", version="", gpu=None)
    spec = parse_learner({"module": "learn2", "version": "0.2", "gpu": GPU})
    assert (spec.module, spec.version, spec.gpu) == ("learn2", "0.2", GPU)


@pytest.mark.parametrize("raw, why", [
    ("learner", "must be an object"),
    ({"modul": "x"}, "unknown keys"),
    ({"module": "../x"}, "module must match"),
    ({"version": 2}, "version must be a string"),
    ({"gpu": {"image": "x"}}, "'model_id'"),
    ({"gpu": dict(GPU, port=0)}, "'port'"),
    ({"gpu": dict(GPU, port=True)}, "'port'"),
    ({"gpu": dict(GPU, gpu_types="RTX")}, "'gpu_types'"),
    ({"gpu": dict(GPU, env={"A": 1})}, "'env'"),
    ({"gpu": dict(GPU, vram=24)}, "unknown keys"),
])
def test_parse_learner_refuses_a_malformed_declaration(raw, why):
    with pytest.raises(LearnerSpecError, match=why):
        parse_learner(raw)


def _impl(tmp_path, meta_src: str, learner_src: str = None, slot="attribute", name="toy"):
    d = tmp_path / slot / name
    d.mkdir(parents=True)
    (d / "head.py").write_text(textwrap.dedent(meta_src))
    if learner_src is not None:
        (d / "learner.py").write_text(textwrap.dedent(learner_src))
    return d / "head.py"


def test_impl_meta_learner_problems(tmp_path):
    base = ('IMPL_META = {"slot": "attribute", "name": "toy", "version": "1", '
            '"contract": "attribute-v1", %s}\n')
    ok = _impl(tmp_path / "a", base % '"learner": {"gpu": {"image": "i", "model_id": "m"}}', "")
    assert read_impl_meta(ok)[1] == []
    missing = _impl(tmp_path / "b", base % '"learner": {"module": "teacher"}', "")
    assert read_impl_meta(missing)[1] == ["IMPL_META learner module teacher.py is not in the package"]
    bad = _impl(tmp_path / "c", base % '"learner": {"gpu": {"image": "i"}}', "")
    assert any("'model_id'" in p for p in read_impl_meta(bad)[1])
    dynamic = _impl(tmp_path / "d", "import os\n" + base % '"learner": {"module": os.getenv("X")}', "")
    assert read_impl_meta(dynamic)[1] == [
        "IMPL_META learner is not static (literals and module constants only)"]
    none = _impl(tmp_path / "e", base % '"description": "no learner"')
    assert read_impl_meta(none)[1] == []


# ---- the live part's side: parked results ------------------------------------------------------

def test_parked_result_carries_the_values_and_no_addresses():
    r = parked_result("affordance", [{"value": "flarb", "token": "flarbed", "sentence": "It flarbed."}],
                      mode="ingest")
    assert isinstance(r, EncodingResult) and r.addresses == []
    assert r.metadata["mode"] == "ingest"
    assert parked_values(r) == [{"value": "flarb", "token": "flarbed", "sentence": "It flarbed."}]
    assert parked_values({"metadata": r.metadata}) == parked_values(r)
    assert parked_values(EncodingResult("affordance", [], {})) is None
    assert parked_values({"metadata": None}) is None


@pytest.mark.parametrize("bad", [[], [{"token": "x"}], [{"value": ""}], [{"value": "x", "obj": object()}]])
def test_parked_result_refuses_what_the_host_cannot_carry(bad):
    with pytest.raises((ValueError, TypeError)):
        parked_result("affordance", bad)


def test_a_head_does_not_park_until_the_host_says_so():
    class H(EncoderHead):
        def _initialize(self):
            pass

        def encode(self, text, context=None):
            return EncodingResult(self.head_type, [], {})

        encode_query = encode

    h = H(head_type="attribute")
    assert h.parking is False
    h.parking = True
    assert H(head_type="attribute").parking is False       # an instance flag, never the class's


# ---- the learner part --------------------------------------------------------------------------

class Toy(HeadLearner):
    """A CPU learner: a value is a colour when its record says so."""

    def decision_key(self, value):
        return f"learned:{value}"

    def build_item(self, item_id, value, occurrence, record, decided):
        text = record.get("text") or occurrence.get("sentence") or ""
        if value not in text:
            return None
        return {"item": item_id, "value": value, "text": text}

    def decide_local(self, items):
        return [{"colour": "colour" in it["text"]} for it in items]

    def decision(self, value, item, answer, decided):
        if not isinstance(answer, dict) or "colour" not in answer:
            return self.leave_out(value, "no answer")
        return {"kind": "colour" if answer["colour"] else "other", "value": value}


def _store():
    return MemoryStore(StoreSpec(name="toy"))


def test_pending_reads_the_store_once_and_splits_the_values():
    st = _store()
    st.put_doc("learned:red", {"kind": "colour"})
    toy = Toy("attribute", st)
    reads = st.reads
    todo, decided = toy.pending(["red", "blue", "green"])
    assert st.reads == reads + 1
    assert todo == ["blue", "green"] and decided == {"red": {"kind": "colour"}}
    assert toy.pending([]) == ([], {})


def test_store_writes_round_trip_through_pending():
    st = _store()
    toy = Toy("attribute", st)
    st.put_many(toy.store_writes({"blue": {"kind": "colour"}, "rock": toy.leave_out("rock", "x")}))
    todo, decided = toy.pending(["blue", "rock"])
    assert todo == [] and decided["rock"]["kind"] == LEAVE_OUT and decided["blue"] == {"kind": "colour"}


def test_a_learner_without_gpu_runs_locally_and_one_with_gpu_takes_config_overrides():
    assert Toy("attribute", _store()).gpu_workload() is None

    class NoLocal(Toy):
        decide_local = HeadLearner.decide_local

    with pytest.raises(NotImplementedError):
        NoLocal("attribute", _store()).decide_local([])
    gpu = Toy("attribute", _store(), {"gpu": GPU}).gpu_workload()
    assert gpu == GPU
    with pytest.raises(LearnerSpecError):
        Toy("attribute", _store(), {"gpu": {"image": "only"}}).gpu_workload()


def test_load_learner_class_and_declared_gpu(tmp_path, monkeypatch):
    pkg = tmp_path / "toyimpls" / "attribute" / "toy"
    pkg.mkdir(parents=True)
    for d in (tmp_path / "toyimpls", tmp_path / "toyimpls" / "attribute", pkg):
        (d / "__init__.py").write_text("")
    (pkg / "head.py").write_text(textwrap.dedent('''
        IMPL_META = {"slot": "attribute", "name": "toy", "version": "1", "contract": "attribute-v1",
                     "learner": {"gpu": {"image": "me/learner", "model_id": "toy-v1"}}}
    '''))
    (pkg / "learner.py").write_text(textwrap.dedent('''
        from nam_heads.sdk.learner import HeadLearner

        class ToyLearner(HeadLearner):
            def decision_key(self, value):
                return value
            def build_item(self, item_id, value, occurrence, record, decided):
                return {"item": item_id}
            def decision(self, value, item, answer, decided):
                return {"kind": "x"}
    '''))
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        cls = load_learner_class("toyimpls.attribute.toy", LearnerSpec())
        assert cls.__name__ == "ToyLearner"
        learner = cls("attribute", _store(), {"gpu": {"image": "me/learner:v2"}})
        assert learner.gpu_workload() == {"image": "me/learner:v2", "model_id": "toy-v1"}
        impl = InstalledImpl("attribute", "toy", "toyimpls.attribute.toy", pkg / "head.py")
        built = T.build_learner(impl)
        assert type(built).__name__ == "ToyLearner" and built.slot == "attribute"
        (pkg / "learner2.py").write_text("x = 1\n")
        with pytest.raises(RuntimeError, match="expected one HeadLearner subclass"):
            load_learner_class("toyimpls.attribute.toy", LearnerSpec(module="learner2"))
    finally:
        for m in [m for m in sys.modules if m.startswith("toyimpls")]:
            del sys.modules[m]


# ---- conformance --------------------------------------------------------------------------------

CASES = [{"value": "teal", "occurrence": {"sentence": "teal"}, "record": {"key": "k1", "title": "Teal",
                                                                        "text": "teal is a colour"}},
         {"value": "granite", "occurrence": {}, "record": {"key": "k2", "title": "Granite",
                                                          "text": "granite is a rock"}},
         {"value": "nowhere", "occurrence": {}, "record": {"key": "k3", "title": "", "text": "absent"}}]


def test_check_learner_passes_a_conforming_learner():
    assert T.check_learner(Toy("attribute", _store()), CASES) == []


def test_check_learner_reports_a_learner_that_raises_or_forgets():
    class Raises(Toy):
        def decision(self, value, item, answer, decided):
            return {"kind": "colour" if answer["colour"] else "other"}

    probs = T.check_learner(Raises("attribute", _store()), CASES)
    assert any("raised" in p for p in probs)

    class Forgets(Toy):
        def store_writes(self, decisions):
            return [(f"elsewhere:{v}", json.dumps(d).encode()) for v, d in decisions.items()]

    probs = T.check_learner(Forgets("attribute", _store()), CASES)
    assert any("not read back" in p for p in probs)

    class WrongId(Toy):
        def build_item(self, item_id, value, occurrence, record, decided):
            it = super().build_item(item_id, value, occurrence, record, decided)
            return dict(it, item="other") if it else None

    assert any("must carry 'item'" in p for p in T.check_learner(WrongId("attribute", _store()), CASES))
    assert T.check_learner(object(), CASES) == ["object is not a HeadLearner"]
