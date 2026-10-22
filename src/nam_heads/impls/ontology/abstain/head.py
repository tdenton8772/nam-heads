"""Ontology implementation that types every record bfo:entity.

The control for the ontology slot. Selecting it shows what the ontology head
contributes: every record lands in the bfo:entity partition, as when the
codebook head abstains. It loads no model, so it also exercises a slot swap on
any image.
"""

from typing import Dict, List, Optional

from nam_heads.sdk import EncoderHead, EncodingResult, SemanticAddress
from nam_heads.ontology.bindings import UNCLASSIFIED

IMPL_META = {
    "slot": "ontology",
    "name": "abstain",
    "version": "1.0.0",
    "contract": "ontology-v1",
    "output": "bfo-cco-2.2",
    "resources": {"cpu": 0.25, "memory_mb": 256, "workers": 1},
    "dependencies": [],
    "description": "Types every record bfo:entity: the control for what the ontology head contributes.",
    "requires": [],
}


class AbstainOntologyEncoder(EncoderHead):
    def _initialize(self):
        pass

    def _encode(self, mode: str) -> EncodingResult:
        return EncodingResult(
            head_type=self.head_type,
            addresses=[SemanticAddress(head_type=self.head_type, concept=UNCLASSIFIED,
                                       confidence=0.0,
                                       metadata={"method": "abstain", "abstained": "control"})],
            # An empty sentence map: no sentence votes, and the NATS stage does
            # not fall back to classify_nearest per sentence.
            metadata={"mode": mode, "abstained": "control", "sentence_ontologies": {},
                      "sentence_confidences": {}, "backed_off": False, "top_confidence": 0.0})

    def encode(self, text: str, context: Optional[Dict] = None) -> EncodingResult:
        return self._encode("ingest")

    def encode_query(self, query: str, context: Optional[Dict] = None) -> EncodingResult:
        return self._encode("query")

    def classify_nearest(self, text: str, spacy_doc=None) -> Optional[str]:
        return None

    def emittable_classes(self) -> List[str]:
        return [UNCLASSIFIED]

    def config_summary(self) -> Dict:
        return {"impl": "abstain"}
