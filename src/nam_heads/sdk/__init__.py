"""The contract between NAM and a head implementation."""

from .head import EncoderHead, EncodingResult, SemanticAddress
from .learner import HeadLearner, parked_result, parked_values

__all__ = ["EncoderHead", "EncodingResult", "SemanticAddress", "HeadLearner", "parked_result",
           "parked_values"]
