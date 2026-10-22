"""Seed merging shared by every category-store backend."""
from typing import Optional


def merge_seed(old: Optional[dict], category: str, confidence: float,
               extras: Optional[dict]) -> dict:
    """The row a seed entry leaves behind.

    A missing row is created. An existing row keeps its category and
    confidence (an earlier seed or a learned row wins) and gains only the
    extras it lacks -- so seeding twice, or seeding over a grown store,
    changes nothing it already holds.
    """
    if old is None:
        new = {"category": category, "confidence": confidence}
        new.update(extras or {})
        return new
    new = dict(old)
    for k, v in (extras or {}).items():
        new.setdefault(k, v)
    return new
