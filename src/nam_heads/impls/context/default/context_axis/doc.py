"""Duck-typed access to a parsed document (rule-backend ParsedDocument or LightDoc)."""

import bisect
from typing import List, Tuple


class Span:
    __slots__ = ("start", "end", "label", "text")

    def __init__(self, start: int, end: int, label: str, text: str):
        self.start, self.end, self.label, self.text = start, end, label, text


def tokens(doc) -> list:
    return list(doc) if doc is not None else []


def sentences(doc, n_tokens: int) -> List[Tuple[int, int]]:
    """(start, end) token ranges of each sentence; one range if the doc has none."""
    out = []
    for s in (getattr(doc, "sents", None) or []):
        start, end = getattr(s, "start", None), getattr(s, "end", None)
        if start is None or end is None:
            idx = [t.i for t in s]
            if not idx:
                continue
            start, end = idx[0], idx[-1] + 1
        out.append((start, end))
    return out or ([(0, n_tokens)] if n_tokens else [])


def entities(doc, toks: list) -> List[Span]:
    out = []
    for e in (getattr(doc, "ents", None) or []):
        start, end = getattr(e, "start", None), getattr(e, "end", None)
        if start is None or end is None:
            idx = [t.i for t in e]
            if not idx:
                continue
            start, end = idx[0], idx[-1] + 1
        text = getattr(e, "text", None) or " ".join(t.text for t in toks[start:end])
        out.append(Span(start, end, e.label_, text))
    return out


def sentence_lookup(sents: List[Tuple[int, int]]):
    """A function i -> sentence_of(sents, i), by binary search when the
    ranges are ordered and disjoint (as a parse yields them), else the
    linear scan."""
    starts = [s for s, _ in sents]
    ordered = all(sents[k][1] <= sents[k + 1][0] for k in range(len(sents) - 1)) and \
        all(s <= e for s, e in sents)
    if not ordered:
        return lambda i: sentence_of(sents, i)

    def find(i: int) -> int:
        k = bisect.bisect_right(starts, i) - 1
        if k >= 0 and sents[k][0] <= i < sents[k][1]:
            return k
        return 0
    return find


def sentence_of(sents: List[Tuple[int, int]], i: int) -> int:
    for si, (s, e) in enumerate(sents):
        if s <= i < e:
            return si
    return 0
