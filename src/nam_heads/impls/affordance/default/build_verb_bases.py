"""Build the base-verb vocabulary used to repair rule-lemmatizer output.

The rule lemmatizer strips suffixes without a dictionary check, so it emits
non-words for common verbs ("called" -> "cal", "joined" -> "joine",
"changed" -> "chang"). ``predicates.normalize_verb`` repairs a lemma by
generating candidate base forms from the surface text and keeping the first
one that is a real English verb. This script freezes that set of real verbs:
every single-word verb lemma in WordNet.

Only the membership set is used -- no WordNet sense, synset or hypernym
information -- so no two verbs are ever merged by it.

Usage:
    python -m nam_heads.impls.affordance.default.build_verb_bases [--output path]

Dependencies (build time only): nltk with the wordnet corpus.
"""

import argparse
import json
import os

OUT = os.path.join(os.path.dirname(__file__), "data", "verb_bases.json")


def build() -> list:
    from nltk.corpus import wordnet as wn
    return sorted({l.name().lower() for syn in wn.all_synsets("v")
                   for l in syn.lemmas() if l.name().isalpha()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=OUT)
    args = ap.parse_args()
    bases = build()
    with open(args.output, "w") as f:
        json.dump(bases, f, separators=(",", ":"))
    print(f"wrote {len(bases)} verb bases to {args.output}")


if __name__ == "__main__":
    main()
