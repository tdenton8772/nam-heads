"""
Build verb synonym dictionary from WordNet.

Generates a canonical synonym mapping where semantically equivalent verbs
map to a single canonical form.

  Phase 1 (synset-based): a verb folds into the most frequent lemma of its
           dominant synset when both words share that dominant sense (rules
           in build_synonym_dict). A _NEVER_MERGE word is never a canonical,
           so nothing folds into repair or jump: mend and leap stay themselves.

  Phase 2 (curated overlay): Hand-curated cross-synset merges for common verbs
           that WordNet puts in separate synsets but are functionally identical
           for NAM's purposes (e.g. hop->jump).

Wu-Palmer similarity was evaluated but rejected: the verb taxonomy is too
shallow (max depth 2-3) making WuP unreliable for distinguishing true
synonyms from merely related verbs. Transitive merging through group
operations causes catastrophic drift.

Output: verb_synonyms.json -- a flat dict {variant: canonical}
        Only contains entries where variant != canonical.
        verb_synonyms_unfolded.json -- words an older table folded that this
        one keeps as themselves (identity seed docs, predicate_store).
        A changed table changes the affordance seed version: head pods
        re-seed the store's synonym docs at start.

Usage:
    python -m nam_heads.impls.affordance.default.build_synonyms [--output path]

Dependencies:
    nltk (with wordnet corpus)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from typing import Dict, Set

logger = logging.getLogger(__name__)

# Written next to verb_synonyms.json: words an older table folded and the
# current one does not (see unfolded_words).
UNFOLDED_FILE = "verb_synonyms_unfolded.json"

# Verbs that should NEVER be remapped (too polysemous / light / common)
_NEVER_MERGE: Set[str] = {
    "be", "is", "was", "were", "am", "are",
    "have", "has", "had",
    "do", "does", "did",
    "get", "set", "put", "let", "make",
    "go", "come", "take", "give",
    # Common verbs that WordNet maps to less-common first lemmas
    # (break.v.01 -> interrupt, build.v.01 -> construct)
    "break", "build", "run", "walk", "jump", "fly", "eat", "drink",
    "think", "talk", "hit", "cut", "pull", "push", "throw",
    "repair", "destroy", "understand", "choose", "shout",
    "reduce", "expand", "grow",
    # Common everyday verbs — too polysemous or too basic to remap
    "start", "stop", "open", "close", "move", "turn", "hold", "keep",
    "leave", "find", "show", "tell", "say", "write", "read", "play",
    "work", "live", "die", "kill", "send", "bring", "buy", "sell",
    "pay", "win", "lose", "pass", "fall", "rise", "sit", "stand",
    "hang", "wear", "carry", "catch", "drop", "lift", "touch", "reach",
    "teach", "learn", "speak", "sing", "dance", "fight", "hide", "burn",
    "cook", "wash", "clean", "fill", "cover", "block", "check", "mark",
    "watch", "wait", "help", "save", "serve", "lead", "follow", "meet",
    "join", "share", "test", "try", "use", "change", "fix", "add",
    "remove", "create", "produce", "develop", "design", "plant",
    "charge", "attack", "defend", "protect", "support", "measure",
    "count", "print", "draw", "paint", "drive", "ride", "climb", "swim",
    "pour", "mix", "stir", "shake", "press", "squeeze", "bend", "fold",
    "wrap", "tie", "lock", "load", "aim", "fire", "shoot", "land",
    "launch", "dig", "roll", "slide", "spin", "twist", "flip", "kick",
    "strike", "trigger", "trip",
    # Dominant WordNet (SemCor) sense differs from the encyclopedic sense
    # (checked against the wiki-abstracts verb inventory): promote (to a
    # rank), race (in sport), enrol (at a school), employ, affiliate,
    # apologise (WordNet: justify), animate (film), detect, abbreviate,
    # rely, link.
    "promote", "race", "enrol", "enroll", "employ", "affiliate",
    "apologise", "apologize", "animate", "detect", "abbreviate", "rely", "link",
}

# Folds that pass the frequency rules but change the meaning in
# encyclopedic text (hand audit of the folds firing on 5,000 wiki records,
# 2026-10-07): the verb stays itself. variant -> the reading it would lose.
_KEEP_DISTINCT: Dict[str, str] = {
    "ensure": "make certain (-> assure: tell confidently)",
    "intertwine": "be closely connected (-> twine: twist yarn)",
    "conquer": "take by force (-> inhibit: restrain)",
    "encompass": "include (-> embrace: hug, adopt)",
    "illuminate": "explain, light up (-> light: set fire to)",
    "prescribe": "order a medicine / a rule (-> dictate)",
    "advocate": "argue for (-> recommend)",
    "regain": "get back independence, control (-> retrieve: fetch)",
    "excel": "be very good at (-> surpass)",
    "finish": "end a race in a position (-> complete)",
    "evoke": "call to mind (-> arouse)",
    "react": "react chemically (-> respond)",
    "ordain": "make a priest (-> enact)",
    "spring": "originate (-> leap)",
    "recall": "recall a product / an envoy (-> remember)",
    "piece": "piece together (-> patch)",
    "steep": "steeped in tradition (-> engulf)",
    "number": "count, number among (-> total)",
    "sanction": "penalize (-> approve)",
    "manifest": "appear (-> evidence)",
    "pique": "pique curiosity (-> offend)",
    "wedge": "be squeezed between (-> stick)",
    "adjoin": "lie next to (-> butt)",
    "enthrone": "crown (-> vest)",
    "activate": "turn on (-> spark)",
    "deviate": "depart from a norm (-> divert)",
    "conglomerate": "a company (noun) (-> amass)",
    "amplify": "amplify a signal (-> magnify -> expand)",
}

# Cross-synset merges: verbs that WordNet puts in separate synsets but
# are functionally identical for affordance encoding. These are applied
# AFTER synset-based grouping, mapping variant -> canonical.
_CROSS_SYNSET_MERGES: Dict[str, str] = {
    # movement (hop/skip are separate synsets from jump; leap, bound and
    # spring are not folded into jump -- see the Phase 1 note above)
    "hop": "jump", "skip": "jump",
    "sprint": "run", "jog": "run",
    "stroll": "walk", "hike": "walk", "trek": "walk", "trudge": "walk",
    "stride": "walk", "amble": "walk", "saunter": "walk",
    "glide": "fly",
    # Not folded: dash (hopes were dashed), march (armies, protests), soar
    # (careers, prices).
    # physical
    "construct": "build", "erect": "build",
    "smash": "break", "shatter": "break", "crack": "break", "fracture": "break",
    "seize": "grab", "snatch": "grab", "clutch": "grab",
    "toss": "throw", "hurl": "throw", "fling": "throw", "lob": "throw",
    "haul": "pull", "tug": "pull", "yank": "pull",
    "shove": "push", "thrust": "push",
    "slice": "cut", "chop": "cut",
    "punch": "hit", "slap": "hit", "whack": "hit",
    # communicative
    "yell": "shout", "scream": "shout", "holler": "shout",
    "chat": "talk", "converse": "talk",
    "notify": "inform",
    "instruct": "teach",
    # Not folded: educate / train (-> teach) -- "educated at Trinity
    # College" is attending, "trained soldiers" is drilling.
    "inquire": "ask", "query": "ask",
    # cognitive
    "ponder": "think", "contemplate": "think", "deliberate": "think",
    "muse": "think", "ruminate": "think",
    "comprehend": "understand", "grasp": "understand", "fathom": "understand",
    "select": "choose", "opt": "choose", "pick": "choose",
    "scrutinize": "examine", "inspect": "examine",
    "memorize": "remember", "recollect": "remember",   # not recall (a product, an envoy)
    # transformative
    "revamp": "rebuild", "renovate": "rebuild",
    "enlarge": "expand", "magnify": "expand",   # not amplify: a signal
    "diminish": "reduce", "decrease": "reduce", "lessen": "reduce",
    "wreck": "destroy", "ruin": "destroy", "devastate": "destroy",
    # True synonyms in encyclopedic usage (predicate-mode values).
    "situate": "locate", "purchase": "buy", "wed": "marry",
    "comprise": "consist", "depict": "portray", "commence": "begin",
    "relocate": "move",
    # Removed as distinct senses: race->run (sport), consume->eat (fuel,
    # media), carve->cut (sculpt), trim->cut, assemble->build (a body
    # assembles), question->ask (doubt), alert->inform.
    # consumption
    "devour": "eat", "dine": "eat", "feast": "eat",
    "sip": "drink", "gulp": "drink", "quaff": "drink",
}


def wilson_lower(k: int, n: int, z: float = 1.0) -> float:
    """Wilson score lower bound of the proportion k/n."""
    if n <= 0:
        return 0.0
    p = k / n
    zz = z * z
    return (p + zz / (2 * n) - z * math.sqrt(p * (1 - p) / n + zz / (4 * n * n))) / (1 + zz / n)


def unfolded_words(previous_unfolded, previous_map: Dict[str, str],
                   synonym_map: Dict[str, str]) -> list:
    """Words an older table folded that this one does not. A store seeded
    from the older table still holds their synonym docs (seeding writes the
    seed's keys only), so predicate_store seeds an identity doc for each.
    The list carries forward: a word stays on it until it folds again."""
    return sorted((set(previous_unfolded) | set(previous_map)) - set(synonym_map))


def build_synonym_dict(min_lemma_len: int = 3) -> Dict[str, str]:
    """Build verb synonym dict from WordNet synsets + curated overlay.

    Returns dict mapping variant lemmas to canonical forms.
    Only includes entries where variant != canonical.
    """
    try:
        from nltk.corpus import wordnet as wn
    except ImportError:
        print("nltk not installed. Install with: pip install nltk", file=sys.stderr)
        sys.exit(1)

    try:
        wn.synsets("test")
    except LookupError:
        import nltk
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
        from nltk.corpus import wordnet as wn

    # ------------------------------------------------------------------
    # Phase 1: sense-preserving synset folding
    # ------------------------------------------------------------------
    # v folds into c only when they share their DOMINANT sense:
    #   - S = v's first (most frequent) synset, and S is also c's first
    #     synset -- each word's main meaning is S;
    #   - S carries most of v's usage, with evidence: v's SemCor counts give
    #     S >= 60% AND the Wilson lower bound (z = 1) of that share is >= 0.5,
    #     so 2 of 3 or 3 of 5 sightings are not enough (border, revere,
    #     storm, vacate folded on those). A verb with no counts folds only
    #     when it has ONE verb sense -- nothing to lose;
    #   - c is the most frequent single-word lemma of S (SemCor count, then
    #     name) and is counted in S at least once -- rare words fold to
    #     common ones, never the reverse; with no counts on either side the
    #     choice was the shortest name (popularize -> vulgarize, game ->
    #     gage);
    #   - v is not in _KEEP_DISTINCT.
    # Taking the first lemma of v's first synset (the previous rule) folded
    # v into whatever word WordNet listed first, whose main meaning was
    # often different: consist/comprise/dwell -> brood, earn/gain/derive ->
    # deduce, celebrate/observe -> detect, base/found -> establish.
    synonym_map: Dict[str, str] = {}

    def ok_name(name: str) -> bool:
        name = name.replace("_", " ")
        return " " not in name and "-" not in name and len(name) >= min_lemma_len \
            and name not in _NEVER_MERGE

    all_lemmas: Set[str] = set()
    for ss in wn.all_synsets(pos=wn.VERB):
        for lem in ss.lemmas():
            name = lem.name().lower().replace("_", " ")
            if ok_name(name):
                all_lemmas.add(name)

    def lemma_count(ss, name: str) -> int:
        for lem in ss.lemmas():
            if lem.name().lower() == name:
                return lem.count()
        return 0

    for lemma in sorted(all_lemmas):
        if lemma in _KEEP_DISTINCT:
            continue
        synsets = wn.synsets(lemma, pos=wn.VERB)
        if not synsets:
            continue
        primary_ss = synsets[0]
        counts = [lemma_count(ss, lemma) for ss in synsets]
        total = sum(counts)
        if total > 0:
            if counts[0] / total < 0.6 or wilson_lower(counts[0], total) < 0.5:
                continue
        elif len(synsets) > 1:
            continue
        best = None
        names_in_ss = {l.name().lower() for l in primary_ss.lemmas()}
        for lem in primary_ss.lemmas():
            name = lem.name().lower()
            if not ok_name(name) or name == lemma:
                continue
            # Spelling variants are not synonyms to fold by frequency: the
            # -ize spelling is canonical (predicates.py folds -ise -> -ize).
            if name.endswith(("ise", "yse")) and name[:-2] + "ze" in names_in_ss:
                continue
            c_ss = wn.synsets(name, pos=wn.VERB)
            if not c_ss or c_ss[0] != primary_ss:
                continue
            key = (lem.count(), -len(name), name)
            if best is None or key > best[0]:
                best = (key, name)
        if best is None:
            continue
        canonical = best[1]
        c_count = lemma_count(primary_ss, canonical)
        if c_count < lemma_count(primary_ss, lemma) or c_count == 0:
            continue
        synonym_map[lemma] = canonical

    logger.info("Phase 1 (synset): %d direct mappings from %d lemmas",
                len(synonym_map), len(all_lemmas))

    # ------------------------------------------------------------------
    # Phase 2: curated cross-synset overrides
    # ------------------------------------------------------------------
    # These override phase 1 mappings for verbs that WordNet puts in
    # different synsets but are functionally identical for NAM.
    merges_applied = 0
    for variant, target in _CROSS_SYNSET_MERGES.items():
        if variant in _KEEP_DISTINCT:
            continue
        if variant in all_lemmas or variant in synonym_map:
            old = synonym_map.get(variant)
            if old != target:
                synonym_map[variant] = target
                merges_applied += 1

    # Resolve all chains: if A->B and B->C, make A->C directly.
    # Iterate until no chains remain (typically 2-3 passes).
    for _ in range(10):  # safety bound
        changed = False
        for k in list(synonym_map):
            v = synonym_map[k]
            if v in synonym_map:
                synonym_map[k] = synonym_map[v]
                changed = True
        if not changed:
            break

    # Clean self-mappings
    synonym_map = {k: v for k, v in synonym_map.items() if k != v}

    logger.info("Phase 2 (curated): %d overrides applied", merges_applied)
    logger.info("Final: %d entries", len(synonym_map))
    return synonym_map


def build_category_dict(synonym_map: Dict[str, str]) -> Dict[str, str]:
    """Build verb -> category mapping using WordNet hypernym chains.

    For each canonical verb form (from VERB_CATEGORIES seed + synonym dict
    targets), walks the WordNet hypernym tree to find a synset containing
    a verb whose category is already known from the seed data.

    Returns dict mapping verb lemma -> category string.
    Only includes entries not already in VERB_CATEGORIES seed data.
    """
    from .definitions import VERB_CATEGORIES

    try:
        from nltk.corpus import wordnet as wn
    except ImportError:
        logger.warning("nltk not available, skipping category dict generation")
        return {}

    # Build verb -> category from existing seed
    seed_cats: Dict[str, str] = {}
    for cat, verbs in VERB_CATEGORIES.items():
        for v in verbs:
            seed_cats[v] = cat

    # Build synset -> category from seed verbs
    synset_to_cat: Dict[str, str] = {}
    for verb, cat in seed_cats.items():
        for ss in wn.synsets(verb, pos=wn.VERB):
            for lem in ss.lemmas():
                name = lem.name().lower().replace("_", " ")
                if name == verb:
                    if ss.name() not in synset_to_cat:
                        synset_to_cat[ss.name()] = cat
                    break

    def _classify(lemma: str) -> str:
        """Walk hypernym chain to find category."""
        ss_list = wn.synsets(lemma, pos=wn.VERB)
        if not ss_list:
            return None
        visited: Set[str] = set()
        queue = [ss_list[0]]
        while queue:
            current = queue.pop(0)
            if current.name() in visited:
                continue
            visited.add(current.name())
            if current.name() in synset_to_cat:
                return synset_to_cat[current.name()]
            for hyp in current.hypernyms():
                queue.append(hyp)
        return None

    # Every single-word WordNet verb gets a category where its hypernym
    # chain reaches a seed verb. Coverage used to be tied to the fold's
    # canonicals; the sense-preserving fold has far fewer of them.
    canonicals = {l.name().lower() for ss in wn.all_synsets(pos=wn.VERB) for l in ss.lemmas()
                  if l.name().isalpha()} | set(synonym_map.values()) | set(seed_cats.keys())
    category_map: Dict[str, str] = {}

    for verb in sorted(canonicals):
        if verb in seed_cats:
            continue  # already in seed, no need to duplicate
        cat = _classify(verb)
        if cat:
            category_map[verb] = cat

    logger.info("Category dict: %d entries (%d from seed, %d from hypernyms, %d unclassified)",
                len(seed_cats) + len(category_map), len(seed_cats), len(category_map),
                len(canonicals) - len(seed_cats) - len(category_map))
    return category_map


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Build verb synonym dict from WordNet")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: data/verb_synonyms.json)")
    parser.add_argument("--categories", action="store_true",
                        help="also rewrite data/verb_categories.json from the hypernym walk")
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    if args.output is None:
        args.output = os.path.join(data_dir, "verb_synonyms.json")

    synonym_map = build_synonym_dict()

    previous: Dict[str, str] = {}
    if os.path.exists(args.output):
        with open(args.output) as f:
            previous = json.load(f)
    unfolded_path = os.path.join(os.path.dirname(args.output), UNFOLDED_FILE)
    previous_unfolded = []
    if os.path.exists(unfolded_path):
        with open(unfolded_path) as f:
            previous_unfolded = json.load(f)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(synonym_map, f, indent=2, sort_keys=True)
    unfolded = unfolded_words(previous_unfolded, previous, synonym_map)
    with open(unfolded_path, "w") as f:
        json.dump(unfolded, f, indent=2)
    logger.info("Unfolded words (identity seed docs): %d -> %s", len(unfolded), unfolded_path)

    size_kb = os.path.getsize(args.output) / 1024
    logger.info("Written %d entries to %s (%.0f KB)", len(synonym_map), args.output, size_kb)

    # The category dict (verb_categories.json) is NOT rewritten here unless
    # asked: the shipped file is a superset of what build_category_dict
    # derives from the current fold, and the category store is seeded from it.
    category_map = {}
    cat_kb = 0.0
    if args.categories:
        category_map = build_category_dict(synonym_map)
        cat_path = os.path.join(data_dir, "verb_categories.json")
        with open(cat_path, "w") as f:
            json.dump(category_map, f, indent=2, sort_keys=True)
        cat_kb = os.path.getsize(cat_path) / 1024
        logger.info("Written %d category entries to %s (%.0f KB)",
                    len(category_map), cat_path, cat_kb)

    # Verify key mappings
    print("\nSample mappings:")
    samples = ["hop", "leap", "bound", "mend", "fix", "sprint", "stroll",
               "construct", "smash", "yell", "ponder", "comprehend",
               "devour", "demolish", "toss", "shove"]
    for s in samples:
        canon = synonym_map.get(s, s)
        marker = " <-" if s in synonym_map else ""
        print(f"  {s} -> {canon}{marker}")

    # Stats
    canonicals = set(synonym_map.values())
    all_verbs = set(synonym_map.keys()) | canonicals
    print(f"\nTotal verbs covered: {len(all_verbs)}")
    print(f"Unique canonical forms: {len(canonicals)}")
    print(f"Synonym dict entries: {len(synonym_map)}")
    if args.categories:
        print(f"Category dict entries: {len(category_map)}")
    print(f"Unfolded words: {len(unfolded)}")
    print(f"Synonym file size: {size_kb:.0f} KB")
    if args.categories:
        print(f"Category file size: {cat_kb:.0f} KB")


if __name__ == "__main__":
    main()
