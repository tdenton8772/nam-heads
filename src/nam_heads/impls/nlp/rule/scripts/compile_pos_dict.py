#!/usr/bin/env python3
"""
One-time script: Brown corpus -> pos_dictionary.json

Downloads the Brown corpus via NLTK, counts POS tag frequencies per word,
maps Penn Treebank tags to Universal POS, and emits the most frequent
POS for each lowercase word form.

Output: data/pos_dictionary.json  (word -> Universal POS)
Not needed at runtime.
"""

import json
import os
import sys
from collections import Counter, defaultdict

import nltk

# Penn Treebank -> Universal POS mapping
_PTB_TO_UPOS = {
    # Nouns
    "NN": "NOUN", "NNS": "NOUN", "NN-TL": "NOUN", "NNS-TL": "NOUN",
    "NN$": "NOUN", "NNS$": "NOUN", "NN-HL": "NOUN", "NNS-HL": "NOUN",
    "NN-NC": "NOUN", "NNS-NC": "NOUN",
    "NN+BEZ": "NOUN", "NN+HVZ": "NOUN", "NN+HVD": "NOUN",
    "NN+IN": "NOUN", "NN+MD": "NOUN",
    "NN+NN": "NOUN", "NNS+MD": "NOUN",
    "NNS-TL-HL": "NOUN",
    "NN-TL-HL": "NOUN",
    # Proper nouns
    "NP": "PROPN", "NPS": "PROPN", "NP-TL": "PROPN", "NPS-TL": "PROPN",
    "NP$": "PROPN", "NPS$": "PROPN", "NP-HL": "PROPN", "NPS-HL": "PROPN",
    "NP+BEZ": "PROPN", "NP+HVZ": "PROPN", "NP+MD": "PROPN",
    "NP-TL-HL": "PROPN", "NP-NC": "PROPN",
    # Verbs
    "VB": "VERB", "VBD": "VERB", "VBG": "VERB", "VBN": "VERB", "VBZ": "VERB",
    "VB-TL": "VERB", "VBD-TL": "VERB", "VBG-TL": "VERB", "VBN-TL": "VERB",
    "VBZ-TL": "VERB", "VB-HL": "VERB", "VBD-HL": "VERB", "VBG-HL": "VERB",
    "VBN-HL": "VERB", "VB-NC": "VERB", "VBN-NC": "VERB",
    "VBN+TO": "VERB",
    # Adjectives
    "JJ": "ADJ", "JJR": "ADJ", "JJS": "ADJ", "JJT": "ADJ",
    "JJ-TL": "ADJ", "JJR-TL": "ADJ", "JJS-TL": "ADJ",
    "JJ-HL": "ADJ", "JJR-HL": "ADJ",
    "JJ-NC": "ADJ",
    "JJ$": "ADJ",
    # Adverbs
    "RB": "ADV", "RBR": "ADV", "RBT": "ADV", "RN": "ADV",
    "RB-TL": "ADV", "RBR-TL": "ADV",
    "RB-HL": "ADV", "RB-NC": "ADV",
    "RB$": "ADV",
    "QL": "ADV", "QLP": "ADV", "QL-TL": "ADV",
    "RP": "PART", "RP-TL": "PART", "RP-HL": "PART", "RP-NC": "PART",
    # Determiners / articles
    "AT": "DET", "AT-TL": "DET", "AT-HL": "DET", "AT-NC": "DET",
    "DTI": "DET", "DTS": "DET", "DTX": "DET",
    "DT": "DET", "DT-TL": "DET", "DT-HL": "DET",
    "DT$": "DET",
    "ABN": "DET", "ABX": "DET",
    "AP": "DET", "AP-TL": "DET", "AP-HL": "DET", "AP$": "DET",
    # Pronouns
    "PP$": "PRON", "PP$$": "PRON", "PPL": "PRON", "PPLS": "PRON",
    "PPO": "PRON", "PPS": "PRON", "PPSS": "PRON",
    "PPS+BEZ": "PRON", "PPS+HVD": "PRON", "PPS+HVZ": "PRON",
    "PPS+MD": "PRON", "PPSS+BER": "PRON", "PPSS+BEM": "PRON",
    "PPSS+HV": "PRON", "PPSS+MD": "PRON", "PPSS+VB": "PRON",
    "PPO-TL": "PRON", "PPS-TL": "PRON",
    "PP$-TL": "PRON",
    "PPL-TL": "PRON",
    "EX": "PRON", "EX+BEZ": "PRON", "EX+HVD": "PRON",
    "EX+HVZ": "PRON", "EX+MD": "PRON",
    "WPO": "PRON", "WPS": "PRON", "WPS+BEZ": "PRON",
    "WP$": "PRON", "WDT": "PRON",
    "WDT+BER": "PRON", "WDT+BEZ": "PRON",
    "WDT+DO+PPS": "PRON", "WDT+DOD": "PRON",
    "WDT+HVZ": "PRON",
    # Prepositions / subordinating conjunctions
    "IN": "ADP", "IN-TL": "ADP", "IN-HL": "ADP", "IN-NC": "ADP",
    "IN+IN": "ADP", "IN+PPO": "ADP",
    # Coordinating conjunctions
    "CC": "CCONJ", "CC-TL": "CCONJ", "CC-HL": "CCONJ", "CC-NC": "CCONJ",
    # Subordinating conjunctions (Brown uses CS for these)
    "CS": "SCONJ", "CS-TL": "SCONJ", "CS-HL": "SCONJ", "CS-NC": "SCONJ",
    # Modals
    "MD": "AUX", "MD-TL": "AUX", "MD-HL": "AUX", "MD-NC": "AUX",
    "MD+HV": "AUX", "MD+PPSS": "AUX", "MD+TO": "AUX",
    # Be / Have auxiliaries
    "BEZ": "AUX", "BER": "AUX", "BED": "AUX", "BEDZ": "AUX",
    "BEN": "AUX", "BEG": "AUX", "BEM": "AUX", "BE": "AUX",
    "BEZ-TL": "AUX", "BER-TL": "AUX", "BED-TL": "AUX",
    "BEZ-HL": "AUX", "BEZ-NC": "AUX",
    "HV": "AUX", "HVD": "AUX", "HVG": "AUX", "HVN": "AUX", "HVZ": "AUX",
    "HV-TL": "AUX", "HVD-TL": "AUX", "HVZ-TL": "AUX",
    "HV-HL": "AUX",
    "HV*": "AUX", "HVD*": "AUX", "HVZ*": "AUX",
    "DO": "AUX", "DOD": "AUX", "DOZ": "AUX",
    "DO-TL": "AUX", "DO-HL": "AUX",
    "DO*": "AUX", "DOD*": "AUX", "DOZ*": "AUX",
    # Particles / interjections / misc
    "TO": "PART", "TO-TL": "PART", "TO-HL": "PART", "TO-NC": "PART",
    "UH": "INTJ", "UH-TL": "INTJ", "UH-HL": "INTJ", "UH-NC": "INTJ",
    "WRB": "ADV", "WRB-TL": "ADV", "WRB-HL": "ADV", "WRB-NC": "ADV",
    "WQL": "ADV",
    # Numbers
    "CD": "NUM", "CD-TL": "NUM", "CD-HL": "NUM", "CD-NC": "NUM",
    "CD$": "NUM",
    "OD": "NUM", "OD-TL": "NUM", "OD-HL": "NUM",
    # Punctuation
    ".": "PUNCT", ",": "PUNCT", ":": "PUNCT", "``": "PUNCT", "''": "PUNCT",
    "(": "PUNCT", ")": "PUNCT", "--": "PUNCT", "---": "PUNCT",
    "*": "PUNCT", "(-": "PUNCT", ")-": "PUNCT",
    "(-HL": "PUNCT", ")-HL": "PUNCT",
    # Foreign words
    "FW": "X", "FW-TL": "X", "FW-HL": "X", "FW-NC": "X",
    "FW-AT": "X", "FW-AT-TL": "X",
    "FW-BE": "X", "FW-BER": "X", "FW-BEZ": "X",
    "FW-CC": "X", "FW-CD": "X", "FW-CS": "X",
    "FW-DT": "X", "FW-DTS": "X", "FW-DTI": "X",
    "FW-HV": "X", "FW-IN": "X",
    "FW-JJ": "X", "FW-JJR": "X", "FW-JJT": "X",
    "FW-NN": "X", "FW-NNS": "X", "FW-NP": "X",
    "FW-OD": "X",
    "FW-PP$": "X", "FW-PPL": "X", "FW-PPO": "X",
    "FW-PPS": "X", "FW-PPSS": "X",
    "FW-RB": "X", "FW-RP": "X",
    "FW-TO": "X", "FW-UH": "X",
    "FW-VB": "X", "FW-VBD": "X", "FW-VBG": "X", "FW-VBN": "X",
    "FW-VBZ": "X",
    "FW-WDT": "X", "FW-WPO": "X", "FW-WPS": "X",
    "FW-IN-TL": "X", "FW-JJ-TL": "X", "FW-NN-TL": "X",
    "FW-NNS-TL": "X", "FW-NP-TL": "X", "FW-VB-TL": "X",
    "FW-VBD-TL": "X", "FW-VBG-TL": "X", "FW-VBN-TL": "X",
    "FW-IN+NN": "X", "FW-IN+NP": "X", "FW-IN+AT-TL": "X",
    # Negation
    "*": "PART",
    "NOT": "PART",
    # Others
    "NIL": "X", "NR": "NOUN", "NR-TL": "NOUN", "NR$": "NOUN",
    "NR-HL": "NOUN", "NRS": "NOUN",
}


def main():
    # Download required data
    nltk.download("brown", quiet=True)

    from nltk.corpus import brown

    # Count POS frequencies per word
    word_pos_counts: dict[str, Counter] = defaultdict(Counter)

    for word, tag in brown.tagged_words():
        lower = word.lower()
        upos = _PTB_TO_UPOS.get(tag)
        if upos and upos not in ("PUNCT", "X"):
            word_pos_counts[lower][upos] += 1

    # Keep most frequent POS per word, skip very rare words
    pos_dict: dict[str, str] = {}
    for word, counts in sorted(word_pos_counts.items()):
        # Only include words that appear at least twice
        total = sum(counts.values())
        if total < 2:
            continue
        # Skip pure-number words
        if word.isdigit():
            continue
        # Skip single characters except known ones
        if len(word) == 1 and word not in ("a", "i"):
            continue
        most_common_pos = counts.most_common(1)[0][0]
        pos_dict[word] = most_common_pos

    # Write output
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "pos_dictionary.json")

    with open(out_path, "w") as f:
        json.dump(pos_dict, f, indent=0, sort_keys=True)

    print(f"Wrote {len(pos_dict)} entries to {out_path}")


if __name__ == "__main__":
    main()
