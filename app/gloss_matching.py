"""Match free-text input against the sign dictionary vocabulary.

This is NOT a translator: it finds which dictionary sign names already appear
in the input text, in the order they appear, via greedy longest-phrase
matching plus a small set of alias rules for compound/qualified sign names
(so casual phrasing like bare "you" or "big" still resolves). Producing
grammatically correct ISL gloss order from arbitrary English is a much
harder, unscoped problem - this deliberately doesn't attempt it.
"""
import re

STOPWORDS = {
    "a", "an", "the", "is", "am", "are", "was", "were", "be", "been", "being",
    "to", "of", "and", "or", "but", "so", "if", "as", "at", "by", "for",
    "in", "on", "with", "my", "your", "his", "her", "its", "our", "their",
    "this", "that", "these", "those", "do", "does", "did", "has", "have", "had",
}

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str):
    return tuple(_WORD_RE.findall(text.lower()))


def _aliases(sign_name: str):
    """Alternate typed forms that should also resolve to this sign name."""
    names = {sign_name}
    if "(" in sign_name:
        names.add(sign_name.split("(")[0].strip())
    if " or " in sign_name:
        for part in sign_name.split(" or "):
            names.add(part.strip())
    if sign_name in ("big large", "small little"):
        names.update(sign_name.split(" "))
    if sign_name.startswith("ex. "):
        names.add(sign_name[4:].strip())
    if "-" in sign_name:
        names.add(sign_name.replace("-", " "))
        names.add(sign_name.replace("-", ""))
    return names


def build_alias_index(sign_names):
    """sign_names: iterable of lowercase dictionary sign names.

    Returns {tuple(tokens): canonical_sign_name}. On alias collisions (e.g.
    bare "you" from both "you" and "you (plural)"), the plainer/shorter
    canonical name wins, since it's processed first, so the common case
    matches (qualified variants like "(plural)" still match when typed in
    full).
    """
    index = {}
    ordered = sorted(sign_names, key=lambda n: ("(" in n, len(n)))
    for name in ordered:
        for alias in _aliases(name):
            tokens = _tokenize(alias)
            if tokens and tokens not in index:
                index[tokens] = name
    return index


def parse_sentence(text: str, index: dict):
    """Greedy longest-match tokenizer against `index` (from build_alias_index).

    Returns (gloss, unmatched). `gloss` is the ordered list of matched
    dictionary sign names (duplicates allowed). `unmatched` is content words
    (not in STOPWORDS) that matched nothing - for a soft warning, not an
    error.
    """
    words = list(_tokenize(text))
    max_phrase = max((len(k) for k in index), default=1)

    gloss = []
    unmatched = []
    i = 0
    n = len(words)
    while i < n:
        matched = False
        for span in range(min(max_phrase, n - i), 0, -1):
            phrase = tuple(words[i:i + span])
            if phrase in index:
                gloss.append(index[phrase])
                i += span
                matched = True
                break
        if not matched:
            word = words[i]
            if word not in STOPWORDS:
                unmatched.append(word)
            i += 1

    return gloss, unmatched
