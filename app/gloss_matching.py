"""Turn an English sentence into an ISL gloss, then say which of its signs the
dictionary is missing.

Two users: scripts/build_static_data.py, which proves every curated sentence's
English glosses to exactly its stored gloss with no missing sign; and
app/static/gloss_matching.js, the hand-kept JS mirror that the "write your own
sentence" flow runs in the browser (there is no server on GitHub Pages). Keep
the two in sync by hand.

This is a rule-based glosser, not a translator. It applies the ISL grammar
the app teaches (ISLRTC / Zeshan) - the same rules listed in the grammar
guide on the Sentences page:

  * drop function words that ISL has no sign for (articles, copula, most
    prepositions, "and", auxiliaries do/will/can, possessive determiners);
  * NEGATION goes to the end as one sign, NOT ("don't", "no", "never"...);
  * QUESTION WORDS go to the end (what / where / who / when / why / how);
  * word order is TIME, then SUBJECT / OBJECT / PLACE nouns in their given
    order, then the ADJECTIVE / COLOUR predicate, then the VERB (SOV), then
    NOT, then the question word;
  * pronouns keep their sign whatever their English case (me -> I, him ->
    HE); plurals drop the -s; verbs are reduced to their base form.

Every remaining content word becomes one gloss token. A token whose sign is
not in the dictionary is still kept in the gloss - the learner sees the real
ISL sentence - but is reported as missing, and the app refuses to practise a
sentence with any missing sign, because the coach has no reference for it.
"""
import re

# Words ISL simply does not sign. Auxiliary/modal verbs and the copula are
# here; "have/has/had" are NOT - HAVE is a real verb sign, so they become a
# HAVE token and get flagged like any other verb this dictionary lacks.
FUNCTION_WORDS = {
    "a", "an", "the",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "will", "would", "shall", "should", "can", "could",
    "may", "might", "must",
    "to", "of", "and", "or", "but", "so", "if", "as", "at", "by", "for",
    "in", "on", "with", "from", "into",
    "my", "your", "his", "her", "its", "our", "their",
    "this", "that", "these", "those", "there", "some", "any",
    "very", "really", "please",
}

# Every negation collapses to the single sign NOT, signed last.
NEGATION_WORDS = {"not", "no", "never", "nothing", "none", "nobody"}

# Question words are signed last.
WH_WORDS = {"what", "where", "who", "whom", "when", "why", "how", "which"}

# Object pronouns share the subject pronoun's sign (pointing).
PRONOUN_MAP = {"i": "i", "me": "i", "him": "he", "us": "we", "them": "they"}

# Verbs, so an inflected form can be reduced to its base and ordered last
# (after the object). The dictionary has none of these; they are flagged.
VERB_LEMMAS = {
    "go", "come", "eat", "drink", "have", "want", "like", "love", "need",
    "see", "look", "know", "read", "write", "play", "work", "sleep", "give",
    "take", "buy", "sell", "help", "learn", "teach", "make", "live", "run",
    "walk", "sit", "stand", "open", "close", "watch", "hear", "listen",
    "speak", "say", "talk", "think", "feel", "get", "put", "bring", "meet",
    "call", "wait", "cook", "drive", "ride", "study", "understand",
    "remember", "forget", "ask", "tell", "find", "lose", "win", "start",
    "stop", "try", "use", "visit", "wash", "wear", "carry", "hold", "cut",
    "fall", "fly", "sing", "dance", "laugh", "cry", "smile", "pay", "send",
    "show", "stay", "leave", "return", "sign",
}
IRREGULAR_VERBS = {
    "went": "go", "gone": "go", "goes": "go", "came": "come", "ate": "eat",
    "eaten": "eat", "drank": "drink", "drunk": "drink", "has": "have",
    "had": "have", "saw": "see", "seen": "see", "knew": "know", "known": "know",
    "wrote": "write", "written": "write", "gave": "give", "given": "give",
    "took": "take", "taken": "take", "bought": "buy", "sold": "sell", "made": "make",
    "ran": "run", "sat": "sit", "stood": "stand", "heard": "hear", "spoke": "speak",
    "said": "say", "thought": "think", "felt": "feel", "got": "get", "brought": "bring",
    "met": "meet", "drove": "drive", "rode": "ride", "understood": "understand",
    "forgot": "forget", "told": "tell", "found": "find", "lost": "lose", "won": "win",
    "wore": "wear", "held": "hold", "fell": "fall", "flew": "fly", "sang": "sing",
    "paid": "pay", "sent": "send", "left": "leave", "slept": "sleep", "cried": "cry",
}

# Typed without the apostrophe ("dont", "cant") - common on phones - these
# would otherwise fall through as unknown nouns and be glossed as DONT.
BARE_CONTRACTIONS = {
    "dont": "do not", "doesnt": "does not", "didnt": "did not",
    "isnt": "is not", "arent": "are not", "wasnt": "was not", "werent": "were not",
    "cant": "can not", "wont": "will not", "couldnt": "could not",
    "wouldnt": "would not", "shouldnt": "should not", "mustnt": "must not",
    "havent": "have not", "hasnt": "has not", "hadnt": "had not",
    "im": "i am", "youre": "you are", "theyre": "they are",
    "ive": "i have", "youve": "you have", "weve": "we have", "theyve": "they have",
}

CONTRACTIONS = {
    "can't": "can not", "cannot": "can not", "won't": "will not",
    "i'm": "i am", "you're": "you are", "we're": "we are", "they're": "they are",
    "he's": "he is", "she's": "she is", "it's": "it is",
    "i've": "i have", "you've": "you have", "we've": "we have", "they've": "they have",
    "i'll": "i will", "you'll": "you will", "we'll": "we will", "they'll": "they will",
    "he'll": "he will", "she'll": "she will", "it'll": "it will",
    "i'd": "i would", "you'd": "you would", "we'd": "we would", "they'd": "they would",
    "let's": "let us",
}

# Sentence-initial temporal adverbs. Only these move to the front - the rest of
# the "Days And Time" category (week, month, hour, time...) are ordinary nouns.
TIME_WORDS = {
    "today", "tomorrow", "yesterday", "morning", "afternoon", "evening", "night",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
}

# Dictionary categories whose signs act as the sentence's predicate (comment),
# which ISL places after the noun(s) it describes. "colour" itself is a noun.
PREDICATE_CATEGORIES = {"Adjectives", "Colours"}
PREDICATE_EXCEPTIONS = {"colour"}

# Gloss order classes. Stable sort within a class keeps the English order,
# which already puts subject before object.
CLS_TIME, CLS_NOUN, CLS_PREDICATE, CLS_VERB, CLS_NOT, CLS_WH = 0, 1, 2, 3, 4, 5

_WORD_RE = re.compile(r"[a-z0-9]+")


def _normalize(text: str) -> str:
    t = text.lower()
    t = re.sub(r"[\u2018\u2019\u02bc`]", "'", t)  # curly / typographic apostrophes
    t = re.sub(r"n't\b", " not", t)          # don't / isn't / haven't ...
    t = re.sub(r"\b[a-z]+\b", lambda m: BARE_CONTRACTIONS.get(m.group(0), m.group(0)), t)
    for k, v in CONTRACTIONS.items():
        t = t.replace(k, v)
    # Possessive 's carries no sign (the possessor simply precedes the
    # possessed), so strip it or "friend's" leaves a stray "s" token.
    t = re.sub(r"'s\b", "", t)
    return t


def _tokenize(text: str):
    return tuple(_WORD_RE.findall(_normalize(text)))


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
            tokens = tuple(_WORD_RE.findall(alias.lower()))
            if tokens and tokens not in index:
                index[tokens] = name
    return index


def _singular(word: str):
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("es") and len(word) > 3:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 2:
        return word[:-1]
    return None


def _verb_lemma(word: str):
    if word in VERB_LEMMAS:
        return word
    if word in IRREGULAR_VERBS:
        return IRREGULAR_VERBS[word]
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix):
            base = word[: -len(suffix)]
            if base in VERB_LEMMAS:
                return base
            if suffix == "ing" and base + "e" in VERB_LEMMAS:  # coming, making
                return base + "e"
            if suffix == "ed" and base + "e" in VERB_LEMMAS:   # liked, loved
                return base + "e"
    return None


def _sign_class(sign_name, category):
    if sign_name in TIME_WORDS:
        return CLS_TIME
    if category in PREDICATE_CATEGORIES and sign_name not in PREDICATE_EXCEPTIONS:
        return CLS_PREDICATE
    return CLS_NOUN


def to_gloss(text: str, index: dict, category_by_name: dict):
    """English -> ISL gloss.

    Returns (tokens, missing). tokens is the ordered gloss, each a dict
    {"label": gloss word, "sign": dictionary sign name or None}. missing is
    the list of labels whose sign the dictionary lacks, in gloss order.
    """
    words = list(_tokenize(text))
    max_phrase = max((len(k) for k in index), default=1)

    def sign_token(name, source_word=None):
        return {"label": name, "sign": name, "cls": _sign_class(name, category_by_name.get(name))}

    tokens = []
    i, n = 0, len(words)
    while i < n:
        # Longest dictionary phrase first - "good morning", "train station",
        # "how are you" are single signs even though they contain function
        # words and question words.
        matched = False
        for span in range(min(max_phrase, n - i), 0, -1):
            phrase = tuple(words[i:i + span])
            if phrase in index:
                tokens.append(sign_token(index[phrase]))
                i += span
                matched = True
                break
        if matched:
            continue

        w = words[i]
        i += 1
        if w in NEGATION_WORDS:
            tokens.append({"label": "not", "sign": index.get(("not",)), "cls": CLS_NOT})
        elif w in WH_WORDS:
            tokens.append({"label": w, "sign": index.get((w,)), "cls": CLS_WH})
        elif w in PRONOUN_MAP:
            name = PRONOUN_MAP[w]
            tokens.append({"label": name, "sign": index.get((name,)), "cls": CLS_NOUN})
        elif w in FUNCTION_WORDS:
            continue
        else:
            lemma = _verb_lemma(w)
            if lemma:
                tokens.append({"label": lemma, "sign": index.get((lemma,)), "cls": CLS_VERB})
                continue
            base = _singular(w)
            if base and (base,) in index:
                tokens.append(sign_token(index[(base,)]))
                continue
            tokens.append({"label": w, "sign": None, "cls": CLS_NOUN})

    tokens.sort(key=lambda t: t["cls"])  # stable
    gloss = [{"label": t["label"], "sign": t["sign"]} for t in tokens]
    missing = [t["label"] for t in gloss if t["sign"] is None]
    return gloss, missing
