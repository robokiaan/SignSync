// JS port of app/gloss_matching.py - keep these in sync by hand, there's no
// shared source of truth once the static (GitHub Pages) build stopped
// running the Python version server-side. See that file's docstring for the
// full description: English -> ISL gloss by rule (drop function words, NOT
// and question words to the end, TIME / nouns / adjective / VERB order,
// pronoun and plural and verb normalisation), then every gloss token whose
// sign the dictionary lacks is reported as missing.

const FUNCTION_WORDS = new Set([
    "a", "an", "the",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "will", "would", "shall", "should", "can", "could",
    "may", "might", "must",
    "to", "of", "and", "or", "but", "so", "if", "as", "at", "by", "for",
    "in", "on", "with", "from", "into",
    "my", "your", "his", "her", "its", "our", "their",
    "this", "that", "these", "those", "there", "some", "any",
    "very", "really", "please",
]);

const NEGATION_WORDS = new Set(["not", "no", "never", "nothing", "none", "nobody"]);

const WH_WORDS = new Set(["what", "where", "who", "whom", "when", "why", "how", "which"]);

const PRONOUN_MAP = { i: "i", me: "i", him: "he", us: "we", them: "they" };

const VERB_LEMMAS = new Set([
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
]);
const IRREGULAR_VERBS = {
    went: "go", gone: "go", goes: "go", came: "come", ate: "eat",
    eaten: "eat", drank: "drink", drunk: "drink", has: "have",
    had: "have", saw: "see", seen: "see", knew: "know", known: "know",
    wrote: "write", written: "write", gave: "give", given: "give",
    took: "take", taken: "take", bought: "buy", sold: "sell", made: "make",
    ran: "run", sat: "sit", stood: "stand", heard: "hear", spoke: "speak",
    said: "say", thought: "think", felt: "feel", got: "get", brought: "bring",
    met: "meet", drove: "drive", rode: "ride", understood: "understand",
    forgot: "forget", told: "tell", found: "find", lost: "lose", won: "win",
    wore: "wear", held: "hold", fell: "fall", flew: "fly", sang: "sing",
    paid: "pay", sent: "send", left: "leave", slept: "sleep", cried: "cry",
};

// Typed without the apostrophe ("dont", "cant") - common on phones - these
// would otherwise fall through as unknown nouns and be glossed as DONT.
const BARE_CONTRACTIONS = {
    dont: "do not", doesnt: "does not", didnt: "did not",
    isnt: "is not", arent: "are not", wasnt: "was not", werent: "were not",
    cant: "can not", wont: "will not", couldnt: "could not",
    wouldnt: "would not", shouldnt: "should not", mustnt: "must not",
    havent: "have not", hasnt: "has not", hadnt: "had not",
    im: "i am", youre: "you are", theyre: "they are",
    ive: "i have", youve: "you have", weve: "we have", theyve: "they have",
};

const CONTRACTIONS = {
    "can't": "can not", "cannot": "can not", "won't": "will not",
    "i'm": "i am", "you're": "you are", "we're": "we are", "they're": "they are",
    "he's": "he is", "she's": "she is", "it's": "it is",
    "i've": "i have", "you've": "you have", "we've": "we have", "they've": "they have",
    "i'll": "i will", "you'll": "you will", "we'll": "we will", "they'll": "they will",
    "he'll": "he will", "she'll": "she will", "it'll": "it will",
    "i'd": "i would", "you'd": "you would", "we'd": "we would", "they'd": "they would",
    "let's": "let us",
};

// Sentence-initial temporal adverbs. Only these move to the front - the rest of
// the "Days And Time" category (week, month, hour, time...) are ordinary nouns.
const TIME_WORDS = new Set([
    "today", "tomorrow", "yesterday", "morning", "afternoon", "evening", "night",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
]);

// Dictionary categories whose signs act as the sentence's predicate (comment),
// which ISL places after the noun(s) it describes. "colour" itself is a noun.
const PREDICATE_CATEGORIES = new Set(["Adjectives", "Colours"]);
const PREDICATE_EXCEPTIONS = new Set(["colour"]);

// Gloss order classes. Stable sort within a class keeps the English order,
// which already puts subject before object.
const CLS_TIME = 0, CLS_NOUN = 1, CLS_PREDICATE = 2, CLS_VERB = 3, CLS_NOT = 4, CLS_WH = 5;

const WORD_RE = /[a-z0-9]+/g;

function normalize(text) {
    let t = text.toLowerCase();
    t = t.replace(/[\u2018\u2019\u02bc`]/g, "'"); // curly / typographic apostrophes
    t = t.replace(/n't\b/g, " not"); // don't / isn't / haven't ...
    t = t.replace(/\b[a-z]+\b/g, (w) => (Object.prototype.hasOwnProperty.call(BARE_CONTRACTIONS, w) ? BARE_CONTRACTIONS[w] : w));
    for (const [k, v] of Object.entries(CONTRACTIONS)) t = t.split(k).join(v);
    // Possessive 's carries no sign (the possessor simply precedes the
    // possessed), so strip it or "friend's" leaves a stray "s" token.
    t = t.replace(/'s\b/g, "");
    return t;
}

function tokenize(text) {
    return normalize(text).match(WORD_RE) || [];
}

function aliasesFor(signName) {
    const names = new Set([signName]);
    if (signName.includes("(")) {
        names.add(signName.split("(")[0].trim());
    }
    if (signName.includes(" or ")) {
        for (const part of signName.split(" or ")) {
            names.add(part.trim());
        }
    }
    if (signName === "big large" || signName === "small little") {
        for (const word of signName.split(" ")) names.add(word);
    }
    if (signName.startsWith("ex. ")) {
        names.add(signName.slice(4).trim());
    }
    if (signName.includes("-")) {
        names.add(signName.replace(/-/g, " "));
        names.add(signName.replace(/-/g, ""));
    }
    return names;
}

// signNames: array of lowercase dictionary sign names.
// Returns Map<string /* tokens.join(" ") */, string /* canonical sign name */>.
// On alias collisions (e.g. bare "you" from both "you" and "you (plural)"),
// the plainer/shorter canonical name wins, since it's processed first.
function buildAliasIndex(signNames) {
    const index = new Map();
    const ordered = [...signNames].sort((a, b) => {
        const aHasParen = a.includes("(") ? 1 : 0;
        const bHasParen = b.includes("(") ? 1 : 0;
        if (aHasParen !== bHasParen) return aHasParen - bHasParen;
        return a.length - b.length;
    });
    for (const name of ordered) {
        for (const alias of aliasesFor(name)) {
            const tokens = alias.toLowerCase().match(WORD_RE) || [];
            if (tokens.length === 0) continue;
            const key = tokens.join(" ");
            if (!index.has(key)) index.set(key, name);
        }
    }
    return index;
}

function singular(word) {
    if (word.endsWith("ies") && word.length > 4) return word.slice(0, -3) + "y";
    if (word.endsWith("es") && word.length > 3) return word.slice(0, -2);
    if (word.endsWith("s") && !word.endsWith("ss") && word.length > 2) return word.slice(0, -1);
    return null;
}

function verbLemma(word) {
    if (VERB_LEMMAS.has(word)) return word;
    if (word in IRREGULAR_VERBS) return IRREGULAR_VERBS[word];
    for (const suffix of ["ing", "ed", "es", "s"]) {
        if (word.endsWith(suffix)) {
            const base = word.slice(0, -suffix.length);
            if (VERB_LEMMAS.has(base)) return base;
            if (suffix === "ing" && VERB_LEMMAS.has(base + "e")) return base + "e"; // coming, making
            if (suffix === "ed" && VERB_LEMMAS.has(base + "e")) return base + "e";  // liked, loved
        }
    }
    return null;
}

function signClass(signName, category) {
    if (TIME_WORDS.has(signName)) return CLS_TIME;
    if (PREDICATE_CATEGORIES.has(category) && !PREDICATE_EXCEPTIONS.has(signName)) return CLS_PREDICATE;
    return CLS_NOUN;
}

// English -> ISL gloss. Returns {gloss, missing}: gloss is the ordered list of
// {label, sign} tokens (sign null when the dictionary has no such sign),
// missing is the labels with no sign, in gloss order. index from
// buildAliasIndex, categoryByName: Map<signName, category>.
function toGloss(text, index, categoryByName) {
    const words = tokenize(text);
    let maxPhrase = 1;
    for (const key of index.keys()) {
        const len = key.split(" ").length;
        if (len > maxPhrase) maxPhrase = len;
    }
    const signToken = (name) => ({ label: name, sign: name, cls: signClass(name, categoryByName.get(name)) });
    const lookup = (name) => (index.has(name) ? index.get(name) : null);

    const tokens = [];
    let i = 0;
    const n = words.length;
    while (i < n) {
        // Longest dictionary phrase first - "good morning", "train station",
        // "how are you" are single signs even though they contain function
        // words and question words.
        let matched = false;
        for (let span = Math.min(maxPhrase, n - i); span > 0; span--) {
            const phrase = words.slice(i, i + span).join(" ");
            if (index.has(phrase)) {
                tokens.push(signToken(index.get(phrase)));
                i += span;
                matched = true;
                break;
            }
        }
        if (matched) continue;

        const w = words[i];
        i += 1;
        if (NEGATION_WORDS.has(w)) {
            tokens.push({ label: "not", sign: lookup("not"), cls: CLS_NOT });
        } else if (WH_WORDS.has(w)) {
            tokens.push({ label: w, sign: lookup(w), cls: CLS_WH });
        } else if (w in PRONOUN_MAP) {
            const name = PRONOUN_MAP[w];
            tokens.push({ label: name, sign: lookup(name), cls: CLS_NOUN });
        } else if (FUNCTION_WORDS.has(w)) {
            continue;
        } else {
            const lemma = verbLemma(w);
            if (lemma) {
                tokens.push({ label: lemma, sign: lookup(lemma), cls: CLS_VERB });
                continue;
            }
            const base = singular(w);
            if (base && index.has(base)) {
                tokens.push(signToken(index.get(base)));
                continue;
            }
            tokens.push({ label: w, sign: null, cls: CLS_NOUN });
        }
    }

    const ordered = tokens
        .map((t, idx) => ({ ...t, idx }))
        .sort((a, b) => a.cls - b.cls || a.idx - b.idx)
        .map((t) => ({ label: t.label, sign: t.sign }));
    const missing = ordered.filter((t) => t.sign === null).map((t) => t.label);
    return { gloss: ordered, missing };
}
