// JS port of app/gloss_matching.py - keep these in sync by hand, there's no
// shared source of truth once the static (GitHub Pages) build stopped
// running the Python version server-side. See that file's docstring for the
// "not a translator" design note; logic below mirrors it exactly.

const STOPWORDS = new Set([
    "a", "an", "the", "is", "am", "are", "was", "were", "be", "been", "being",
    "to", "of", "and", "or", "but", "so", "if", "as", "at", "by", "for",
    "in", "on", "with", "my", "your", "his", "her", "its", "our", "their",
    "this", "that", "these", "those", "do", "does", "did", "has", "have", "had",
]);

const WORD_RE = /[a-z0-9]+/g;

function tokenize(text) {
    return (text.toLowerCase().match(WORD_RE)) || [];
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
            const tokens = tokenize(alias);
            if (tokens.length === 0) continue;
            const key = tokens.join(" ");
            if (!index.has(key)) index.set(key, name);
        }
    }
    return index;
}

// Greedy longest-match tokenizer against `index` (from buildAliasIndex).
// Returns {gloss, unmatched}. gloss is the ordered list of matched dictionary
// sign names (duplicates allowed). unmatched is content words (not in
// STOPWORDS) that matched nothing - for a soft warning, not an error.
function parseSentence(text, index) {
    const words = tokenize(text);
    let maxPhrase = 1;
    for (const key of index.keys()) {
        const len = key.split(" ").length;
        if (len > maxPhrase) maxPhrase = len;
    }

    const gloss = [];
    const unmatched = [];
    let i = 0;
    const n = words.length;
    while (i < n) {
        let matched = false;
        for (let span = Math.min(maxPhrase, n - i); span > 0; span--) {
            const phrase = words.slice(i, i + span).join(" ");
            if (index.has(phrase)) {
                gloss.push(index.get(phrase));
                i += span;
                matched = true;
                break;
            }
        }
        if (!matched) {
            const word = words[i];
            if (!STOPWORDS.has(word)) unmatched.push(word);
            i += 1;
        }
    }

    return { gloss, unmatched };
}
