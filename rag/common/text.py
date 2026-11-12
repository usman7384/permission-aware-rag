import re
import unicodedata

_WHITESPACE = re.compile(r"\s+")
_TOKEN = re.compile(r"[^\W_]+")
_SENTENCE = re.compile(r".+?(?:[.!?]+\s+|\n+|\Z)", re.DOTALL)
PII = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+|(?<!\d)\d{13,19}(?!\d)")
_CONTRACTED_NOT = re.compile(r"n['\u2019]t\b", re.IGNORECASE)
NEGATIONS = frozenset(
    {"not", "no", "never", "without", "except", "false", "none", "nor", "neither", "cannot"}
)
INJECTION = re.compile(
    r"ignore (all |any |the )?(previous|prior|above) instructions"
    r"|disregard .{0,40}(rules|instructions)"
    r"|system prompt"
    r"|you are now",
    re.IGNORECASE,
)


def tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


def sentences(text: str) -> list[str]:
    return _SENTENCE.findall(text)


def normalize_query(q: str) -> str:
    q = _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", q).lower()).strip()
    return q.rstrip("?!.").rstrip()


def extract_entities(q: str) -> list[str]:
    found = set()
    for match in _TOKEN.finditer(q):
        token = match.group()
        upper_inside = any(c.isupper() for i, c in enumerate(token, start=match.start()) if i > 0)
        if upper_inside or any(c.isdigit() for c in token):
            found.add(token.lower())
    return sorted(found)


def negation_markers(q: str) -> list[str]:
    return sorted(set(tokens(_CONTRACTED_NOT.sub(" not", q))) & NEGATIONS)
