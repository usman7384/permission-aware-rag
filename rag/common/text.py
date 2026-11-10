import re
import unicodedata

_WHITESPACE = re.compile(r"\s+")
_TOKEN = re.compile(r"[^\W_]+")
_SENTENCE = re.compile(r".+?(?:[.!?]+\s+|\n+|\Z)", re.DOTALL)
PII = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+|(?<!\d)\d{13,19}(?!\d)")
INJECTION = re.compile(
    r"ignore (all |any |the )?(previous|prior|above) instructions"
    r"|disregard .{0,40}(rules|instructions)"
    r"|system prompt"
    r"|you are now",
    re.IGNORECASE,
)


def tokens(text: str) -> list[str]:
    """Lowercase alphanumeric runs."""
    return [t.lower() for t in _TOKEN.findall(text)]


def sentences(text: str) -> list[str]:
    """Pieces ending in `.`, `!` or `?` plus whitespace, a line break, or the end of the text."""
    return _SENTENCE.findall(text)


def normalize_query(q: str) -> str:
    q = _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", q).lower()).strip()
    return q.rstrip("?!.").rstrip()


def extract_entities(q: str) -> list[str]:
    """Sorted distinct tokens holding a digit, or an uppercase letter past the question's start."""
    found = set()
    for match in _TOKEN.finditer(q):
        token = match.group()
        upper_inside = any(c.isupper() for i, c in enumerate(token, start=match.start()) if i > 0)
        if upper_inside or any(c.isdigit() for c in token):
            found.add(token.lower())
    return sorted(found)
