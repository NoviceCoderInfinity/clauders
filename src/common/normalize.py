"""Per-record normaliser for business_name and business_address.

norm_name(raw)  -> (name_clean, is_alias, is_domain, is_indic)
norm_addr(raw)  -> (addr_clean, nums, addr_null)
to_latin(name_clean, tmap) -> name_latin
"""
import re
import unicodedata

# ---------------------------------------------------------------- Latin accents
# Translate table built per code point over Latin-1 Supplement + Latin Extended-A/B only.
# Never NFKD the whole string: that would strip Indic vowel signs (matras).
_LATIN = {}
for cp in range(0xC0, 0x250):
    ch = chr(cp)
    base = "".join(c for c in unicodedata.normalize("NFKD", ch) if ord(c) < 128)
    if base and base != ch:
        _LATIN[cp] = base
_LATIN.update({ord("ß"): "ss", ord("æ"): "ae", ord("Æ"): "AE", ord("œ"): "oe", ord("Œ"): "OE",
               ord("ø"): "o", ord("Ø"): "O", ord("đ"): "d", ord("ł"): "l", ord("ı"): "i"})


def strip_latin_accents(s: str) -> str:
    return s.translate(_LATIN)


_WS = re.compile(r"\s+")

# ---------------------------------------------------------------- name rules
# Tag suffixes. Token reordering sometimes moves tags into the middle of the name
# (e.g. "75270) Foo (ID:"), so these match anywhere, not only at the end.
_TAGS = [
    re.compile(r"\[?\(?\s*www\.[^\s\]\)]+\s*\]?(\(https?://[^\)]*\))?\)?"),  # www.x.com, [www.x.com](https://www.x.com)
    re.compile(r"\|"),
    re.compile(r"\(ID:?\s*\d*\)?", re.I),
    re.compile(r"(?<![\w(])\d+\)"),          # orphaned "12345)" left by reordering
    re.compile(r"(?:\s-)?\s\d{10}\b"),       # " - 4725144658"
    re.compile(r"(?:^|\s)-(?=\s|$)"),        # orphaned " - "
    re.compile(r"#\d{3,5}\b"),               # " #52747"
]
_JUNK = re.compile(r"(?:^|\s)(?:\*\*\*|--|>>|<<|\.\.\.)(?=\s|$)")
_ALIAS = re.compile(
    r"(?:\s|^)(?:dba:?|d/b/a|d\.b\.a\.|doing business as|aka|a/k/a|fka|f/k/a|formerly known as|"
    r"formerly:|formerly|t/a|trading as|née)\s", re.I)
_HONOR = re.compile(r"^(?:m/s\.?|shri|sri|smt\.?|mr\.?|dr\.?|the)\s+", re.I)
_DOMAIN = re.compile(r"\.(?:com|net|org|co\.in|in|co|biz|info|io|fr)\b", re.I)
_PUNCT = re.compile(r"[()\[\]{}<>*\"_|:;!?]")
_EDGE = re.compile(r"^[\s\-.,/&@#+']+|[\s\-.,/&+']+$")


def _is_indic(s: str) -> bool:
    return any("ऀ" <= c <= "෿" for c in s)


def norm_name(raw):
    if raw is None:
        return "", 0, 0, 0
    s = raw
    for r in _TAGS:
        s = r.sub(" ", s)
    s = _JUNK.sub(" ", s)
    is_alias = 0
    m = _ALIAS.search(s)
    if m and s[m.end():].strip():
        s = s[m.end():]
        is_alias = 1
    s = _WS.sub(" ", s).strip()
    is_domain = int(bool(_DOMAIN.search(s)) or s[:1] in ("@", "#"))
    if is_domain:
        s = _DOMAIN.sub(" ", s).lstrip("@#")
    while True:
        s2 = _HONOR.sub("", s)
        if s2 == s:
            break
        s = s2
    s = strip_latin_accents(s.lower())
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    s = _EDGE.sub("", s)
    return s, is_alias, is_domain, int(_is_indic(s))


# ---------------------------------------------------------------- address rules
_PLACEHOLDER = {"", "null", "<null>", "n/a", "na", "none", "nil", "-"}
_STREET = re.compile(
    r"\b(?:st|street|rd|road|ave|avenue|av|blvd|boulevard|dr|drive|ln|lane|way|ct|court|pl|place|"
    r"pkwy|parkway|hwy|highway|ter|terrace|cir|circle|sq|square|marg|path|nagar|main|cross|rue|r|"
    r"chemin|allee|impasse|quai|cours|route)\b")
_NUMPFX = re.compile(r"(?:#|\bh\.?\s?no\b\.?|\bdoor\s+no\b\.?|\bno\b\.?|n°|nº)\s*")
_DIG = re.compile(r"\d+")


def norm_addr(raw):
    if raw is None:
        return "", "", 1
    comps = []
    for c in raw.split(","):
        c = _WS.sub(" ", strip_latin_accents(c.strip().lower()))
        if c in _PLACEHOLDER or c.startswith("pmb ") or c.startswith("po box"):
            continue
        if (c.startswith("unit ") or c.startswith("#")) and not _STREET.search(c):
            continue
        comps.append(c)
    addr = ", ".join(comps)
    nums = []
    for d in _DIG.findall(_NUMPFX.sub(" ", addr)):
        d = d.lstrip("0") or "0"
        if d not in nums:
            nums.append(d)
    return addr, " ".join(nums), int(not comps)


def to_latin(name_clean: str, tmap: dict) -> str:
    if not _is_indic(name_clean):
        return name_clean
    return " ".join(tmap.get(t, t) for t in name_clean.split(" "))
