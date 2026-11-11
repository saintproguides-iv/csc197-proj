"""
text_utils.py - shared text cleaning and perturbation helpers.

clean_text()  is used by train.py (on the training and validation titles) AND by
              inference.py (on whatever the user types), so the model always sees
              the same kind of text.

What clean_text does
  1. Runs of 3+ identical letters collapse to one letter:  "Developpp" -> "Develop",
     "workshhheet" -> "worksheet".  English words never have 3 identical letters in a
     row, so this is safe.  Side effects: "CCC100" -> "C100", "XXXX" -> "X", "III" -> "I".
     This is applied identically in training, so the model learns the same convention.
  2. Collapses repeated whitespace and trims the ends.
  3. Lowercases (optional).

add_typo / repeat_letter / change_case create realistic damage. They are used by
augment_typos.py (to teach the model that typos do not change a task) and by
robustness_check.py (to measure how much predictions move).
"""
import re

_REPEAT = re.compile(r"([A-Za-z])\1{2,}", re.IGNORECASE)
_SPACES = re.compile(r"\s+")

_NEIGHBORS = {
    "q": "wa", "w": "qes", "e": "wrd", "r": "etf", "t": "ryg", "y": "tuh", "u": "yij",
    "i": "uok", "o": "ipl", "p": "ol", "a": "qsz", "s": "awdx", "d": "sefc", "f": "drgv",
    "g": "fthb", "h": "gyjn", "j": "hukm", "k": "jil", "l": "kop", "z": "asx", "x": "zsdc",
    "c": "xdfv", "v": "cfgb", "b": "vghn", "n": "bhjm", "m": "njk",
}


def clean_text(text, lowercase=True):
    t = str(text)
    t = _REPEAT.sub(lambda m: m.group(1), t)
    t = _SPACES.sub(" ", t).strip()
    return t.lower() if lowercase else t


def _letter_positions(chars):
    return [i for i, c in enumerate(chars) if c.isalpha()]


def add_typo(text, rng, ops=("swap", "delete", "repeat", "substitute")):
    """Apply one random character-level typo to a letter. `rng` is a random.Random."""
    chars = list(text)
    pos = _letter_positions(chars)
    if len(pos) < 4:
        return text
    i = rng.choice(pos)
    op = rng.choice(ops)
    if op == "swap" and i + 1 < len(chars) and chars[i + 1].isalpha():
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
    elif op == "delete":
        del chars[i]
    elif op == "repeat":
        chars[i] = chars[i] * rng.choice([2, 3, 4])
    elif op == "substitute":
        nb = _NEIGHBORS.get(chars[i].lower())
        if nb:
            r = rng.choice(nb)
            chars[i] = r.upper() if chars[i].isupper() else r
    return "".join(chars)


def repeat_letter(text, rng):
    """Repeat one letter 3-5 times in a row (e.g. 'Developpp')."""
    chars = list(text)
    pos = _letter_positions(chars)
    if not pos:
        return text
    i = rng.choice(pos)
    chars[i] = chars[i] * rng.choice([3, 4, 5])
    return "".join(chars)


def change_case(text, mode):
    if mode == "lower":
        return text.lower()
    if mode == "upper":
        return text.upper()
    if mode == "title":
        return text.title()
    return text
