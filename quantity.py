"""
quantity.py - read a repeat count from a task title and scale the estimate.

Why: the model ignores numbers ("1 vs 100 sorting problems" get the same hours).
How: find "<number> <up to 2 words> <plural noun>", ask the model about ONE unit written with a
     "1" ("Solve 5 sorting problems" -> "Solve 1 sorting problem"), then
         total = unit_hours * n ** alpha
Using "1" (not "a") makes the one-unit estimate equal to what the model gives for a single item,
so the estimates are continuous: n = 1 gives the unit estimate, n = 5 gives it times 5 ** alpha.
alpha is an ASSUMPTION, not a fitted value. alpha = 1.0 means every unit takes equally long;
below 1.0 means later units are faster (shared setup, practice). We have no data to fit it.

Counts:        "5 problems", "five essays", "a dozen worksheets", "a couple of labs"
Not counts:    "10-page report", "2D game", "CRUD app with 5 tables", "v1.10", "CCC100",
               "study for 3 hours", numbers before size or time units (pages, slides, words, hours...),
               years and ids ("February 2012 bug fixes", "task 5026")

This is only used in the interactive demo (predict.py). evaluate_test.py does not use it,
so the reported test scores are not affected.
"""
import re

WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
         "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
         "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
         "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "hundred": 100}
PHRASES = {("a", "couple", "of"): 2, ("a", "pair", "of"): 2, ("a", "dozen"): 12, ("a", "couple"): 2}

# nouns that measure SIZE or TIME, not repeated copies of the whole task
SIZE_TIME = {"page", "slide", "word", "line", "character", "token", "minute", "hour", "day", "week",
             "month", "year", "semester", "kb", "mb", "gb", "percent"}
# a number right after one of these describes a part or a limit, not a repeat count
BLOCK_BEFORE = {"with", "of", "in", "for", "using", "across", "over", "within", "under", "at", "than",
                "from", "by", "to", "after", "before", "per", "about", "around", "last", "next", "top",
                # a number after these is an identifier ("Version 2", "Lab 3", "Task 5"), not a count
                "version", "v", "release", "phase", "stage", "part", "chapter", "lab", "week", "day", "level",
                "grade", "section", "figure", "step", "task", "ticket", "issue", "sprint", "module", "unit"}
MAX_N = 500


def _strip(tok):
    return re.sub(r"^[^\w]+|[^\w]+$", "", tok)


def _looks_plural(word):
    w = word.lower()
    if word.isupper():                      # acronyms such as SCMS or USM are not plural nouns
        return False
    return len(w) >= 4 and w.endswith("s") and not w.endswith(("ss", "us", "is", "ics"))


def singular(word):
    w = word
    lw = w.lower()
    if lw.endswith("ies") and len(lw) > 4:
        return w[:-3] + ("Y" if w[-3:].isupper() else "y")
    if lw.endswith(("sses", "xes", "ches", "shes", "zes")):
        return w[:-2]
    return w[:-1]


def parse_quantity(text):
    """Return {"n", "unit_text", "noun", "matched"} for the first repeat count found, else None."""
    toks = str(text).split()
    for i in range(len(toks)):
        core = _strip(toks[i]).lower()
        n, used = None, 1
        for ph, val in PHRASES.items():                      # "a dozen", "a couple of"
            if tuple(_strip(t).lower() for t in toks[i:i + len(ph)]) == ph:
                n, used = val, len(ph)
                break
        if n is None and core.isdigit() and toks[i].strip(",.;:") == core:   # a pure number (not 2D, 10-page, v1.10)
            n = int(core)
        elif n is None and core in WORDS and core != "one":
            n = WORDS[core]
        if n is None or n < 2:
            continue
        if n > 1000 or 1900 <= n <= 2100:      # ids and years ("Task 5026", "February 2012"), not counts
            continue
        prev = _strip(toks[i - 1]).lower() if i > 0 else ""
        if prev in BLOCK_BEFORE:
            continue
        # the plural noun must follow within 3 words
        j0 = i + used
        for j in range(j0, min(j0 + 3, len(toks))):
            word = _strip(toks[j])
            if _looks_plural(word):
                sing = singular(word)
                if sing.lower() in SIZE_TIME or word.lower() in SIZE_TIME:
                    break
                art = "1"      # same phrasing the model sees for a single item ("Solve 1 sorting problem")
                head = toks[:i]
                tail_pre = toks[j0:j]
                tail = toks[j + 1:]
                suffix = toks[j][len(word) + toks[j].index(word):] if word in toks[j] else ""
                unit = " ".join(head + [art] + tail_pre + [sing + suffix] + tail)
                return {"n": min(n, MAX_N), "unit_text": unit, "noun": sing,
                        "matched": " ".join(toks[i:j + 1])}
    return None


def estimate_with_quantity(predict_one, text, alpha=0.8):
    """predict_one: str -> hours (float). Returns a dict with the breakdown."""
    q = parse_quantity(text)
    if q is None:
        h = float(predict_one(text))
        return {"total": round(h, 2), "n": 1, "unit_hours": round(h, 2), "unit_text": text, "alpha": alpha, "used_rule": False}
    unit = float(predict_one(q["unit_text"]))
    total = unit * (q["n"] ** alpha)
    return {"total": round(total, 2), "n": q["n"], "unit_hours": round(unit, 2), "unit_text": q["unit_text"],
            "noun": q["noun"], "alpha": alpha, "used_rule": True}
