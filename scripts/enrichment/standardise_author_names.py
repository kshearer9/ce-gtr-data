"""
standardise_author_names.py
===========================
Put every author name into one consistent format, without ever asserting that
two names are the same person.

Design rule
-----------
FORMATTING ONLY. This script never merges rows, never collapses variants and
never infers that "Ji, S." and "Ji, Shouxun" are the same researcher. It takes
each author record as its source supplied it and renders it consistently.

That rule exists because the two possible errors are not equally bad. Merging
"Zhang, Jianguo" and "Zhang, Jianhui" publishes a false claim about two real
named people. Leaving one person as two entries is untidy but true. For a
public-facing output the second is much the lesser harm, so nothing here
depends on guessing.

Where the forenames come from
-----------------------------
Each source already supplies structured name parts that the collectors were
discarding in favour of a pre-flattened string:

  WoS      first_name and last_name on 99.6% of author slots
  Scopus   ce:given-name on 82.7% of slots (the collector kept only
           ce:indexed-name, ie "Smith J.", throwing the forename away)
  OpenAlex full display names on effectively all slots
  GtR      a single author string, usually already "Surname Initials"

So most rows can be rendered with a real forename using only what that row's
own source provided. No forename is ever borrowed from another source, because
doing so would require deciding that two records describe the same person,
which is exactly the judgement this script refuses to make.

Casing
------
Tokens that are ALL CAPS or all lowercase are re-cased. Tokens that are already
mixed case are left exactly as they are, so MacLeod, McDonald, O'Brien and
van der Berg survive intact rather than being flattened to Macleod. Accents and
diacritics are preserved: Müller stays Müller.

Input:
  data/cleaned/authors/authors_long.csv   (from harvest_author_identifiers.py)
  data/cleaned/outcomes/gtr_all_outcomes_clean.csv

Output:
  data/cleaned/authors/authors_standardised.csv

Run from the repository root:
    python scripts/enrichment/standardise_author_names.py
"""

from pathlib import Path
import re
import sys
import unicodedata

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
AUTHORS_DIR = ROOT_DIR / "data" / "cleaned" / "authors"
CLEANED_DIR = ROOT_DIR / "data" / "cleaned" / "outcomes"
OUTPUT_PATH = AUTHORS_DIR / "authors_standardised.csv"
EXCEL_PATH = AUTHORS_DIR / "authors_standardised_excel.csv"

LONG_PATH = AUTHORS_DIR / "authors_long.csv"
GTR_PATH = CLEANED_DIR / "gtr_all_outcomes_clean.csv"

# Name particles, kept lowercase when a token is being re-cased.
PARTICLES = {
    "van", "von", "der", "den", "de", "del", "della", "di", "da", "dos", "das",
    "du", "la", "le", "el", "al", "bin", "ibn", "ter", "ten", "op", "zu",
}

# Dropped from names entirely.
HONORIFICS = {"dr", "prof", "professor", "mr", "mrs", "ms", "miss", "sir"}

# Kept, but never treated as a forename.
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "phd", "md", "msc", "bsc"}


# ---------------------------------------------------------------------------
# CASING
# ---------------------------------------------------------------------------

def case_token(token, allow_particle=True):
    """
    Re-case a single token, preserving any casing the source already applied.

    "SMITH"     -> "Smith"        (all caps, re-cased)
    "smith"     -> "Smith"        (all lower, re-cased)
    "MacLeod"   -> "MacLeod"      (mixed, left alone)
    "O'BRIEN"   -> "O'Brien"      (re-cased across the apostrophe)
    "SMITH-JONES" -> "Smith-Jones"
    "VAN"       -> "van"          (recognised particle)
    """
    if not token:
        return ""

    # Already carries deliberate internal casing, so do not touch it.
    if not token.isupper() and not token.islower():
        return token

    bare = token.strip(".").lower()
    if allow_particle and bare in PARTICLES:
        return bare

    def capitalise_segment(segment):
        return segment[:1].upper() + segment[1:].lower() if segment else segment

    # Split on every separator at once, keeping the separators. Doing one pass
    # per separator would undo the previous pass, turning "Smith-Jones" into
    # "Smith-jones" and "O'Brien" into "O'brien".
    parts = re.split(r"([-'’])", token.lower())
    return "".join(
        part if part in "-'’" else capitalise_segment(part) for part in parts)


def is_initials_token(token):
    """
    Is this token one or more initials? "J", "JA", "Y.C." yes; "Ann", "Wei" no.

    Full stops between single letters are decisive whatever the length, so
    "J.A.B." is caught while "ANN" is not.
    """
    bare = token.replace(".", "").replace("’", "").replace("'", "")
    if not bare:
        return False
    # A single CJK character is a whole name, not an initial, so it must never
    # gain a full stop: 真 is not "真.".
    if not is_latin(bare):
        return False
    if "." in token:
        segments = [p for p in token.split(".") if p]
        if segments and all(len(p) == 1 for p in segments):
            return True
    if len(bare) == 1:
        return True
    return len(bare) <= 2 and bare.isupper()


def expand_initials(token):
    """"JA" -> ["J.", "A."];  "Y.C." -> ["Y.", "C."];  "n" -> ["N."]"""
    bare = token.replace(".", "").replace("’", "").replace("'", "")
    return [f"{character.upper()}." for character in bare]


def case_name_part(text, allow_particle=True):
    """Re-case every token in a name fragment."""
    if not text:
        return ""
    return " ".join(
        case_token(token, allow_particle) for token in str(text).split() if token)


# ---------------------------------------------------------------------------
# PARSING
# ---------------------------------------------------------------------------

ORCID_PATTERN = re.compile(r"\d{4}-\d{4}-\d{4}-\d{3}[\dXx]")
EMBEDDED_ID_PATTERN = re.compile(
    r"[;,]?\s*(?:id_)?orcid[:\s]*\d{4}-\d{4}-\d{4}-\d{3}[\dXx]\s*", re.IGNORECASE)

# Scripts that the "Surname, Forename" convention does not describe well.
NON_LATIN_PATTERN = re.compile(
    r"[぀-ヿ㐀-䶿一-鿿가-힯Ѐ-ӿͰ-Ͽ؀-ۿ]")

# Unicode dash variants. Sources mix U+2010 HYPHEN, U+2011, U+2013 EN DASH and
# others into names, eg "Guy‐Bart Stan". They are folded to an ASCII hyphen so
# the hyphen survives tokenising and so the same name renders identically
# whichever dash its source happened to use.
DASHES = dict.fromkeys(
    [0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2015, 0x2212, 0xFE63, 0xFF0D],
    "-")


def fold_dashes(text):
    """Convert every Unicode dash variant to an ASCII hyphen."""
    if text is None or pd.isna(text):
        return ""
    return str(text).translate(DASHES)


# Letters that NFKD does not decompose, so they need mapping by hand when
# comparing an unaccented indexed name against an accented full name.
EXTRA_FOLDS = str.maketrans({
    "ı": "i", "İ": "i", "ł": "l", "Ł": "l", "ø": "o", "Ø": "o",
    "đ": "d", "Đ": "d", "ß": "s", "æ": "ae", "Æ": "ae", "œ": "oe", "Œ": "oe",
})


def fold_for_match(text):
    """
    Accent- and case-insensitive key, for COMPARISON only.

    Never used to build a display value, so no accent is ever lost from output.
    """
    folded = unicodedata.normalize("NFKD", fold_dashes(text).lower())
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return folded.translate(EXTRA_FOLDS).strip()


def strip_embedded_identifiers(raw):
    """
    Remove an ORCID that has leaked into a name string, and return it.

    OpenAlex supplies a handful of display names such as
    "Wooles, Ashley; id_orcid 0000-0001-7411-9627" and
    "Louise S.; id_orcid 0000-0002-9451-3557 Natrajan", where the identifier is
    embedded in the middle of the name. Left alone these render as
    "Wooles, Ashley Id_orcid 0000-0001-7411-9627".

    Returns (cleaned_name, recovered_orcid).
    """
    # NOT str(raw or ""): a float NaN is truthy, so that yields the literal
    # string "nan", which then reads as a forename.
    text = fold_dashes(raw)
    found = ORCID_PATTERN.search(text)
    recovered = found.group(0).upper() if found else ""

    cleaned = EMBEDDED_ID_PATTERN.sub(" ", text)
    # Any bare ORCID left behind, plus the label if it was separated.
    cleaned = ORCID_PATTERN.sub(" ", cleaned)
    cleaned = re.sub(r"(?i)\bid_orcid\b", " ", cleaned)
    cleaned = re.sub(r"\s*;\s*", "; ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ;,")

    return cleaned, recovered


def flag_name(raw, display):
    """Mark rows a human should look at. Empty string means nothing to see."""
    text = str(raw or "")
    flags = []
    if ORCID_PATTERN.search(text) or re.search(r"(?i)id_orcid", text):
        flags.append("identifier_in_name")
    if NON_LATIN_PATTERN.search(text):
        flags.append("non_latin_script")
    if re.search(r"\d", str(display or "")):
        flags.append("digits_remain")
    if len(str(display or "")) > 60:
        flags.append("unusually_long")
    return "; ".join(flags)


def clean_tokens(text):
    """
    Split into tokens, dropping honorifics, stray punctuation and bare numbers.

    Bare numbers appear as footnote markers, eg "Hogg-Johnson 2, S.", and are
    never part of a name.
    """
    if text is None or pd.isna(text) or not str(text):
        return []
    cleaned = re.sub(r"\([^)]*\)", " ", fold_dashes(text))
    cleaned = re.sub(r"[^\w\s,\-'.’]", " ", cleaned)
    tokens = []
    for token in cleaned.replace(",", " ").split():
        bare = token.strip(".")
        if bare.lower() in HONORIFICS:
            continue
        if bare.isdigit():
            continue
        tokens.append(token)
    return tokens


def is_latin(text):
    """False for CJK and other non-Latin scripts, where initials make no sense."""
    return not NON_LATIN_PATTERN.search(str(text or ""))


def is_initial(token):
    """
    Is this token an initial rather than a name?

    Delegates to is_initials_token so the two never disagree. They previously
    did: is_initial capped the length at 2, so "J.M.F." was not recognised as
    initials and "Mendoza J.M.F." was parsed with J.M.F. as the surname,
    producing "J.m.f., Mendoza".
    """
    return is_initials_token(token)


def split_raw_name(raw):
    """
    Split an unstructured name string into (given, surname).

    Used only where the source gave no structured parts, which in practice
    means GtR. Handles the comma form, trailing initials and particles.
    """
    text = str(raw or "").strip()
    if not text:
        return "", ""

    if "," in text:
        surname_part, _, given_part = text.partition(",")
        return " ".join(clean_tokens(given_part)), " ".join(clean_tokens(surname_part))

    tokens = [t for t in clean_tokens(text)
              if t.strip(".").lower() not in SUFFIXES]
    if not tokens:
        return "", ""
    if len(tokens) == 1:
        return "", tokens[0]

    # Trailing initials mean the surname leads: "Smith J", "van der Berg M".
    cut = len(tokens)
    while cut > 1 and is_initial(tokens[cut - 1]):
        cut -= 1
    if cut < len(tokens):
        return " ".join(tokens[cut:]), " ".join(tokens[:cut])

    # Otherwise natural order, absorbing particles into the surname.
    surname_tokens = [tokens[-1]]
    index = len(tokens) - 2
    while index >= 0 and tokens[index].strip(".").lower() in PARTICLES:
        surname_tokens.insert(0, tokens[index])
        index -= 1
    return " ".join(tokens[: index + 1]), " ".join(surname_tokens)


def resolve_parts(given, surname, raw):
    """
    Work out (given, surname) for a row, using the structured fields the source
    supplied and falling back to the raw string only to fill gaps.

    Shared by format_display and classify so the rendered name and the
    completeness label can never disagree.
    """
    given = "" if pd.isna(given) else str(given).strip()
    surname = "" if pd.isna(surname) else str(surname).strip()

    if not surname:
        given, surname = split_raw_name(raw)
    elif not given:
        # Scopus supplies ce:surname but not always ce:given-name, while the
        # indexed name it also supplies carries the forename or initials.
        # Recover them rather than rendering a bare surname.
        #
        # Knowing the surname also resolves the word order, which is otherwise
        # ambiguous: "Zhang Jianguo" parses as given "Zhang", surname
        # "Jianguo", but if the source says the surname is Zhang then the
        # remainder must be the forename. This matters most for Chinese names,
        # which is exactly where fragmentation is worst.
        parsed_given, parsed_surname = split_raw_name(raw)
        # Compare accent-folded. The indexed name is often unaccented
        # ("Patay-Horvath A.") while the structured surname is not
        # ("Patay-Horváth"); an exact comparison fails and the initial is lost.
        target = fold_for_match(surname)
        if parsed_given and fold_for_match(parsed_surname) == target:
            given = parsed_given
        elif parsed_surname and fold_for_match(parsed_given) == target:
            given = parsed_surname

    # A particle belongs with the surname, not the forename. "Ehecatl Antonio
    # del" + "Rio-Chanona" must become "Ehecatl Antonio" + "del Rio-Chanona",
    # otherwise it renders as "Rio-Chanona, Ehecatl Antonio Del".
    given_tokens = given.split()
    moved = []
    while given_tokens and given_tokens[-1].strip(".").lower() in PARTICLES:
        moved.insert(0, given_tokens.pop())
    if moved and surname:
        given = " ".join(given_tokens)
        surname = " ".join(moved + [surname])

    return given, surname


def format_display(given, surname, raw):
    """
    Render "Surname, First Middle".

    Returns the tidied raw string if there is no surname to work with.
    """
    given, surname = resolve_parts(given, surname, raw)

    surname_cased = case_name_part(surname)
    given_tokens = [t for t in clean_tokens(given)
                    if t.strip(".").lower() not in SUFFIXES]

    # Initials are detected on the ORIGINAL token, before casing, because
    # case_token would turn "JA" into "Ja" and hide that it is two initials.
    # Particles inside a forename are rare and usually wrong, so they are not
    # lowercased there.
    rendered = []
    for token in given_tokens:
        if is_initials_token(token):
            rendered.extend(expand_initials(token))
        else:
            rendered.append(case_token(token, allow_particle=False))
    given_cased = " ".join(rendered)

    # Sources occasionally carry a stray leading dash, eg "-Green, Morgan
    # Reynolds", which is a truncation artefact rather than part of the name.
    surname_cased = surname_cased.strip("-  ")
    given_cased = given_cased.strip("-  ")

    if surname_cased and given_cased:
        return f"{surname_cased}, {given_cased}"
    if surname_cased:
        return surname_cased
    return case_name_part(fold_dashes(raw).strip())


def classify(given, surname, raw):
    """Record how much real name information this row actually has."""
    given, surname = resolve_parts(given, surname, raw)
    if not surname:
        return "unparsed"
    tokens = [t for t in clean_tokens(given)
              if t.strip(".").lower() not in SUFFIXES]
    if not tokens:
        return "surname_only"
    if any(len(t.replace(".", "")) > 2 for t in tokens):
        return "full_forename"
    return "initials_only"


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def load_gtr():
    """GtR outcome authors, which have no structured name parts anywhere."""
    if not GTR_PATH.exists():
        print(f"  GtR file not found at {GTR_PATH}, skipped")
        return pd.DataFrame()

    df = pd.read_csv(GTR_PATH, low_memory=False)
    column = "author" if "author" in df.columns else None
    if column is None:
        print("  GtR has no author column, skipped")
        return pd.DataFrame()

    blank = pd.Series([""] * len(df))
    rows = []
    for outcome_id, project_id, doi, value in zip(
            df.get("outcome_id", blank),
            df.get("project_id", blank),
            df.get("doi", blank),
            df[column]):
        if pd.isna(value):
            continue
        outcome_id = "" if pd.isna(outcome_id) else str(outcome_id)
        doi = "" if pd.isna(doi) or str(doi).strip().lower() in {"", "nan"} \
            else str(doi).strip().lower()
        for position, name in enumerate(
                [n.strip() for n in re.split(r"[;|]", str(value)) if n.strip()], 1):
            rows.append({
                "source": "gtr",
                # GtR's outcome id is already the id the cleaners use, so the
                # key and the id are the same value here.
                "outcome_id": outcome_id,
                "outcome_key": outcome_id,
                "doi": doi,
                "author_position": position,
                "raw_name": name,
                "given_name": "",
                "surname": "",
                "orcid": "",
                "native_id": "",
                "native_id_type": "",
                "project_ids": "" if pd.isna(project_id) else str(project_id),
            })
    print(f"  gtr:      {len(rows):,} author rows")
    return pd.DataFrame(rows)


def main():
    print(f"Repository root: {ROOT_DIR}\n")
    print("Loading author records...")

    frames = []
    if LONG_PATH.exists():
        harvested = pd.read_csv(LONG_PATH, low_memory=False)
        keep = ["source", "outcome_id", "outcome_key", "doi", "author_position",
                "raw_name", "given_name", "surname", "orcid", "native_id",
                "native_id_type", "project_ids"]
        for column in keep:
            if column not in harvested.columns:
                harvested[column] = ""
        harvested = harvested[keep]
        for source, group in harvested.groupby("source"):
            print(f"  {source + ':':<9} {len(group):>7,} author rows")
        frames.append(harvested)
    else:
        print(f"  {LONG_PATH} not found. Run harvest_author_identifiers.py first.")

    gtr = load_gtr()
    if len(gtr):
        frames.append(gtr)

    if not frames:
        print("\nNothing to standardise.")
        return

    df = pd.concat(frames, ignore_index=True)
    row_count_in = len(df)

    print(f"\nFormatting {row_count_in:,} names...")

    # Strip identifiers that have leaked into name strings, keeping any ORCID
    # they carried, before anything is parsed or cased.
    # The structured given_name and surname were parsed upstream from the same
    # dirty string, so they carry the contamination too and must be cleaned as
    # well. Cleaning raw_name alone leaves the identifier in the display,
    # because format_display prefers the structured parts when they exist.
    cleaned_raw, recovered_orcids = [], []
    for raw in df["raw_name"]:
        cleaned, recovered = strip_embedded_identifiers(raw)
        cleaned_raw.append(cleaned)
        recovered_orcids.append(recovered)

    cleaned_given = [strip_embedded_identifiers(v)[0] for v in df["given_name"]]
    cleaned_surname = [strip_embedded_identifiers(v)[0] for v in df["surname"]]
    df["given_name"] = cleaned_given
    df["surname"] = cleaned_surname

    # OpenAlex publishes only a single display_name, so the given/surname
    # columns for those rows were GUESSED by the harvest's simple splitter
    # rather than supplied by the source. That splitter mishandles particles
    # and multi-letter initials, giving "Rio-Chanona, Ehecatl Antonio Del" and
    # "J.k., Penhaul Smith". Discard the guess and let split_raw_name, which
    # handles both, derive it from the display name instead.
    guessed = df["source"] == "openalex"
    n_guessed = int(guessed.sum())
    df.loc[guessed, "given_name"] = ""
    df.loc[guessed, "surname"] = ""
    if n_guessed:
        print(f"  re-parsed {n_guessed:,} OpenAlex names from the display "
              f"string (the source has no separate name fields)")

    n_cleaned = sum(
        1 for original, cleaned in zip(df["raw_name"], cleaned_raw)
        if str(original) != cleaned)
    n_identifiers = sum(1 for value in recovered_orcids if value)
    if n_cleaned:
        print(f"  normalised {n_cleaned:,} name strings "
              f"(dash variants, embedded identifiers, spacing)")
    if n_identifiers:
        print(f"  recovered {n_identifiers:,} ORCIDs embedded in name fields")

    existing_orcid = df["orcid"].fillna("").astype(str)
    df["orcid"] = [
        current if current else recovered
        for current, recovered in zip(existing_orcid, recovered_orcids)
    ]

    df["author_display"] = [
        format_display(g, s, r)
        for g, s, r in zip(df["given_name"], df["surname"], cleaned_raw)
    ]
    df["name_completeness"] = [
        classify(g, s, r)
        for g, s, r in zip(df["given_name"], df["surname"], cleaned_raw)
    ]
    df["name_flag"] = [
        flag_name(raw, display)
        for raw, display in zip(df["raw_name"], df["author_display"])
    ]

    # Split the rendered display back into parts, so the web can re-render it
    # any way it likes without re-parsing.
    surnames, forenames = [], []
    for display in df["author_display"]:
        if ", " in display:
            surname, _, forename = display.partition(", ")
        else:
            surname, forename = display, ""
        surnames.append(surname)
        forenames.append(forename)
    df["display_surname"] = surnames
    df["display_forenames"] = forenames

    columns = [
        "source", "outcome_id", "doi", "project_ids", "author_position",
        "raw_name", "author_display", "display_surname", "display_forenames",
        "name_completeness", "name_flag", "orcid", "native_id",
        "native_id_type", "outcome_key",
    ]
    df = df[columns]

    # The whole point of this script is that it does not merge anything.
    assert len(df) == row_count_in, "row count changed: something merged"

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"Wrote {len(df):,} rows to {OUTPUT_PATH}")
    print("Row count unchanged, so nothing was merged.")

    # Excel on macOS assumes Mac Roman unless a byte order mark says otherwise,
    # which turns "Benoît" into "Beno√Æt". The file above is correct UTF-8 and
    # is the one to use in code; this copy exists purely so Excel opens it
    # correctly on a double click.
    df.to_csv(EXCEL_PATH, index=False, encoding="utf-8-sig")
    print(f"Wrote an Excel-safe copy (BOM) to {EXCEL_PATH}")

    flagged = df[df["name_flag"] != ""]
    if len(flagged):
        print(f"\n  {len(flagged):,} rows carry a name_flag for review:")
        counts = {}
        for value in flagged["name_flag"]:
            for flag in value.split("; "):
                counts[flag] = counts.get(flag, 0) + 1
        for flag, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"    {flag:<22} {count:>5,}")

    print(f"\n{'=' * 70}\nNAME COMPLETENESS\n{'=' * 70}")
    for source, group in df.groupby("source"):
        counts = group["name_completeness"].value_counts()
        total = len(group)
        full = counts.get("full_forename", 0)
        print(f"\n  {source}  ({total:,} rows)")
        print(f"    real forename:  {full:>7,}  ({100 * full / total:5.1f}%)")
        for label in ("initials_only", "surname_only", "unparsed"):
            value = counts.get(label, 0)
            if value:
                print(f"    {label + ':':<16}{value:>7,}  "
                      f"({100 * value / total:5.1f}%)")

    total = len(df)
    full_total = int((df["name_completeness"] == "full_forename").sum())
    with_orcid = int((df["orcid"].fillna("").astype(str) != "").sum())
    print(f"\n{'=' * 70}")
    print(f"  rows:                    {total:,}")
    print(f"  with a real forename:    {full_total:,} "
          f"({100 * full_total / total:.1f}%)")
    print(f"  with an ORCID attached:  {with_orcid:,} "
          f"({100 * with_orcid / total:.1f}%)")
    print("\n  ORCID sits in its own column and collapses nothing. Group on it")
    print("  later if you want to; the displayed names do not depend on it.")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
