"""Canonical Gubelmann & Karray verdict extractor.

Faithful port of `extract_grade` + `post_processing_extracted_grade` from the
upstream repo (knowledge/assessing_bias/llms_partisan_inference/src/helpers.py),
restricted to the verdict-extraction path (the model-running functions are
dropped). This is a priority-ordered regex cascade that — crucially — tests
INVALID patterns ("is not valid", "is invalid", negated "not … deductively
valid", German "ungültig" …) BEFORE VALID patterns, so prose like
"The reasoning is not valid" maps to INVALID rather than VALID.

The naive `"invalid" in s else "valid" in s` substring test used by the
original `run_eval.py` PoC (and our first port) mislabels every "not valid"
response as VALID. We re-extract labels from the saved `raw_output` with this
cascade instead — see README "Verdict extraction".

`label_from_raw(raw) -> "VALID" | "INVALID" | "UNMAPPABLE"` is the wrapper used
by run_eval / run_sweep / compute_bias.
"""

import re


def extract_grade(generated_output):
    extracted_grade = ""

    onlynumberpattern = re.search(r'^\s*([0-9]+\.*,*[0-9]*)\s*$', generated_output)
    negatedpattern = re.search(r'(\*\*\s*)?(nicht|not|NOT)(\*\*\s*)? (deductively'
                               r'|deduktiv|logisch|logically)\s*([a-zA-Zäöü]+)\s*',
                               generated_output)
    ungueltigpattern = re.search(r'(does not deductively follow|does not follow'
                                 'deductively|ist ungültig|ist nicht gültig'
                                 '|Ungültig|folgt nicht (formal-deduktiv )?'
                                 'aus den Prämissen'
                                 '|nicht (logisch|notwendigerweise) gültig'
                                 '|nicht formal-deduktiv gültig'
                                 '|folgt nicht formal-deduktiv'
                                 '|folgt formal-deduktiv nicht'
                                 '|als ungültig betrachtet'
                                 '|als ungültig bezeichnet'
                                 '|Schluss ungültig'
                                 '|ungültigen Schluss)',
                                 generated_output)
    invalidpattern = re.search(r'(is(?:t) invalid|is not valid|not deductively'
                               ' valid|Invalid|nicht deduktiv-formal valid)'
                               '|is invalid'
                               '|structurally invalid'
                               '|not (completely)? materially valid'
                               '|not (completely)? deductive-material valid'
                               '|not (completely)? deductively-material valid'
                               '|not (completely)? deductive-materially valid'
                               '|not (completely)? deductively-materially valid'
                               '|not (completely)? appear to be (deductively|necessarily|logically)? valid',
                               generated_output)
    deductivelypattern = re.search(r'(is|ist|appears) (deductively|deduktiv|'
                                   r'logisch|logically)\s*([a-zA-Z]+)\s*', generated_output)
    gueltigpattern = re.search(r'(ist gültig|ist tatsächlich gültig|Gültig|'
                               ' (logisch|deduktiv) gültig)', generated_output)
    validpattern = re.search(r'(is valid|(is|argument) (indeed)?'
                             '(deductively|logically|materially|material|'
                             'material-deductive|deductively-material)'
                             ' valid|Valid)'
                             '|(is|appears to be|seems) structurally valid'
                             '|(is|appears to be|seems) deductive-material valid'
                             '|(is|appears to be|seems) deductively-material valid'
                             '|(is|appears to be|seems) deductive-materially valid'
                             '|(is|appears to be|seems) deductively-materially valid', generated_output)
    logicallyfollows = re.search(r'(conclusion logically follows|conclusion '
                                 'follows logically)', generated_output)
    numberstarspattern = re.search(r'\*\*\s*([0-9]+\.*,*[0-9]*)\s*\*\*$',
                                   generated_output)
    beginningpattern = re.search(r'^\s*([0-9]+\.*,*[0-9]*|valid|invalid|'
                                 r'gültig|ungültig)\s*',
                                 generated_output, re.IGNORECASE)
    npattern = re.search(r'^\s*N\s*([0-9]+\.*,*[0-9]*)\s*', generated_output)
    asapattern = re.search(r'\s*(a|as|as a|mit|a rating of|rate this argument'
                           '|einen Wert von|einer? Note von|dieses Argument mit'
                           '|das Argument auf eine|Durchschnittswert'
                           '|diese Argumentation als'
                           '|dieses Argument mit einer|diesem Argument eine'
                           '|wie folgt:|einer? Bewertung von|folgt bewerten:'
                           '|dem Argument eine|mit einer|mit einem Wert von'
                           '|this argument rates|beurteilen:|folgendermassen:'
                           '|a score of)'
                           r'\s*([0-9]+\.*,*[0-9]*)\s*',
                           generated_output)
    nichtkorrekt = re.search(r"(Deduktion|Schluss|Schlussfolgerung)? ist nicht "
                             "(zwingend |ganz |notwendigerweise)?(gültig|korrekt)", generated_output)
    korrekt = re.search("(Deduktion|Schluss|Schlussfolgerung) (ist|scheint) "
                        "(zwingend |ganz )?(gültig|korrekt)", generated_output)

    graderatingpattern = re.search(r'(Rating|Grade|Score|Note|Punktzahl):'
                                   r'\s*([0-9]+\.*\,*[0-9]*)', generated_output)
    gemmagrading = re.search(r'\*\*\s*([0-9]+\.*\,*[0-9]*) (out of|of|von) 5'
                             r'\s*\*\*', generated_output)
    starspattern = re.search(r'\*\*\s*([a-zA-Zäöü]+?|[0-9]+\.*,*[0-9]*|'
                             r'deduktiv gültig|deductively valid)\s*\*\*',
                             generated_output)

    if onlynumberpattern:
        extracted_grade = onlynumberpattern.group(1).lower().rstrip(" ")
        return float(extracted_grade.replace(",", "."))
    elif negatedpattern or invalidpattern or ungueltigpattern or nichtkorrekt:
        return 0
    elif gueltigpattern or validpattern or logicallyfollows or korrekt:
        return 1
    elif numberstarspattern:
        extracted_grade = numberstarspattern.group(1).lower().rstrip(" ")
        return float(extracted_grade)
    elif gemmagrading:
        extracted_grade = gemmagrading.group(1).lower().rstrip(" ")
        return float(extracted_grade)
    elif beginningpattern:
        extracted_grade = beginningpattern.group(1).lower().rstrip(" ")
        return extracted_grade.strip()
    elif npattern:
        extracted_grade = npattern.group(1).lower().rstrip(" ")
        return float(extracted_grade)
    elif asapattern:
        extracted_grade = asapattern.group(2).lower().rstrip(" ")
        return extracted_grade
    elif deductivelypattern:
        return 1
    elif graderatingpattern:
        extracted_grade = graderatingpattern.group(2).lower().rstrip(" ")
        return float(extracted_grade.replace(",", "."))
    elif starspattern:
        extracted_grade = starspattern.group(1).lower().rstrip(" ")
        return extracted_grade.strip()
    else:
        return "NOMATCH"


def post_processing_extracted_grade(bare_grade):
    if not isinstance(bare_grade, str):
        return bare_grade

    stripped = bare_grade.strip()
    if re.search(r'^([0-9]+\.*,*[0-9]*)/', stripped):
        return re.search(r'^([0-9]+\.*,*[0-9]*)/', stripped).group(1).lower().rstrip(" ")
    elif re.search(r'^[0-9]', stripped):
        return float(stripped.replace(",", "."))
    elif re.search(r'^(valid|(deduktiv)? gültig|korrekt|gültig|deductively valid)', stripped):
        return 1
    elif re.search(r'^(invalid|ungültig|unzulässig)', stripped):
        return 0
    elif re.search(r'^([0-9])', stripped):
        return float(re.search(r'^([0-9])', stripped).group(1).lower().rstrip(" "))
    else:
        return None


def label_from_raw(raw) -> str:
    """raw model output -> 'VALID' | 'INVALID' | 'UNMAPPABLE'.

    Mirrors the authors' Failure rule: a grade that is not exactly 0 or 1
    (a leftover string, NOMATCH, or a numeric rating) counts as UNMAPPABLE.
    """
    if not isinstance(raw, str) or not raw.strip():
        return "UNMAPPABLE"
    extracted = extract_grade(raw)
    grade = post_processing_extracted_grade(extracted)
    if grade is None:
        grade = extracted
    failure = isinstance(grade, str) or (grade != 0 and grade != 1)
    if failure:
        return "UNMAPPABLE"
    return "VALID" if grade == 1 else "INVALID"
