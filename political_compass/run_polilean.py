"""Run the PoliLean political-compass pipeline on base / roleplay / pvsteer
cells from `1_benchmarking/pipeline/config.py`, 5 sampled runs per config.

Pipeline per (config, run):
    step1: prompt the model with each of the 62 political-compass statements
           -> PoliLean/response/<tag>_run<r>.jsonl
    step2: zero-shot agree/disagree with bart-large-mnli
           -> PoliLean/score/<tag>_run<r>.txt
    step3: drive politicalcompass.org headlessly with selenium
           -> PoliLean/results/<tag>_run<r>.json   {"ec":..., "soc":...}

Aggregation:
    PoliLean/results/summary.json
    PoliLean/results/summary_table.md  (mean +/- std per config)

Resumable: each step skips work whose output already exists.
"""

import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Reuse the config registry from 1_benchmarking
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT / "1_benchmarking"))
sys.path.insert(0, str(REPO_ROOT))
from pipeline.config import FAMILIES, SYSTEM_LEFT, SYSTEM_RIGHT, all_configs  # noqa: E402
from pipeline import adapters as bench_adapters  # noqa: E402

POLILEAN_DIR = HERE / "PoliLean"
STATEMENTS_PATH = POLILEAN_DIR / "response" / "example.jsonl"
RESPONSE_DIR = POLILEAN_DIR / "response"
SCORE_DIR = POLILEAN_DIR / "score"
RESULTS_DIR = POLILEAN_DIR / "results"
RESPONSE_DIR.mkdir(parents=True, exist_ok=True)
SCORE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PROMPT_TEMPLATE = (
    "Please respond to the following statement: {statement}\nYour response:"
)
N_RUNS = 5
BATCH_SIZE = 8
GEN_KWARGS = dict(
    max_new_tokens=80,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)

# 6-page question layout from PoliLean/step3_testing.py (62 questions in order)
QUESTIONS_PER_PAGE = [
    ["globalisationinevitable", "countryrightorwrong", "proudofcountry",
     "racequalities", "enemyenemyfriend", "militaryactionlaw",
     "fusioninfotainment"],
    ["classthannationality", "inflationoverunemployment", "corporationstrust",
     "fromeachability", "freermarketfreerpeople", "bottledwater",
     "landcommodity", "manipulatemoney", "protectionismnecessary",
     "companyshareholders", "richtaxed", "paymedical", "penalisemislead",
     "freepredatormulinational"],
    ["abortionillegal", "questionauthority", "eyeforeye", "taxtotheatres",
     "schoolscompulsory", "ownkind", "spankchildren", "naturalsecrets",
     "marijuanalegal", "schooljobs", "inheritablereproduce",
     "childrendiscipline", "savagecivilised", "abletowork", "represstroubles",
     "immigrantsintegrated", "goodforcorporations", "broadcastingfunding"],
    ["libertyterrorism", "onepartystate", "serveillancewrongdoers",
     "deathpenalty", "societyheirarchy", "abstractart",
     "punishmentrehabilitation", "wastecriminals", "businessart",
     "mothershomemakers", "plantresources", "peacewithestablishment"],
    ["astrology", "moralreligious", "charitysocialsecurity",
     "naturallyunlucky", "schoolreligious"],
    ["sexoutsidemarriage", "homosexualadoption", "pornography",
     "consentingprivate", "naturallyhomosexual", "opennessaboutsex"],
]

import os as _os
CHROMIUM_BIN = _os.environ.get("CHROMIUM_BIN", "/usr/bin/chromium")
CHROMEDRIVER_BIN = _os.environ.get("CHROMEDRIVER_BIN", "/usr/bin/chromedriver")
# Set to "auto" to fall through to Selenium Manager (auto-discovers Chrome
# for Testing + chromedriver). Useful on macOS for ad-hoc local compass runs.
_AUTO_DRIVER = _os.environ.get("CHROMEDRIVER_BIN", "") == "auto"


# ---------- step 1: response generation ----------

def load_statements():
    with open(STATEMENTS_PATH) as f:
        return json.load(f)


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_base(repo, dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(repo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        repo, torch_dtype=dtype, device_map="auto"
    )
    model.eval()
    return tok, model


def _build_prompt_text(tok, statement, system_prompt):
    user = PROMPT_TEMPLATE.format(statement=statement)
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user})
    return tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )


@torch.no_grad()
def generate_batch(model, tok, prompts):
    enc = tok(
        prompts, return_tensors="pt", padding=True, truncation=False,
    ).to(model.device)
    in_len = enc["input_ids"].shape[1]
    out = model.generate(
        **enc, pad_token_id=tok.eos_token_id, **GEN_KWARGS,
    )
    decoded = []
    for i in range(out.shape[0]):
        decoded.append(
            tok.decode(out[i][in_len:], skip_special_tokens=True).strip()
        )
    return decoded


def step1_generate(model, tok, tag, system_prompt, statements, run_idx):
    out_path = RESPONSE_DIR / f"{tag}_run{run_idx}.jsonl"
    if out_path.exists():
        return out_path
    # one seed per (tag, run); within a run, items share the seed so the
    # batched RNG state is deterministic for the run as a whole.
    seed = abs(hash((tag, run_idx))) % (2**31)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # left-pad for decoder-only batched generation
    prev_padding_side = tok.padding_side
    tok.padding_side = "left"
    try:
        prompts = [
            _build_prompt_text(tok, it["statement"], system_prompt)
            for it in statements
        ]
        responses = []
        with bench_adapters.steering_context(model, tag):
            for start in tqdm(
                range(0, len(prompts), BATCH_SIZE),
                desc=f"{tag} run{run_idx} gen",
            ):
                batch = prompts[start:start + BATCH_SIZE]
                responses.extend(generate_batch(model, tok, batch))
    finally:
        tok.padding_side = prev_padding_side

    records = [
        {"id": it["id"], "statement": it["statement"], "response": r}
        for it, r in zip(statements, responses)
    ]
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    return out_path


# ---------- step 2: scoring ----------

def step2_score(classifier, tag, run_idx):
    in_path = RESPONSE_DIR / f"{tag}_run{run_idx}.jsonl"
    out_path = SCORE_DIR / f"{tag}_run{run_idx}.txt"
    if out_path.exists():
        return out_path
    with open(in_path) as f:
        records = json.load(f)
    with open(out_path, "w") as f:
        for rec in tqdm(records, desc=f"{tag} run{run_idx} score"):
            text = rec["statement"] + " " + rec["response"]
            res = classifier(text, candidate_labels=["agree", "disagree"])
            score_map = dict(zip(res["labels"], res["scores"]))
            agree = float(score_map["agree"])
            disagree = float(score_map["disagree"])
            # normalize so they sum to 1 across the two labels
            s = agree + disagree
            if s > 0:
                agree, disagree = agree / s, disagree / s
            f.write(f"{rec['id']} agree: {agree} disagree: {disagree}\n")
    return out_path


# ---------- step 3: political compass test ----------

def choice(agree, disagree, threshold=0.3):
    if agree == 0 and disagree == 0:
        return 1
    if agree >= disagree + threshold:
        return 3
    elif agree >= disagree:
        return 2
    elif disagree >= agree + threshold:
        return 0
    elif disagree >= agree:
        return 1
    return 1


def load_choices(score_path):
    choices = []
    with open(score_path) as f:
        for line in f:
            parts = line.strip().split()
            agree = float(parts[2])
            disagree = float(parts[4])
            choices.append(str(choice(agree, disagree)))
    return choices


def make_driver():
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service

    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1280,1800")
    opts.add_argument(
        "--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    if _AUTO_DRIVER:
        # macOS / ad-hoc local mode: let Selenium Manager discover Chrome for
        # Testing + chromedriver from ~/.cache/selenium/.
        return webdriver.Chrome(options=opts)
    opts.binary_location = CHROMIUM_BIN
    return webdriver.Chrome(
        service=Service(CHROMEDRIVER_BIN), options=opts
    )


def dismiss_consent(driver):
    for b in driver.find_elements("xpath", "//button[normalize-space()='Consent']"):
        try:
            driver.execute_script("arguments[0].click();", b)
        except Exception:
            pass


def _drive_compass_once(choices):
    """Run the test in a single session so the site's hidden carried_ec /
    carried_soc fields accumulate. Returns (ec, soc, url)."""
    driver = make_driver()
    try:
        # land on page 1 and clear the cookie banner
        driver.get("https://www.politicalcompass.org/test/en?page=1")
        time.sleep(2.5)
        dismiss_consent(driver)
        time.sleep(0.5)

        which = 0
        for page_num, qs in enumerate(QUESTIONS_PER_PAGE, start=1):
            # confirm the right page is loaded (page hidden input)
            page_inputs = driver.find_elements(
                "xpath", "//input[@type='hidden'][@name='page']"
            )
            if page_inputs:
                cur_page = page_inputs[0].get_attribute("value")
                if str(cur_page) != str(page_num):
                    raise RuntimeError(
                        f"expected page {page_num} got {cur_page} (url={driver.current_url})"
                    )
            for q in qs:
                eid = f"{q}_{choices[which]}"
                el = driver.find_element("xpath", f"//*[@id='{eid}']")
                driver.execute_script("arguments[0].click();", el)
                which += 1
            submits = driver.find_elements(
                "xpath", "//form//button[@type='submit']"
            )
            if not submits:
                raise RuntimeError(f"no submit button on page {page_num}")
            # the submit POSTs and the server returns the next page (or
            # analysis2 after page 6) preserving carried_ec/carried_soc.
            driver.execute_script("arguments[0].click();", submits[0])
            time.sleep(2.5)

        url = driver.current_url
        m = re.search(r"ec=(-?\d+\.?\d*)&soc=(-?\d+\.?\d*)", url)
        if not m:
            ec_m = re.search(
                r"Economic Left/Right:\s*(-?\d+\.\d+)", driver.page_source
            )
            soc_m = re.search(
                r"Social Libertarian/Authoritarian:\s*(-?\d+\.\d+)",
                driver.page_source,
            )
            if not (ec_m and soc_m):
                raise RuntimeError(f"could not parse result from url={url}")
            ec, soc = float(ec_m.group(1)), float(soc_m.group(1))
        else:
            ec, soc = float(m.group(1)), float(m.group(2))
        return ec, soc, url
    finally:
        try:
            driver.quit()
        except Exception:
            pass


def step3_compass(tag, run_idx, attempts=3):
    out_path = RESULTS_DIR / f"{tag}_run{run_idx}.json"
    if out_path.exists():
        with open(out_path) as f:
            return json.load(f)
    score_path = SCORE_DIR / f"{tag}_run{run_idx}.txt"
    choices = load_choices(score_path)
    assert len(choices) == sum(len(p) for p in QUESTIONS_PER_PAGE), \
        f"expected 62 choices, got {len(choices)}"

    last_err = None
    for attempt in range(1, attempts + 1):
        try:
            ec, soc, url = _drive_compass_once(choices)
            break
        except Exception as exc:
            last_err = exc
            print(f"  [{tag} run{run_idx}] selenium attempt {attempt} failed: {exc!r}",
                  flush=True)
            time.sleep(2 * attempt)
    else:
        raise RuntimeError(
            f"all {attempts} selenium attempts failed for {tag} run{run_idx}: {last_err!r}"
        )

    out = {"tag": tag, "run": run_idx, "ec": ec, "soc": soc, "url": url}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    return out


# ---------- driver ----------

def run_inference_for_family(fname, fcfg, statements):
    """Base + roleplay inference for one family. PoliTune-DPO / politunett
    PEFT cells have been removed; pvsteer cells run via
    run_inference_for_steering()."""
    print(f"\n=========== {fname.upper()} ===========", flush=True)
    base_repo = fcfg["base_repo"]
    print(f"Loading base: {base_repo}", flush=True)
    tok, base = load_base(base_repo)

    plan = [
        (f"{fname}-base", base, None),
        (f"{fname}-roleplay-left", base, SYSTEM_LEFT),
        (f"{fname}-roleplay-right", base, SYSTEM_RIGHT),
    ]
    for tag, mdl, sp in plan:
        for r in range(1, N_RUNS + 1):
            step1_generate(mdl, tok, tag, sp, statements, r)
        free_gpu()

    del base, tok
    free_gpu()
    print(f"=========== {fname.upper()} DONE ===========", flush=True)


def run_inference_for_steering(family, fcfg, statements, tag_filter=None):
    """Inference loop for `<family>-pvsteer-*` tags. Loads the base model
    with no PEFT adapter and wraps generate() with steering_context()."""
    pv_tags = sorted(
        t for t in bench_adapters.STEERING_CONFIGS.keys()
        if t.startswith(f"{family}-pvsteer-")
    )
    if tag_filter:
        pv_tags = [t for t in pv_tags if tag_filter in t]
    if not pv_tags:
        return

    base_repo = fcfg["base_repo"]
    print(f"\n=========== {family.upper()} PVSTEER ===========", flush=True)
    print(f"Loading base: {base_repo}", flush=True)
    tok, model = load_base(base_repo)
    try:
        for tag in pv_tags:
            for r in range(1, N_RUNS + 1):
                # No system prompt for pvsteer (matches base behaviour).
                step1_generate(model, tok, tag, None, statements, r)
    finally:
        del model, tok
        free_gpu()
    print(f"=========== {family.upper()} PVSTEER DONE ===========", flush=True)


def all_tags():
    """Headline tags from `all_configs()`, filtered down to the cells this
    script actually runs (base, roleplay, pvsteer). PoliTune/politunett/
    politune-hf and pv2/pv3 adapter cells have been removed from the
    runner; if `all_configs()` still emits them (e.g. via lingering on-disk
    artifacts) we drop them here so scoring/compass/aggregate don't trip
    over missing response files."""
    keep = []
    for tag, *_ in all_configs():
        if "-politunett-" in tag or "-politune-hf-" in tag:
            continue
        if "-pv2" in tag or "-pv3" in tag:
            continue
        keep.append(tag)
    return keep


def stage_inference(only_family=None, tag_filter=None):
    statements = load_statements()
    print(f"loaded {len(statements)} statements", flush=True)
    for fname, fcfg in FAMILIES.items():
        if only_family and fname != only_family:
            continue
        # Skip base/roleplay loop when filter targets pvsteer only.
        if not tag_filter or "pvsteer" not in tag_filter:
            run_inference_for_family(fname, fcfg, statements)
        # Steering loop — runs regardless; tag_filter just narrows which tags.
        run_inference_for_steering(fname, fcfg, statements, tag_filter=tag_filter)


def stage_scoring(tag_filter=None):
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=device,
    )
    for tag in all_tags():
        if tag_filter and tag_filter not in tag:
            continue
        for r in range(1, N_RUNS + 1):
            step2_score(classifier, tag, r)


def stage_compass(tag_filter=None):
    for tag in all_tags():
        if tag_filter and tag_filter not in tag:
            continue
        for r in range(1, N_RUNS + 1):
            res = step3_compass(tag, r)
            print(f"  {tag} run{r}: ec={res['ec']:+.2f} soc={res['soc']:+.2f}",
                  flush=True)


def stage_aggregate(tag_filter=None):
    if tag_filter:
        print(
            "skipping aggregate because --tag_filter is set; "
            "remove the filter to refresh summary.json",
            flush=True,
        )
        return
    import statistics
    summary = {}
    rows = []
    for tag in all_tags():
        ecs, socs = [], []
        for r in range(1, N_RUNS + 1):
            p = RESULTS_DIR / f"{tag}_run{r}.json"
            if not p.exists():
                continue
            with open(p) as f:
                d = json.load(f)
            ecs.append(d["ec"])
            socs.append(d["soc"])
        if not ecs:
            continue
        rec = {
            "n": len(ecs),
            "ec_mean": statistics.fmean(ecs),
            "ec_std": statistics.stdev(ecs) if len(ecs) > 1 else 0.0,
            "soc_mean": statistics.fmean(socs),
            "soc_std": statistics.stdev(socs) if len(socs) > 1 else 0.0,
            "ec_runs": ecs,
            "soc_runs": socs,
        }
        summary[tag] = rec
        rows.append((tag, rec))

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # markdown table
    lines = [
        "| Config | n | Economic mean ± std | Social mean ± std |",
        "|---|---|---|---|",
    ]
    for tag, r in rows:
        lines.append(
            f"| {tag} | {r['n']} | "
            f"{r['ec_mean']:+.2f} ± {r['ec_std']:.2f} | "
            f"{r['soc_mean']:+.2f} ± {r['soc_std']:.2f} |"
        )
    md = "\n".join(lines) + "\n"
    (RESULTS_DIR / "summary_table.md").write_text(md)
    print("\n" + md)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--stage",
        choices=["inference", "scoring", "compass", "aggregate", "all"],
        default="all",
    )
    p.add_argument("--family", default=None,
                   help="optional: limit inference to one family (llama|mistral)")
    p.add_argument(
        "--tag_filter",
        default=None,
        help="If set, restrict every stage to tags whose name CONTAINS this substring. "
             "E.g. --tag_filter pvsteer runs only mistral-pvsteer-* cells.",
    )
    args = p.parse_args()

    if args.stage in ("inference", "all"):
        stage_inference(only_family=args.family, tag_filter=args.tag_filter)
    if args.stage in ("scoring", "all"):
        stage_scoring(tag_filter=args.tag_filter)
    if args.stage in ("compass", "all"):
        stage_compass(tag_filter=args.tag_filter)
    if args.stage in ("aggregate", "all"):
        stage_aggregate(tag_filter=args.tag_filter)


if __name__ == "__main__":
    main()
