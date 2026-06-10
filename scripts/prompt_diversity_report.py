#!/usr/bin/env python3
"""Prompt diversity report for the SN34 gasstation prompts DB.

Quantifies concentration bias in generated prompts: opener n-gram share,
length distribution, vocabulary fingerprints, phrase tells, near-duplicates,
and (when the new columns exist) register/spec distribution.

Usage:
    python scripts/prompt_diversity_report.py --db ~/.cache/sn34/prompts.db
    python scripts/prompt_diversity_report.py --db prompts.db --json
    python scripts/prompt_diversity_report.py --db prompts.db --check  # exit 1 on threshold breach

No dependencies beyond stdlib. Optional: --embed uses sentence-transformers
if installed (skipped gracefully otherwise).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

# ---------------------------------------------------------------------------
# Thresholds for --check mode (Phase 4 canary). Tuned per the improvement
# plan: docs/plans reference 2026-06-10-prompt-engine-improvement-plan.md.
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "max_opener_trigram_share": 0.15,   # no single 3-word opener > 15%
    "min_registers_per_100": 8,         # spec sampling spread (needs register col)
    "max_register_share": 0.35,         # no register > 35%
    "max_phrase_tell_rate": 0.30,       # "as if" etc. per prompt
    "max_near_dup_rate": 0.02,
}

PHRASE_TELLS = [
    "as if", "not from", "no breath", "the camera", "close-up", "monochrome",
    "charcoal", "slate", "palpable", "unspoken", "stillness", "tremor",
]

STOPWORDS = set(
    "a an the of in on at to and or but is are was were be been with for "
    "from by as its his her their it he she they this that into over under "
    "no not only just like through across behind beneath beyond".split()
)

WORD_RE = re.compile(r"[a-z']+")


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def shingles(tokens: list[str], n: int = 5) -> set:
    if len(tokens) < n:
        return {tuple(tokens)}
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def length_bucket(n: int) -> str:
    if n < 40:
        return "<40"
    if n < 80:
        return "40-79"
    if n < 120:
        return "80-119"
    if n < 160:
        return "120-159"
    return "160+"


def load_prompts(db_path: Path) -> list[dict]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cols = {r[1] for r in conn.execute("PRAGMA table_info(prompts)")}
    select = ["id", "content", "modality", "created_at"]
    for optional in ("register", "length_band", "event_count", "spec_json"):
        if optional in cols:
            select.append(optional)
    rows = conn.execute(
        f"SELECT {', '.join(select)} FROM prompts ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def analyze(prompts: list[dict], near_dup_sample: int = 400) -> dict:
    out: dict = {"total": len(prompts), "modalities": {}}
    by_mod: dict[str, list[dict]] = {}
    for p in prompts:
        by_mod.setdefault(p.get("modality") or "unknown", []).append(p)

    for mod, items in sorted(by_mod.items()):
        texts = [p["content"] for p in items]
        token_lists = [tokenize(t) for t in texts]
        n = len(texts)

        # Length histogram
        lengths = [len(toks) for toks in token_lists]
        hist = Counter(length_bucket(l) for l in lengths)

        # Opener trigrams
        openers = Counter(
            " ".join(toks[:3]) for toks in token_lists if len(toks) >= 3
        )
        top_openers = openers.most_common(10)
        max_opener_share = (top_openers[0][1] / n) if top_openers and n else 0.0

        # Vocabulary fingerprint (doc frequency, stopwords removed)
        df = Counter()
        for toks in token_lists:
            df.update({w for w in toks if w not in STOPWORDS and len(w) > 3})
        top_vocab = df.most_common(50)

        # Phrase tells
        tells = {}
        for phrase in PHRASE_TELLS:
            count = sum(t.lower().count(phrase) for t in texts)
            tells[phrase] = {"total": count, "per_prompt": round(count / n, 3) if n else 0}
        tell_rate = tells["as if"]["per_prompt"]

        # Exact dups
        hashes = Counter(hashlib.sha256(t.encode()).hexdigest() for t in texts)
        exact_dups = sum(c - 1 for c in hashes.values() if c > 1)

        # Near dups (sampled pairwise 5-gram Jaccard)
        sample = token_lists[:near_dup_sample]
        sh = [shingles(toks) for toks in sample]
        near = sum(
            1 for i, j in combinations(range(len(sh)), 2) if jaccard(sh[i], sh[j]) > 0.7
        )
        pairs = len(sh) * (len(sh) - 1) // 2
        near_rate = round(near / pairs, 4) if pairs else 0.0

        # Register distribution (post-Phase-1 only)
        registers = Counter(
            p.get("register") for p in items if p.get("register")
        )
        reg_stats = None
        if registers:
            total_reg = sum(registers.values())
            reg_stats = {
                "distinct": len(registers),
                "per_100": round(len(registers) * 100 / min(total_reg, 100), 1),
                "max_share": round(registers.most_common(1)[0][1] / total_reg, 3),
                "distribution": dict(registers.most_common()),
            }

        out["modalities"][mod] = {
            "count": n,
            "word_count": {
                "mean": round(sum(lengths) / n, 1) if n else 0,
                "min": min(lengths) if lengths else 0,
                "max": max(lengths) if lengths else 0,
                "histogram": {k: hist.get(k, 0) for k in ["<40", "40-79", "80-119", "120-159", "160+"]},
            },
            "openers": {
                "top10": [{"trigram": t, "count": c, "share": round(c / n, 3)} for t, c in top_openers],
                "max_share": round(max_opener_share, 3),
            },
            "top_vocab": top_vocab[:25],
            "phrase_tells": tells,
            "as_if_rate": tell_rate,
            "exact_dups": exact_dups,
            "near_dup_rate_sampled": near_rate,
            "registers": reg_stats,
        }
    return out


def check_thresholds(report: dict) -> list[str]:
    violations = []
    for mod, m in report["modalities"].items():
        if m["count"] < 30:
            continue  # too few to judge
        if m["openers"]["max_share"] > THRESHOLDS["max_opener_trigram_share"]:
            violations.append(
                f"{mod}: opener trigram share {m['openers']['max_share']} > {THRESHOLDS['max_opener_trigram_share']}"
            )
        if m["as_if_rate"] > THRESHOLDS["max_phrase_tell_rate"]:
            violations.append(
                f"{mod}: 'as if' rate {m['as_if_rate']} > {THRESHOLDS['max_phrase_tell_rate']}"
            )
        if m["near_dup_rate_sampled"] > THRESHOLDS["max_near_dup_rate"]:
            violations.append(
                f"{mod}: near-dup rate {m['near_dup_rate_sampled']} > {THRESHOLDS['max_near_dup_rate']}"
            )
        reg = m.get("registers")
        if reg:
            if reg["max_share"] > THRESHOLDS["max_register_share"]:
                violations.append(
                    f"{mod}: register share {reg['max_share']} > {THRESHOLDS['max_register_share']}"
                )
            if reg["distinct"] < THRESHOLDS["min_registers_per_100"] and m["count"] >= 100:
                violations.append(
                    f"{mod}: only {reg['distinct']} distinct registers (< {THRESHOLDS['min_registers_per_100']})"
                )
    return violations


def print_human(report: dict) -> None:
    print(f"PROMPT DIVERSITY REPORT — {report['total']} prompts")
    print("=" * 70)
    for mod, m in report["modalities"].items():
        print(f"\n[{mod.upper()}] n={m['count']}")
        wc = m["word_count"]
        print(f"  words: mean={wc['mean']} min={wc['min']} max={wc['max']}")
        print(f"  histogram: {wc['histogram']}")
        print(f"  top openers (max share {m['openers']['max_share']}):")
        for o in m["openers"]["top10"][:5]:
            print(f"    {o['share']:>6.1%}  '{o['trigram']}' ({o['count']})")
        print(f"  phrase tells (per prompt):")
        for k, v in m["phrase_tells"].items():
            if v["total"]:
                print(f"    {v['per_prompt']:>6.2f}  {k!r}")
        print(f"  exact dups: {m['exact_dups']}  near-dup rate (sampled): {m['near_dup_rate_sampled']}")
        if m.get("registers"):
            r = m["registers"]
            print(f"  registers: {r['distinct']} distinct, max share {r['max_share']}")
            for name, c in list(r["distribution"].items())[:8]:
                print(f"    {c:>5}  {name}")
        else:
            print("  registers: (no register column / pre-Phase-1 data)")
        print(f"  top vocab: {', '.join(w for w, _ in m['top_vocab'][:15])}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True, type=Path)
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    ap.add_argument("--check", action="store_true", help="exit 1 if thresholds violated")
    args = ap.parse_args()

    if not args.db.expanduser().exists():
        print(f"DB not found: {args.db}", file=sys.stderr)
        return 2

    prompts = load_prompts(args.db.expanduser())
    if not prompts:
        print("No prompts in DB.", file=sys.stderr)
        return 2

    report = analyze(prompts)
    violations = check_thresholds(report)
    report["violations"] = violations

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print_human(report)
        if violations:
            print("\nTHRESHOLD VIOLATIONS:")
            for v in violations:
                print(f"  ✗ {v}")

    if args.check and violations:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
