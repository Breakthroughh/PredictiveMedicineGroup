#!/usr/bin/env python3
# rule_activations_extraction.py
#
# Purpose:
#   Extract "rule activation" events from Werewolf game logs to verify
#   that parsing works before any training/optimization.
#
# Output:
#   - Pretty-printed schema and sample rows
#   - Summary stats (counts by rule, by archetype, wolf vs non-wolf)
#   - Optional CSV / NDJSON dumps for downstream work
#
# Usage examples:
#   python rule_activations_extraction.py --glob "logs/*.json" --show 20
#   python rule_activations_extraction.py --glob "logs/*.json" --to-csv out/rule_hits.csv
#   python rule_activations_extraction.py --glob "logs/*.json" --to-ndjson out/rule_hits.ndjson --show 5
#   python rule_activations_extraction.py --glob "logs/*.json" --k-rules 8 --strict
#
# Notes:
#   - Multi-wolf games are supported: we mark is_target_wolf = (target in wolves).
#   - Rule IDs are extracted from rubric rationale lines: "because of rule <id>" (case-insensitive).
#   - Targets are parsed from lines like "Suspicion score Agent3:" (case-insensitive).
#   - If your line format changes, update the regexes below.

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

# -------- Regex patterns (tweak here if your prompt format changes) --------
RULE_RE = re.compile(r"because of rule\s+(\d+)", re.I)
TGT_RE  = re.compile(r"Suspicion score\s+(Agent\d+)\s*:", re.I)

# -------- Data schema: each extracted hit is a dict with these keys --------
SCHEMA = [
    ("game",            "str",   "Path or ID of game log"),
    ("day",             "int",   "Day index from log event"),
    ("round",           "int",   "Round index from log event"),
    ("stage",           "str",   "Stage label from log event (can be numeric or 'pre', 'post', etc.)"),
    ("rater",           "str",   "Agent who produced the rubric/rationale"),
    ("target",          "str",   "Agent the suspicion line refers to"),
    ("rule_id",         "int",   "Rule number parsed from rationale"),
    ("archetype",       "str",   "Werewolf archetype from metadata"),
    ("is_target_wolf",  "bool",  "Whether target is one of the wolves in that game"),
]


# -------- IO helpers --------

def load_games(pattern: str) -> List[Tuple[str, dict]]:
    """Load all JSON game logs that match a glob pattern."""
    paths = sorted(glob.glob(pattern))
    games: List[Tuple[str, dict]] = []
    for p in paths:
        try:
            with open(p, "r") as f:
                games.append((p, json.load(f)))
        except Exception as e:
            print(f"[warn] Failed to load {p}: {e}", file=sys.stderr)
    if not games:
        print(f"[warn] No files matched pattern: {pattern}", file=sys.stderr)
    return games

# -------- Core parsing --------

def wolves_from_roles(meta_roles: Dict[str, str]) -> set:
    """Return set of wolf agents from metadata.roles."""
    return {a for a, r in meta_roles.items() if str(r).lower() == "werewolf"}

def extract_rule_hits(game_json: dict, game_id: Optional[str] = None) -> List[dict]:
    """
    Extract rule activations (hits) from a single game JSON.
    Returns a list of dicts following SCHEMA.
    """
    roles: Dict[str, str] = game_json["metadata"]["roles"]
    wolves = wolves_from_roles(roles)
    arche = game_json["metadata"].get("werewolf_archetype", "unknown")

    out: List[dict] = []
    for ev in game_json.get("events", []):
        if ev.get("event_type") != "inter_agent_ratings":
            continue
        if not ev.get("rubric_applied"):
            continue

        day   = ev.get("day")
        rd    = ev.get("round")
        stage = ev.get("stage")
        rater = ev.get("rater")

        for line in ev.get("rubric_rationale_lines", []):
            m_rule = RULE_RE.search(line or "")
            m_tgt  = TGT_RE.search(line or "")
            if not (m_rule and m_tgt):
                continue
            try:
                rule_id = int(m_rule.group(1))
            except Exception:
                continue
            target = m_tgt.group(1)
            out.append({
    "game": game_id,
    "day": int(day) if isinstance(day, (int, float)) else -1,
    "round": int(rd) if isinstance(rd, (int, float)) else -1,
    "stage": str(stage),   # <-- keep as string, don’t cast to int
    "rater": str(rater),
    "target": str(target),
    "rule_id": rule_id,
    "archetype": str(arche),
    "is_target_wolf": (target in wolves),
})

    return out

# -------- Validation & summaries --------

def validate_hits(hits: List[dict], k_rules: Optional[int], strict: bool) -> None:
    """
    Basic validations:
      - rule_id must be positive (and <= k_rules if provided)
      - rater/target look like Agent\d+
    If strict=True, raise on first error; otherwise print warnings.
    """
    def bad(msg: str):
        if strict:
            raise ValueError(msg)
        else:
            print(f"[warn] {msg}", file=sys.stderr)

    for i, h in enumerate(hits[:10000]):  # limit to avoid huge spam
        rid = h.get("rule_id")
        if not isinstance(rid, int) or rid <= 0:
            bad(f"Invalid rule_id at idx {i}: {rid}")
        if k_rules is not None and (rid < 1 or rid > k_rules):
            bad(f"rule_id {rid} outside 1..{k_rules} at idx {i}")
        for k in ("rater", "target"):
            v = h.get(k, "")
            if not re.fullmatch(r"Agent\d+", str(v) if v is not None else ""):
                bad(f"{k} '{v}' does not match 'Agent\\d+' at idx {i}")

def summarize_hits(hits: List[dict]) -> None:
    """Print quick summaries for eyeballing."""
    if not hits:
        print("[info] No rule hits extracted.")
        return

    print("\n=== Summary: overall ===")
    print(f"Total hits: {len(hits)}")
    games = {h["game"] for h in hits}
    print(f"Games covered: {len(games)}")

    by_rule = Counter(h["rule_id"] for h in hits)
    by_arch = Counter(h["archetype"] for h in hits)
    wolf_hits = sum(1 for h in hits if h["is_target_wolf"])
    nonwolf_hits = len(hits) - wolf_hits

    print("\nHits by rule_id (top 10):")
    for rid, c in by_rule.most_common(10):
        print(f"  rule {rid}: {c}")

    print("\nHits by archetype (top 10):")
    for a, c in by_arch.most_common(10):
        print(f"  {a}: {c}")

    print("\nTarget class breakdown:")
    print(f"  → on wolves    : {wolf_hits}")
    print(f"  → on non-wolves: {nonwolf_hits}")
    if len(hits) > 0:
        print(f"  → share on wolves: {wolf_hits / len(hits):.3f}")

    # Optional: rule × target-is-wolf matrix (top few rules)
    print("\nRule × target-is-wolf (top 8 rules):")
    top_rules = [rid for rid, _ in by_rule.most_common(8)]
    table: Dict[int, Dict[str, int]] = defaultdict(lambda: {"wolf": 0, "nonwolf": 0})
    for h in hits:
        key = "wolf" if h["is_target_wolf"] else "nonwolf"
        table[h["rule_id"]][key] += 1
    for rid in top_rules:
        row = table[rid]
        print(f"  rule {rid:>2}: wolf={row['wolf']:<5} nonwolf={row['nonwolf']:<5}")

def print_schema() -> None:
    print("=== Extracted row schema ===")
    for name, typ, desc in SCHEMA:
        print(f"- {name:15s} : {typ:5s} — {desc}")

def print_samples(hits: List[dict], n: int) -> None:
    print(f"\n=== First {min(n, len(hits))} rows ===")
    cols = [name for (name, _, _) in SCHEMA]
    # compute column widths
    widths = {c: max(len(c), 10) for c in cols}
    for row in hits[:n]:
        for c in cols:
            widths[c] = max(widths[c], len(str(row.get(c, ""))))
    header = " | ".join(f"{c:<{widths[c]}}" for c in cols)
    print(header)
    print("-" * len(header))
    for row in hits[:n]:
        line = " | ".join(f"{str(row.get(c,'')):<{widths[c]}}" for c in cols)
        print(line)

# -------- Writers --------

def write_csv(hits: List[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = [name for (name, _, _) in SCHEMA]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for h in hits:
            w.writerow({c: h.get(c) for c in cols})
    print(f"[ok] Wrote CSV: {path}")

def write_ndjson(hits: List[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for h in hits:
            f.write(json.dumps(h, ensure_ascii=False) + "\n")
    print(f"[ok] Wrote NDJSON: {path}")

# -------- CLI --------

def main():
    ap = argparse.ArgumentParser(description="Extract and inspect rule activations from Werewolf logs.")
    ap.add_argument("--glob", required=True, help="Glob for game logs, e.g. 'logs/*.json'")
    ap.add_argument("--k-rules", type=int, default=None, help="If set, validate that rule_id ∈ [1..k_rules].")
    ap.add_argument("--show", type=int, default=10, help="Number of sample rows to print.")
    ap.add_argument("--to-csv", type=str, default=None, help="Optional path to write CSV (e.g., out/rule_hits.csv).")
    ap.add_argument("--to-ndjson", type=str, default=None, help="Optional path to write NDJSON (e.g., out/rule_hits.ndjson).")
    ap.add_argument("--strict", action="store_true", help="Raise on validation errors (default warns only).")

    args = ap.parse_args()

    games = load_games(args.glob)
    if not games:
        sys.exit(2)

    # Extract across all games
    all_hits: List[dict] = []
    for gid, gj in games:
        hits = extract_rule_hits(gj, gid)
        all_hits.extend(hits)

    print_schema()
    print_samples(all_hits, args.show)
    validate_hits(all_hits, args.k_rules, args.strict)
    summarize_hits(all_hits)

    # Optional dumps
    if args.to_csv:
        write_csv(all_hits, args.to_csv)
    if args.to_ndjson:
        write_ndjson(all_hits, args.to_ndjson)

if __name__ == "__main__":
    main()
