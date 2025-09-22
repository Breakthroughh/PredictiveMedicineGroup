#!/usr/bin/env python3
# CMA-ES_training_pipeline.py
#
# Train rule weights β on top of CSVs produced by rule_activations_extraction.py
# using CMA-ES. Objective can be "hitrate" (majority lynches a wolf) or a smooth
# log-likelihood surrogate. Voting can be "thresholded" (tau + softmax T) or "softmax".
#
# How to use:
#  1) Edit the Config block below (paths, k_rules, objective, CMA options).
#  2) Run:  python CMA-ES_training_pipeline.py

from __future__ import annotations
import csv
import math
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Sequence
from collections import defaultdict

# =========================
# ======= Config ==========
# =========================

@dataclass
class Config:
    # Data
    csv_paths: List[str] = field(default_factory=lambda: [
        "out/batch2.1_ruleHits.csv",
    ])
    k_rules: int = 8

    # Vote model + objective
    vote_model: str = "thresholded"      # "thresholded" | "softmax"
    objective: str = "hitrate"           # "hitrate" | "loglik"
    tau: float = 2.6                     # used if vote_model == "thresholded"
    T: float = 0.75                      # softmax temperature (both modes)
    optimize_tau: bool = True            # include tau in CMA search?
    optimize_T: bool = True              # include T in CMA search?

    # Training / sampling
    seed: int = 0
    max_generations: int = 3
    popsize: int = 8                     # number of candidates per generation
    sigma0: float = 0.2                  # CMA-ES initial step size (“learning rate”)
    eval_games_per_gen: Optional[int] = None
    # If None: evaluate on ALL available games each generation.
    # If an int: sample that many distinct games per generation (fixed per gen).

    # Initialization
    init_beta: Optional[List[float]] = None   # e.g., [0.6,1.2,...] or None to start at ones
    normalize_beta: bool = True               # L2-normalize β before evaluating objective

    # Regularization (to prevent runaway scaling; complements normalization)
    l2_coeff: float = 1e-3

    # Voting details
    forbid_self_vote: bool = True

    # CMA-ES advanced options (optional)
    cma_options: Dict[str, float] = field(default_factory=lambda: {
        # e.g. 'CMA_dampfac': 1.0, 'tolx': 1e-11, 'tolfun': 1e-12,
    })

# =========================
# ====== Data Model =======
# =========================

# Each CSV row should have columns:
#   game, day, round, stage, rater, target, rule_id, archetype, is_target_wolf

from dataclasses import dataclass

@dataclass
class Hit:
    game: str
    rater: str
    target: str
    rule_id: int
    archetype: str
    is_target_wolf: bool

@dataclass
class GameBundle:
    game: str
    hits: List[Hit]
    wolves: List[str]  # list of wolf agent names (could be multiple)

# =========================
# ======= Loading =========
# =========================

def load_hits_from_csv(paths: Sequence[str]) -> Dict[str, GameBundle]:
    """Load hits from one or more CSVs and return {game_id: GameBundle}."""
    by_game: Dict[str, List[Hit]] = defaultdict(list)
    wolf_seen: Dict[str, set] = defaultdict(set)

    for path in paths:
        if not os.path.exists(path):
            print(f"[warn] CSV not found: {path}")
            continue
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    game = str(row["game"])
                    rater = str(row["rater"])
                    target = str(row["target"])
                    rid = int(row["rule_id"])
                    arche = str(row.get("archetype", "unknown"))
                    itw_raw = row.get("is_target_wolf", "False")
                    itw = str(itw_raw).strip().lower() in ("1", "true", "yes")
                except Exception as e:
                    print(f"[warn] skipping bad row in {path}: {e}")
                    continue

                hit = Hit(
                    game=game, rater=rater, target=target,
                    rule_id=rid, archetype=arche, is_target_wolf=itw
                )
                by_game[game].append(hit)
                if itw:
                    wolf_seen[game].add(target)

    bundles: Dict[str, GameBundle] = {}
    dropped = 0
    for game, hits in by_game.items():
        wolves = sorted(list(wolf_seen.get(game, set())))
        if not wolves:
            dropped += 1
            continue
        bundles[game] = GameBundle(game=game, hits=hits, wolves=wolves)

    if dropped:
        print(f"[info] Dropped {dropped} game(s) with no wolf mentions in hits (could not infer wolves).")
    print(f"[info] Loaded {len(bundles)} games with usable wolf info.")
    return bundles

# =========================
# ====== Scoring/Vote =====
# =========================

def score_table_from_hits(hits: Sequence[Hit], beta: Dict[int, float]) -> Dict[str, Dict[str, float]]:
    """Aggregate suspicion scores per rater→target using weights β."""
    scores: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for h in hits:
        w = beta.get(h.rule_id, 0.0)
        scores[h.rater][h.target] += w
    return scores

def softmax(xs: Sequence[float], T: float) -> List[float]:
    if not xs:
        return []
    m = max(xs)
    exps = [math.exp((x - m) / max(T, 1e-8)) for x in xs]
    s = sum(exps)
    if s <= 0:
        return [1.0 / len(xs)] * len(xs)
    return [e / s for e in exps]

def vote_from_scores_thresholded(
    rater_scores: Dict[str, Dict[str, float]],
    tau: float, T: float, rng: random.Random, forbid_self: bool
) -> Dict[str, str]:
    votes: Dict[str, str] = {}
    for rater, tmap in rater_scores.items():
        if not tmap:
            continue
        items = list(tmap.items())
        if forbid_self and rater in tmap:
            items = [(t, s) for t, s in items if t != rater]
        if not items:
            continue

        cands = [(t, s) for (t, s) in items if s >= tau]
        if cands:
            ts = [s for _, s in cands]
            ps = softmax(ts, T)
            pick = rng.choices(range(len(cands)), weights=ps, k=1)[0]
            votes[rater] = cands[pick][0]
        else:
            maxs = max(s for _, s in items)
            tops = sorted([t for t, s in items if s == maxs])
            votes[rater] = rng.choice(tops)
    return votes

def vote_from_scores_softmax(
    rater_scores: Dict[str, Dict[str, float]],
    T: float, rng: random.Random, forbid_self: bool
) -> Dict[str, str]:
    votes: Dict[str, str] = {}
    for rater, tmap in rater_scores.items():
        if not tmap:
            continue
        items = list(tmap.items())
        if forbid_self and rater in tmap:
            items = [(t, s) for t, s in items if t != rater]
        if not items:
            continue
        ts = [s for _, s in items]
        ps = softmax(ts, T)
        pick = rng.choices(range(len(items)), weights=ps, k=1)[0]
        votes[rater] = items[pick][0]
    return votes

def majority(votes: Dict[str, str], scores: Optional[Dict[str, Dict[str, float]]] = None) -> Optional[str]:
    tally: Dict[str, int] = defaultdict(int)
    for v in votes.values():
        tally[v] += 1
    if not tally:
        return None
    top = max(tally.values())
    winners = [t for t, c in tally.items() if c == top]
    if len(winners) == 1:
        return winners[0]
    # tie-break by highest total suspicion, then lexicographic
    if scores:
        best = None
        best_val = -1e18
        for t in winners:
            tot = sum(s.get(t, 0.0) for s in scores.values())
            if tot > best_val or (abs(tot - best_val) < 1e-12 and (best is None or t < best)):
                best_val = tot
                best = t
        return best
    return sorted(winners)[0]

# =========================
# ====== Objectives =======
# =========================

def evaluate_hitrate_on_games(
    games: Sequence[GameBundle],
    beta: Dict[int, float],
    vote_model: str,
    tau: float,
    T: float,
    seed: int,
    forbid_self: bool,
) -> float:
    rng = random.Random(seed)
    if not games:
        return 0.0
    hits = 0
    for gb in games:
        scores = score_table_from_hits(gb.hits, beta)
        if vote_model == "thresholded":
            votes = vote_from_scores_thresholded(scores, tau=tau, T=T, rng=rng, forbid_self=forbid_self)
        elif vote_model == "softmax":
            votes = vote_from_scores_softmax(scores, T=T, rng=rng, forbid_self=forbid_self)
        else:
            raise ValueError("vote_model must be 'thresholded' or 'softmax'")
        lynch = majority(votes, scores=scores)
        if lynch in gb.wolves:
            hits += 1
    return hits / len(games)

def evaluate_loglik_on_games(
    games: Sequence[GameBundle],
    beta: Dict[int, float],
    vote_model: str,
    tau: float,
    T: float,
    forbid_self: bool,
) -> float:
    eps = 1e-12
    tot_ll = 0.0
    tot_rat = 0
    for gb in games:
        scores = score_table_from_hits(gb.hits, beta)
        for rater, tmap in scores.items():
            items = list(tmap.items())
            if forbid_self and rater in tmap:
                items = [(t, s) for t, s in items if t != rater]
            if not items:
                continue

            if vote_model == "thresholded":
                cands = [(t, s) for (t, s) in items if s >= tau]
                if cands:
                    ts = [s for _, s in cands]
                    ps = softmax(ts, T)
                    p_map = {t: p for (t, _), p in zip(cands, ps)}
                else:
                    ts = [s for _, s in items]
                    ps = softmax(ts, T)
                    p_map = {t: p for (t, _), p in zip(items, ps)}
            elif vote_model == "softmax":
                ts = [s for _, s in items]
                ps = softmax(ts, T)
                p_map = {t: p for (t, _), p in zip(items, ps)}
            else:
                raise ValueError("vote_model must be 'thresholded' or 'softmax'")

            p_any_wolf = sum(p_map.get(t, 0.0) for t in gb.wolves)
            tot_ll += math.log(max(p_any_wolf, eps))
            tot_rat += 1

    if tot_rat == 0:
        return -1e9
    return tot_ll / tot_rat

# =========================
# ======== Trainer ========
# =========================

def sample_games(all_games: List[GameBundle], k: Optional[int], gen_idx: int, seed: int) -> List[GameBundle]:
    if k is None or k <= 0 or k >= len(all_games):
        return all_games
    rng = random.Random((seed + 73) * (gen_idx + 1))
    return rng.sample(all_games, k)

def build_beta_map(vec: Sequence[float]) -> Dict[int, float]:
    return {i + 1: float(vec[i]) for i in range(len(vec))}

def normalize_beta_if_needed(beta: Dict[int, float], do_norm: bool) -> Dict[int, float]:
    if not do_norm:
        return dict(beta)
    denom = math.sqrt(sum(v * v for v in beta.values())) or 1.0
    return {k: v / denom for k, v in beta.items()}

def run_training(cfg: Config):
    try:
        import cma
    except Exception as e:
        raise RuntimeError("Please install the 'cma' package:  pip install cma") from e

    # Load data
    game_bundles_map = load_hits_from_csv(cfg.csv_paths)
    all_games = list(game_bundles_map.values())
    if not all_games:
        raise RuntimeError("No usable games loaded. Check your CSV paths and contents.")

    # Init parameters vector [β1..βK] (+ tau?) (+ T?)
    if cfg.init_beta is not None:
        if len(cfg.init_beta) != cfg.k_rules:
            raise ValueError(f"init_beta must have length {cfg.k_rules}, got {len(cfg.init_beta)}")
        x0 = list(cfg.init_beta)
    else:
        x0 = [1.0] * cfg.k_rules

    param_names = [f"beta_{i+1}" for i in range(cfg.k_rules)]
    if cfg.optimize_tau:
        x0.append(cfg.tau)
        param_names.append("tau")
    if cfg.optimize_T:
        x0.append(cfg.T)
        param_names.append("T")

    def split_params(x: Sequence[float]) -> Tuple[Dict[int, float], float, float]:
        b = x[:cfg.k_rules]
        tau = x[cfg.k_rules] if cfg.optimize_tau else cfg.tau
        T = x[cfg.k_rules + (1 if cfg.optimize_tau else 0)] if cfg.optimize_T else cfg.T
        return build_beta_map(b), tau, max(T, 1e-3)  # clamp T>0

    # Objective wrapper for CMA (minimization)
    def cma_obj(x: Sequence[float], gen_idx: int) -> float:
        beta_map, tau, T = split_params(x)
        beta_eval = normalize_beta_if_needed(beta_map, cfg.normalize_beta)
        games = sample_games(all_games, cfg.eval_games_per_gen, gen_idx, cfg.seed)

        if cfg.objective == "hitrate":
            val = evaluate_hitrate_on_games(
                games, beta_eval, vote_model=cfg.vote_model,
                tau=tau, T=T, seed=cfg.seed, forbid_self=cfg.forbid_self_vote
            )
            score = val
        elif cfg.objective == "loglik":
            score = evaluate_loglik_on_games(
                games, beta_eval, vote_model=cfg.vote_model,
                tau=tau, T=T, forbid_self=cfg.forbid_self_vote
            )
        else:
            raise ValueError("objective must be 'hitrate' or 'loglik'")

        penalty = cfg.l2_coeff * sum(v * v for v in beta_map.values())
        return -(score) + penalty

    # Set up CMA-ES
    es = cma.CMAEvolutionStrategy(x0, cfg.sigma0, {
        'popsize': cfg.popsize,
        'seed': cfg.seed,
        **cfg.cma_options,
    })

    best_x = list(x0)
    best_val = float('inf')

    print(f"[info] Training with {len(all_games)} games, objective={cfg.objective}, vote_model={cfg.vote_model}")
    print(f"[info] Params: {', '.join(param_names)}")
    print(f"[info] popsize={cfg.popsize}, sigma0={cfg.sigma0}, generations={cfg.max_generations}")
    if cfg.eval_games_per_gen:
        print(f"[info] Sampling {cfg.eval_games_per_gen} game(s) per generation")

    for gen in range(cfg.max_generations):
        xs = es.ask()
        vals = [cma_obj(x, gen) for x in xs]
        es.tell(xs, vals)
        es.disp()

        # track best
        for x, v in zip(xs, vals):
            if v < best_val:
                best_val = v
                best_x = list(x)

        # Optional: quick status on full set
        beta_map, tau, T = split_params(best_x)
        beta_eval = normalize_beta_if_needed(beta_map, cfg.normalize_beta)
        if cfg.objective == "hitrate":
            cur = evaluate_hitrate_on_games(
                all_games, beta_eval, vote_model=cfg.vote_model,
                tau=tau, T=T, seed=cfg.seed, forbid_self=cfg.forbid_self_vote
            )
        else:
            cur = evaluate_loglik_on_games(
                all_games, beta_eval, vote_model=cfg.vote_model,
                tau=tau, T=T, forbid_self=cfg.forbid_self_vote
            )
        print(f"[gen {gen+1}/{cfg.max_generations}] best_objective={-best_val:.6f} (eval on full set => {cur:.6f})")

    # Final report
    beta_map, tau, T = split_params(best_x)
    beta_final = normalize_beta_if_needed(beta_map, cfg.normalize_beta)
    if cfg.objective == "hitrate":
        final_obj = evaluate_hitrate_on_games(
            all_games, beta_final, vote_model=cfg.vote_model,
            tau=tau, T=T, seed=cfg.seed, forbid_self=cfg.forbid_self_vote
        )
    else:
        final_obj = evaluate_loglik_on_games(
            all_games, beta_final, vote_model=cfg.vote_model,
            tau=tau, T=T, forbid_self=cfg.forbid_self_vote
        )

    print("\n=== Training Result ===")
    for i in range(cfg.k_rules):
        print(f"  beta[{i+1}] = {beta_final[i+1]: .6f}")
    if cfg.optimize_tau:
        print(f"  tau         = {tau: .6f}")
    if cfg.optimize_T:
        print(f"  T           = {T: .6f}")
    print(f"Objective ({cfg.objective}, vote_model={cfg.vote_model}): {final_obj:.6f}")

    return beta_final, tau, T, final_obj

# =========================
# ========= Main ==========
# =========================

if __name__ == "__main__":
    cfg = Config(
        csv_paths=[
            "out/batch1_ruleHits.csv",
        ],
        k_rules=8,

        # Objective & vote model
        objective="hitrate",           # or "loglik"
        vote_model="thresholded",      # or "softmax"
        tau=2.6,
        T=0.75,
        optimize_tau=True,
        optimize_T=True,

        # CMA-ES knobs
        seed=0,
        max_generations=60,
        popsize=12,
        sigma0=0.5,

        # Per-gen sampling (None => all games)
        eval_games_per_gen=None,

        # Start point: drop your initial 500-game weights here if you have them
        init_beta=[0.6, 1.2, 0.7, 0.5, 1.0, 0.8, 0.5, -0.4],   
        normalize_beta=True,

        # Regularization
        l2_coeff=1e-3,
    )

    run_training(cfg)



"""
We treat each rule activation (e.g., “Rule 3 fired on Agent X”) as a feature with a weight β that contributes to suspicion. 
At the end of a day, an agent’s total suspicion score is just the weighted sum of all rules triggered on them. 
The CMA-ES optimizer maintains a population of candidate β-vectors, tries them out on the training games, 
and measures how often the resulting votes hit the werewolf (the objective). 
It then perturbs and recombines the best candidates to explore better β’s across generations. 
Over time, the population “climbs” toward weightings that maximize villager success. 
The end product is a β vector (plus τ, T parameters) that makes villagers’ votes 
line up with actual werewolves as often as possible.
"""