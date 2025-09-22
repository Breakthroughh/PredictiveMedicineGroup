from agents import Agent
from gemini_wrapper import (
    gemini_respond,
    extract_vote,
    create_agent_chat,
    gemini_vote_with_reason,
    gemini_rate_agents,       # now T/Tw/I only
    gemini_accuse_left,       # Day-1 seed utterances (kept; neutral signal)
    gemini_update_suspicion,  # LLM applies rubric to update suspicion + per-target audit lines (villagers only)
    gemini_score_suspicion_freeform,  # NEW: werewolves assign suspicion freeform in [-10,10] (no rubric)
)
import random
import time
import os
import json
from datetime import datetime
import re  # for vote-line stripping helpers
import math  # for softmax voting
from collections import defaultdict


class WerewolfGame:
    """Core class for a game of Werewolf: 
    assigns roles, runs night/day phases, logs events, and checks win conditions.

    Key behavior:
      • Suspicion is updated ONLY by the rubric (via LLM) for VILLAGERS.
        Werewolves use freeform LLM scoring in [-10,10] (no rubric).
      • Persistent per-rater suspicion with per-round decay (×0.9 at start of each discussion round).
      • Day flow: RATING → discuss round 1 → RATING → discuss round 2 → RATING → FINAL VOTE.
    """

    def __init__(
        self,
        num_agents=5,
        num_werewolves=1,
        log_path=None,
        discussion_rounds=2,
        archetype_mode: str = "default",
        archetype_overrides: dict | None = None,
        rng_seed: int | None = None,
        # ---------- Rule-based villager voting knobs ----------
        villager_vote_policy: str = "softmax",  # "argmax" (deterministic) or "softmax"
        vote_softmax_temp: float = 0.75,        # softmax temperature (lower → peakier)
        vote_threshold: float = 2.6,            # τ used as a *bias* in shifted-softmax (not a gate)
    ):
        """
        Initialize a new game with agents, roles, archetypes, and an output log.

        Notes:
            - Suspicion decay ×0.9 is applied at the start of EACH discussion round (rounded to 1dp).
            - At each RATING step we (a) get T/Tw/I ratings and (b) update suspicion:
                • Villagers: rubric-driven update
                • Werewolves: freeform suspicion scoring (no rubric)
            - Suspicion is stored at 1 decimal place and later used for final vote.
        """
        if rng_seed is not None:
            random.seed(rng_seed)

        self.day_count = 0
        self.logs = []
        self.log_path = log_path or "log/game_log.json"
        self.discussion_rounds = max(1, int(discussion_rounds))
        self.archetype_mode = archetype_mode
        self.archetype_overrides = archetype_overrides or {}
        self.rng_seed = rng_seed  # persist for reproducibility in logs

        # Voting/policy knobs
        self.villager_vote_policy = villager_vote_policy
        self.vote_softmax_temp = max(1e-6, float(vote_softmax_temp))
        self.vote_threshold = float(vote_threshold)

        self.agents = self._assign_roles(num_agents, num_werewolves)

        # Persistent seating order for Day-1 seeding
        self.seating_order = random.sample(range(len(self.agents)), len(self.agents))
        self._did_seed_day1 = False

        # Rubric text (passed verbatim to the LLM) — tuned to reduce early over-penalization of normal chatter
        # and to add a small pro-social negative bump. Weights softened as requested.
        self.rubric_text = (
            "Rules rubric (4 villagers, 1 werewolf; no public votes):\n"
            "- Rule 1 Accusation (seed or otherwise): if i accuses j → suspicion(j) += +0.39\n"
            "- Rule 2 Reciprocal accusation: if i accuses j and j then accuses i back → suspicion(j) += +0.47\n"
            "- Rule 3 Persistent tunnel: if j accuses the same person across ≥2 consecutive rounds → suspicion(j) += 0.02\n"
            "- Rule 4 Scatter/flip accusations (down-weighted): if j accuses ≥2 different players in the same day or flips between rounds → suspicion(j) += +0.67\n"
            "- Rule 5 Deflection: if j is under fire and redirects suspicion onto a NEW target (not one of their accusers) → suspicion(j) += +0.38\n"
            "- Rule 6 Coalition defense: if j repeatedly defends or pushes an alternative vs a widely suspected leader across rounds → suspicion(j) += 0.00\n"
            "- Rule 7 Opportunistic pile-on: if j only accuses after ≥2 others already named the same target → suspicion(j) += +0.16\n"
            "- Rule 8 Pro-social consistency (NEW): if j asks clarifying Qs, de-escalates, or seeks consensus WITHOUT shifting targets → suspicion(j) += -0.10\n"
            "Note: The Day-1 seed ritual is intentionally noisy; it COUNTS for Rule 1 but has no extra bonuses/penalties.\n"
            "Apply mild, additive updates based on the latest public discussion/events only. Keep numbers in [-10, 10]."
        )

        # Persistent per-rater suspicion state (LLM-owned; we store it)
        # suspicion_state['AgentA'] = {'AgentB': float, ...} over alive others.
        self.suspicion_state: dict[str, dict[str, float]] = defaultdict(dict)

        # Momentum: remember each rater's last top-suspect at the most recent rating checkpoint (for a small vote bias)
        self.last_top_suspect: dict[str, str | None] = defaultdict(lambda: None)

        # Init log metadata
        self.metadata = {
            "num_agents": num_agents,
            "num_werewolves": num_werewolves,
            "discussion_rounds": self.discussion_rounds,
            "start_time": datetime.now().isoformat(),
            "roles": {f"Agent{a.agent_id}": a.role for a in self.agents},
            "archetypes": {f"Agent{a.agent_id}": a.archetype for a in self.agents},
            "seating_order": [f"Agent{i}" for i in self.seating_order],
            "villager_vote_policy": self.villager_vote_policy,
            "vote_softmax_temp": self.vote_softmax_temp,
            "vote_threshold": self.vote_threshold,
            "rubric_text": self.rubric_text,
            "rng_seed": self.rng_seed,
        }
        with open(self.log_path, "w") as f:
            json.dump({"metadata": self.metadata, "events": [], "winner": None}, f, indent=2)

    def _assign_roles(self, num_agents, num_werewolves):
        roles = ["werewolf"] * num_werewolves + ["villager"] * (num_agents - num_werewolves)
        random.shuffle(roles)

        from agents import WEREWOLF_ARCHETYPES, VILLAGER_ARCHETYPES
        ww_names = [k for k in WEREWOLF_ARCHETYPES.keys() if k != "default"]
        vg_names = [k for k in VILLAGER_ARCHETYPES.keys() if k != "default"]

        agents = []
        for i, role in enumerate(roles):
            if role == "werewolf":
                goal = "Stay alive as long as possible and try to win by killing all other villagers."
            else:
                goal = "You are a villager. Try to deduce who are the werewolves and eliminate them. Also try to minimise villager casualties."
            if i in self.archetype_overrides:
                chosen = self.archetype_overrides[i]
            elif self.archetype_mode == "random":
                chosen = random.choice(ww_names if role == "werewolf" else vg_names)
            else:
                chosen = "default"
            chat = create_agent_chat(goal, i, role, discussion_rounds=self.discussion_rounds)
            agents.append(Agent(i, role, goal, chat, archetype=chosen))
        return agents

    def _append_event(self, event):
        with open(self.log_path, "r") as f:
            data = json.load(f)
        data["events"].append(event)
        with open(self.log_path, "w") as f:
            json.dump(data, f, indent=2)

    def _update_winner(self, winner):
        with open(self.log_path, "r") as f:
            data = json.load(f)
        data["winner"] = winner
        with open(self.log_path, "w") as f:
            json.dump(data, f, indent=2)

    def get_alive_agents(self):
        return [a for a in self.agents if a.alive]

    def get_alive_werewolves(self):
        return [a for a in self.get_alive_agents() if a.role == "werewolf"]

    def get_alive_villagers(self):
        return [a for a in self.get_alive_agents() if a.role == "villager"]

    # ---------- Suspicion state helpers (LLM-owned values) ----------

    def _decay_suspicion_start_of_round(self):
        """Decay all persistent suspicion by ×0.9 at the START of each discussion round (rounded to 1dp)."""
        for rater in list(self.suspicion_state.keys()):
            for tgt in list(self.suspicion_state[rater].keys()):
                self.suspicion_state[rater][tgt] = round(self.suspicion_state[rater][tgt] * 0.9, 1)

    def _clamp_suspicion(self):
        """Clamp all suspicion values into [-10, 10] for numerical safety."""
        for rater in list(self.suspicion_state.keys()):
            for tgt in list(self.suspicion_state[rater].keys()):
                v = self.suspicion_state[rater][tgt]
                self.suspicion_state[rater][tgt] = max(-10.0, min(10.0, float(v)))

    def _ensure_alive_targets(self):
        """Ensure suspicion vectors only contain alive others; add missing alive others with 0.0."""
        alive_names = {f"Agent{a.agent_id}" for a in self.get_alive_agents()}
        for a in self.get_alive_agents():
            r = f"Agent{a.agent_id}"
            cur = self.suspicion_state[r]
            # remove dead or self
            for k in list(cur.keys()):
                if k not in alive_names or k == r:
                    cur.pop(k, None)
            # add missing
            for k in alive_names:
                if k != r and k not in cur:
                    cur[k] = 0.0

    # ---------- Voting policy from suspicion ----------

    def _choose_vote_from_suspicion(self, susp_scores: dict[str, float], rater_name: str | None = None):
        """
        Choose a vote from suspicion dict.

        Variant A: τ is a *bias* (not a gate). If policy == 'softmax', sample from
            p(j) ∝ exp((s_j - τ + momentum_bonus_j) / T)    over ALL alive opponents.

        Momentum bonus:
            small +0.2 added to the candidate that matches this rater’s last_top_suspect (if any).

        For 'argmax', pick highest suspicion; if tie, uniformly at random among ties.

        Returns (choice: 'AgentK' | None, debug: dict for logging).
        """
        debug = {
            "policy": self.villager_vote_policy,
            "tau_bias": self.vote_threshold,
            "temp": self.vote_softmax_temp,
            "probs": {},
            "u": None,
            "momentum_bonus_target": None,
            "momentum_bonus": 0.2
        }
        if not susp_scores:
            return None, debug

        # Ensure only numeric and stable 1dp floats are considered (they are stored at 1dp already)
        items = [(k, float(v)) for k, v in susp_scores.items()]
        if not items:
            return None, debug

        # Momentum boost: if this rater had a last_top_suspect, nudge that candidate
        bonus_target = self.last_top_suspect.get(rater_name or "", None)
        bonus_map = {}
        if bonus_target and bonus_target in dict(items):
            bonus_map[bonus_target] = 0.2
            debug["momentum_bonus_target"] = bonus_target

        if self.villager_vote_policy == "softmax":
            # Shifted-softmax over ALL candidates (τ as bias), include momentum
            mx = max((v - self.vote_threshold + bonus_map.get(k, 0.0)) for k, v in items)
            logits = [(k, (v - self.vote_threshold + bonus_map.get(k, 0.0) - mx) / self.vote_softmax_temp) for k, v in items]
            exps = [(k, math.exp(z)) for k, z in logits]
            Z = sum(e for _, e in exps) or 1.0
            probs = [(k, e / Z) for k, e in exps]
            debug["probs"] = {k: float(f"{p:.4f}") for k, p in probs}
            u = random.random()
            debug["u"] = float(f"{u:.6f}")
            acc = 0.0
            choice = items[-1][0]
            for k, p in probs:
                acc += p
                if u <= acc:
                    choice = k
                    break
            if rater_name:
                pretty = ", ".join([f"{k}:{p:.2f}" for k, p in probs])
                mb = f" momentum(+0.2→{bonus_target})" if bonus_target else ""
                print(f"[vote-debug] {rater_name}: policy=softmax τ(bias)={self.vote_threshold} T={self.vote_softmax_temp}{mb} "
                      f"scores={[(k, round(v,1)) for k,v in items]} probs={{{{ {pretty} }}}} u={u:.4f} → Vote={choice}")
            return choice, debug

        # ARGMAX with RANDOM tie-break on the top score (no τ filtering; include momentum by adding to the value)
        adjusted = [(k, v + bonus_map.get(k, 0.0)) for k, v in items]
        max_score = max(v for _, v in adjusted)
        top = [k for k, v in adjusted if v == max_score]
        if len(top) == 1:
            choice = top[0]
            if rater_name:
                print(f"[vote-debug] {rater_name}: policy=argmax scores={[(k, round(v,1)) for k,v in items]} "
                      f"(with momentum on {bonus_target if bonus_target else 'none'}) max={max_score:.1f} → Vote={choice}")
            return choice, debug
        else:
            u = random.random()
            idx = int(u * len(top))
            if idx == len(top):  # extremely rare when u==1.0
                idx -= 1
            choice = top[idx]
            if rater_name:
                print(f"[vote-debug] {rater_name}: policy=argmax scores={[(k, round(v,1)) for k,v in items]} "
                      f"(with momentum on {bonus_target if bonus_target else 'none'}) tie_set={top} u={u:.4f} idx={idx} → Vote={choice}")
            return choice, debug

    # ---------- Ratings + suspicion updates (villagers: rubric; werewolves: freeform) ----------

    def _collect_inter_agent_ratings_and_rubric_updates(self, stage: str, rnd: int, memory: str):
        """
        At a RATING checkpoint:
          (1) Ask each agent for T/Tw/I ratings on other ALIVE agents.
          (2) Update suspicion:
                • Villagers: apply rubric (LLM) to update their suspicion vector.
                • Werewolves: ask LLM to assign suspicion in [-10,10] freeform (no rubric).
        We then log ratings with suspicion values injected from the updated suspicion state.

        Also: track each rater's current top-suspect to provide a small momentum bonus at the final vote.
        """
        self._ensure_alive_targets()

        alive = self.get_alive_agents()
        alive_ids = [a.agent_id for a in alive]
        matrix = {}

        # --- Collect from each agent & print justification lines immediately ---
        for agent in self.agents:  # all agents provide ratings & updates
            # 1) Inter-agent ratings (T/Tw/I only)
            ratings = gemini_rate_agents(agent.chat, agent.agent_id, agent.role, memory, alive_ids)
            if "__ERROR__" in ratings:
                msg = ratings["__ERROR__"]
                print(f"[warn] Agent{agent.agent_id} inter-agent ratings error: {msg}")
                self._append_event(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "phase": "day",
                        "day": self.day_count,
                        "round": rnd,
                        "event_type": "inter_agent_ratings_error",
                        "stage": stage,
                        "rater": f"Agent{agent.agent_id}",
                        "error_message": msg,
                    }
                )
                ratings = {}

            rname = f"Agent{agent.agent_id}"
            prev_susp = {k: float(v) for k, v in self.suspicion_state.get(rname, {}).items()}

            # 2) Suspicion update: villagers (rubric) vs werewolves (freeform)
            if agent.role == "villager":
                update = gemini_update_suspicion(
                    chat=agent.chat,
                    agent_id=agent.agent_id,
                    role=agent.role,
                    memory=memory,
                    rubric_text=self.rubric_text,
                    alive_agent_ids=alive_ids,
                    previous_suspicion=prev_susp
                )
                new_susp = update.get("suspicion", {}) if isinstance(update, dict) else {}
                rationale_lines = update.get("rationale_lines", []) if isinstance(update, dict) else []
                source = "rubric"
            else:
                update = gemini_score_suspicion_freeform(
                    chat=agent.chat,
                    agent_id=agent.agent_id,
                    role=agent.role,
                    memory=memory,
                    alive_agent_ids=alive_ids,
                    previous_suspicion=prev_susp
                )
                new_susp = update.get("suspicion", {}) if isinstance(update, dict) else {}
                # rationale_lines may be absent; synthesize simple lines
                rationale_lines = update.get("rationale_lines", []) if isinstance(update, dict) else []
                source = "freeform"

            # Persist suspicion vector (clamped & 1dp)
            cleaned = {}
            for i in alive_ids:
                tgt = f"Agent{i}"
                if tgt == rname:
                    continue
                v = float(new_susp.get(tgt, 0.0))
                v = max(-10.0, min(10.0, v))
                cleaned[tgt] = round(v, 1)
            self.suspicion_state[rname] = cleaned

            # Track current top-suspect for momentum (argmax over cleaned; ties broken stably by name)
            top_tgt = None
            if cleaned:
                max_val = max(cleaned.values())
                # deterministic order: sort names then pick first with max
                for cname in sorted(cleaned.keys()):
                    if cleaned[cname] == max_val:
                        top_tgt = cname
                        break
            self.last_top_suspect[rname] = top_tgt

            # Inject suspicion into the ratings payload for logging (ALL roles)
            for i in alive_ids:
                tgt = f"Agent{i}"
                if tgt == rname:
                    continue
                ratings.setdefault(tgt, {})
                ratings[tgt]["suspicion"] = round(self.suspicion_state[rname].get(tgt, 0.0), 1)

            # ---- PRINT justification lines BEFORE the matrix ----
            print(f"\n[{source}] Agent{agent.agent_id} justification for suspicion updates (stage={stage}, round={rnd}):")
            if isinstance(rationale_lines, list) and rationale_lines:
                for line in rationale_lines:
                    print(f"  - {line}")
            else:
                # Fallback: synthesize simple prev->new lines if not provided
                for tgt, new_v in cleaned.items():
                    prev_v = prev_susp.get(tgt, 0.0)
                    print(f"  - Suspicion score {tgt}: {prev_v:.1f} -> {new_v:.1f} ({source}).")

            # Persist per-rater event
            event_payload = {
                "timestamp": datetime.now().isoformat(),
                "phase": "day",
                "day": self.day_count,
                "round": rnd,
                "event_type": "inter_agent_ratings",
                "stage": stage,  # 'pre', 'mid', 'post'
                "rater": rname,
                "ratings": ratings,                       # T/Tw/I + suspicion (from rubric OR freeform)
                "rubric_applied": (source == "rubric"),
                "rubric_text": self.rubric_text if source == "rubric" else None,
                "update_source": source,
                "rubric_rationale_lines": rationale_lines if source == "rubric" else [],
                "rubric_prev_suspicion": {k: round(v, 1) for k, v in prev_susp.items()},
                "rubric_new_suspicion": {k: round(v, 1) for k, v in cleaned.items()},
                "last_top_suspect": top_tgt,
                "agents_alive": [f"Agent{a.agent_id}" for a in self.get_alive_agents()],
            }
            self._append_event(event_payload)
            matrix[rname] = ratings

        # --- After printing all justifications, print the compact matrix snapshot ---
        snapshot_event = {
            "timestamp": datetime.now().isoformat(),
            "phase": "day",
            "day": self.day_count,
            "round": rnd,
            "event_type": "inter_agent_ratings_snapshot",
            "stage": stage,
            "matrix": matrix,
            "agents_alive": [f"Agent{a.agent_id}" for a in self.get_alive_agents()],
        }
        self._append_event(snapshot_event)

        # Debug print
        try:
            cols = [f"Agent{x}" for x in alive_ids]
            header = " " * 7 + " | ".join([f"{c:^17}" for c in cols])
            print("\n[debug] Inter-agent ratings snapshot "
                  f"(day {self.day_count}, round {rnd}, stage {stage}) — (Truth,Trust,Influence,Suspicion)")
            print(header)
            print("-" * len(header))
            for rater, row in matrix.items():
                cells = []
                for c in cols:
                    cell = row.get(c, {})
                    t = cell.get("truthfulness", "")
                    w = cell.get("trustworthiness", "")
                    i = cell.get("influence", "")
                    s = cell.get("suspicion", "")
                    s_disp = f"{s:.1f}" if isinstance(s, (int, float)) else str(s)
                    cells.append(f"({str(t):>2},{str(w):>2},{str(i):>2},{s_disp:>4})".rjust(17))
                print(f"{rater:>7} " + " | ".join(cells))
            print()
        except Exception:
            pass

    @staticmethod
    def _strip_vote_line(text: str) -> str:
        """Remove a final 'VOTE:' line from a block of text (case-insensitive)."""
        if not text:
            return text
        lines = [ln for ln in text.splitlines()]
        idx = len(lines) - 1
        while idx >= 0 and lines[idx].strip() == "":
            idx -= 1
        if idx >= 0:
            last = lines[idx].strip()
            if re.match(r"^VOTE\s*:\s*(?:AGENT\d+|NOONE)\s*$", last, flags=re.I):
                del lines[idx]
                while lines and lines[-1].strip() == "":
                    lines.pop()
        return "\n".join(lines)

    # -------------------- Day-1 accuse-left seeding --------------------

    def _next_alive_left_neighbor(self, src_id: int, alive_set: set[int]) -> int | None:
        """Return the next alive agent id clockwise on the ring from src_id."""
        if not alive_set or len(alive_set) <= 1:
            return None
        ring = self.seating_order
        pos = {aid: idx for idx, aid in enumerate(ring)}
        if src_id not in pos:
            return None
        N = len(ring)
        i = (pos[src_id] + 1) % N
        while ring[i] != src_id:
            if ring[i] in alive_set:
                return ring[i]
            i = (i + 1) % N
        return None

    def _seed_accusations(self, memory_for_public: str):
        """Create a Day-1 'accuse-left' ritual (no rule weight; just to stir noise)."""
        alive_agents = self.get_alive_agents()
        alive_ids = {a.agent_id for a in alive_agents}
        if len(alive_ids) <= 1:
            return
        pairs = {}
        utterances = {}
        order_names = [f"Agent{i}" for i in self.seating_order]
        for a in alive_agents:
            target = self._next_alive_left_neighbor(a.agent_id, alive_ids)
            if target is None:
                continue
            pairs[f"Agent{a.agent_id}"] = f"Agent{target}"
            style = a.get_style_instructions()
            msg = gemini_accuse_left(
                a.chat, a.agent_id, memory_for_public, target_name=f"Agent{target}", style=style
            )
            if msg:
                msg = msg.strip().splitlines()[0][:300]
            utterances[f"Agent{a.agent_id}"] = msg
        event = {
            "timestamp": datetime.now().isoformat(),
            "phase": "day",
            "day": self.day_count,
            "round": 0,
            "event_type": "seed_accusation_round",
            "pattern": "left",
            "order": order_names,
            "pairs": pairs,
            "utterances": utterances,
            "agents_alive": [f"Agent{a.agent_id}" for a in self.get_alive_agents()],
        }
        print("\n[seed] First day 'accuse-left' ritual created.")
        for k, v in pairs.items():
            print(f"  {k} → {v} :: {utterances.get(k, '')}")
        self._append_event(event)

    # -------------------- Night phase (unchanged logic) --------------------

    def night_phase(self):
        """Run the night phase: 2 private werewolf discussion rounds, then kill one target by final-round plurality."""
        self.day_count += 1
        werewolves = self.get_alive_werewolves()
        candidates = [a for a in self.get_alive_agents() if a.role != "werewolf"]
        if not candidates:
            return None

        print("\n--- Night Phase ---")
        with open(self.log_path, "r") as f:
            data = json.load(f)
        memory_for_wolves = self._compose_memory(data["events"], role="werewolf")

        alive_agents = self.get_alive_agents()
        alive_info = ", ".join([f"Agent{a.agent_id}" for a in alive_agents])
        alive_context = f"Currently alive agents: {alive_info}."

        # Werewolf private discussion (2 rounds)
        wolf_order = list(werewolves)
        random.shuffle(wolf_order)

        last_round_votes_counter = {}
        num_wolf_rounds = 2

        for rnd in range(1, num_wolf_rounds + 1):
            print(f"\n--- Werewolf Private Discussion Round {rnd}/{num_wolf_rounds} ---")
            responses = {}
            wolves_list_str = ", ".join([f"Agent{w.agent_id}" for w in werewolves])

            night_discuss_instructions = (
                f"This is a PRIVATE werewolf-only discussion (Round {rnd} of {num_wolf_rounds}). "
                "Coordinate subtly. You may speak or remain silent. "
                "End with exactly one vote line indicating who the werewolves should kill tonight.\n"
                "IMPORTANT: End with 'VOTE: AgentX' or 'VOTE: NOONE'."
            )

            for wolf in wolf_order:
                if not wolf.alive:
                    continue
                style = wolf.get_style_instructions()
                style_block = f"\nSTYLE INSTRUCTIONS:\n{style}\n" if style else ""
                if self.day_count == 1:
                    night_header = (
                        "It's Night 1. No DAYTIME discussions have occurred before. "
                        "Private werewolf chat only.\n"
                        f"Werewolves present: {wolves_list_str}.\n"
                        f"{alive_context}\n"
                    )
                else:
                    night_header = f"Werewolves present: {wolves_list_str}.\n{alive_context}\n"
                prompt = f"{night_header}{night_discuss_instructions}{style_block}\nSpeak if helpful, then end with your vote line."
                response = gemini_respond(wolf.chat, wolf.agent_id, "werewolf", memory_for_wolves, prompt)
                print(f"\U0001f43a Agent{wolf.agent_id} (Werewolf — {wolf.archetype}) [Night Round {rnd}] says:\n{response}\n")
                responses[wolf.agent_id] = response
                time.sleep(3)

                lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
                spoke = False
                if lines:
                    if len(lines) > 1:
                        spoke = True
                    elif not lines[0].upper().startswith("VOTE:"):
                        spoke = True

                self._append_event(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "phase": "night",
                        "day": self.day_count,
                        "round": rnd,
                        "event_type": "wolf_discussion_round",
                        "actor": f"Agent{wolf.agent_id}",
                        "content": response if spoke else None,
                        "spoke": spoke,
                        "wolves_present": [f"Agent{w.agent_id}" for w in werewolves],
                        "agents_alive": [f"Agent{a.agent_id}" for a in self.get_alive_agents()],
                        "archetype": wolf.archetype,
                    }
                )

            vote_record = {}
            votes_counter = {}
            for wolf_id, response in responses.items():
                v = extract_vote(response)
                vote_record[wolf_id] = f"Agent{v}" if v is not None else "NO VOTE"
                if v is not None:
                    votes_counter[v] = votes_counter.get(v, 0) + 1

            self._append_event(
                {
                    "timestamp": datetime.now().isoformat(),
                    "phase": "night",
                    "day": self.day_count,
                    "round": rnd,
                    "event_type": "wolf_vote_round",
                    "votes": {f"Agent{k}": v for k, v in vote_record.items()},
                    "wolves_present": [f"Agent{w.agent_id}" for w in werewolves],
                    "agents_alive": [f"Agent{a.agent_id}" for a in self.get_alive_agents()],
                }
            )
            last_round_votes_counter = votes_counter

        if not last_round_votes_counter:
            return None

        target_id = max(set(last_round_votes_counter.keys()), key=last_round_votes_counter.get)
        victim = self.agents[target_id]
        if not victim.alive:
            return None

        for wolf in werewolves:
            self._append_event(
                {
                    "timestamp": datetime.now().isoformat(),
                    "phase": "night",
                    "day": self.day_count,
                    "event_type": "kill_vote",
                    "actor": f"Agent{wolf.agent_id}",
                    "justification": f"[Private Night Round 2 final preference] VOTE: Agent{target_id}",
                    "vote": target_id,
                    "agents_alive": [f"Agent{a.agent_id}" for a in self.get_alive_agents()],
                    "archetype": wolf.archetype,
                }
            )

        victim.alive = False
        event = {
            "timestamp": datetime.now().isoformat(),
            "phase": "night",
            "day": self.day_count,
            "event_type": "kill",
            "actor": "Werewolves",
            "target": f"Agent{victim.agent_id}",
            "agents_alive": [f"Agent{a.agent_id}" for a in self.get_alive_agents()],
        }
        print(f"Night {self.day_count}: Agent{victim.agent_id} was killed.")
        self._append_event(event)
        return victim.agent_id

    def _compose_memory(self, events, role):
        """Build a role-specific textual summary of past events for LLM context (wolves see private wolf items)."""
        lines = []
        for e in events:
            if e["event_type"] == "kill":
                lines.append(f"Night {e['day']}: {e['target']} was killed.")
            elif e["event_type"] == "seed_accusation_round":
                pairs = e.get("pairs", {})
                order = e.get("order") or sorted(pairs.keys())
                seen = set()
                seq = []
                for name in order:
                    if name in pairs and name not in seen:
                        seq.append(f"{name}→{pairs[name]}")
                        seen.add(name)
                if seq:
                    lines.append(
                        f"Before Day {e.get('day')} Round 1: ritual seed (treat as noisy) — " + ", ".join(seq)
                    )
            elif e["event_type"] == "discussion_round":
                rnd = e.get("round")
                if e.get("content"):
                    if rnd is not None:
                        lines.append(f"Round {rnd} — {e['actor']} says: {e['content']}")
                    else:
                        lines.append(f"{e['actor']} says: {e['content']}")
            elif e["event_type"] == "elimination":
                lines.append(
                    f"{e['target']} was eliminated."
                    if e["target"]
                    else e.get("reason", "No one was eliminated.")
                )
            elif e["event_type"] == "kill_vote":
                if role == "werewolf":
                    lines.append(f"Night {e['day']}: {e['actor']} voted to kill Agent{e['vote']}.")
            elif e["event_type"] == "wolf_discussion_round":
                if role == "werewolf":
                    rnd = e.get("round")
                    if e.get("content"):
                        lines.append(f"[WOLF] Round {rnd} — {e['actor']} says: {e['content']}")
            elif e["event_type"] == "wolf_vote_round":
                if role == "werewolf":
                    rnd = e.get("round")
                    votes = e.get("votes", {})
                    if votes:
                        pretty = ", ".join([f"{k}→{v}" for k, v in votes.items()])
                        lines.append(f"[WOLF] Night Round {rnd} votes: {pretty}")
        return "\n".join(lines)

    # -------------------- Day phase --------------------

    def day_phase(self):
        """
        Run the day phase with suspicion updates (Villagers: rubric; Werewolves: freeform) and 3 rating checkpoints:
          RATING → discuss round 1 → RATING → discuss round 2 → RATING → FINAL VOTE
        """
        print("\n--- Day Phase: Discussion ---")
        with open(self.log_path, "r") as f:
            data = json.load(f)
        memory = self._compose_memory(data["events"], role="villager")

        # Day-1 accuse-left seeding (once)
        if self.day_count == 1 and not self._did_seed_day1:
            self._seed_accusations(memory_for_public=memory)
            self._did_seed_day1 = True
            with open(self.log_path, "r") as f:
                data = json.load(f)
            memory = self._compose_memory(data["events"], role="villager")

        # ===== RATING (before Round 1) =====
        print("\n--- Ratings BEFORE Round 1 ---")
        self._collect_inter_agent_ratings_and_rubric_updates(stage="pre", rnd=1, memory=memory)

        # ===== Discussion Round 1 =====
        print(f"\n--- Discussion Round 1/{self.discussion_rounds} ---")
        self._decay_suspicion_start_of_round()  # decay ×0.9 at start (rounded to 1dp)

        alive_agents = self.get_alive_agents()
        day_order = list(alive_agents)
        random.shuffle(day_order)
        alive_info = ", ".join([f"Agent{a.agent_id}" for a in alive_agents])
        alive_context = f"Currently alive agents: {alive_info}."

        round_instructions = (
            "This is a public discussion round. Speak briefly if helpful. "
            "Do NOT include a vote line now; the final vote happens after the last ratings."
        )
        for agent in day_order:
            if not agent.alive:
                continue
            style = agent.get_style_instructions()
            style_block = f"\nSTYLE INSTRUCTIONS:\n{style}\n" if style else ""
            prompt = (
                f"{alive_context}\n"
                f"{round_instructions}{style_block}\n"
                "Remember: do not reveal your role."
            )
            response = gemini_respond(agent.chat, agent.agent_id, agent.role, memory, prompt)
            print(f"Agent{agent.agent_id} (Round 1 — {agent.role} — {agent.archetype}) says:\n{response}\n")
            time.sleep(3)

            content_public = self._strip_vote_line(response)
            self._append_event(
                {
                    "timestamp": datetime.now().isoformat(),
                    "phase": "day",
                    "day": self.day_count,
                    "round": 1,
                    "event_type": "discussion_round",
                    "actor": f"Agent{agent.agent_id}",
                    "content": content_public,
                    "spoke": bool(content_public),
                    "agents_alive": [f"Agent{a.agent_id}" for a in self.get_alive_agents()],
                    "archetype": agent.archetype,
                }
            )

        # Update memory after round 1
        with open(self.log_path, "r") as f:
            data = json.load(f)
        memory = self._compose_memory(data["events"], role="villager")

        # ===== RATING (after Round 1) =====
        print("\n--- Ratings AFTER Round 1 ---")
        self._collect_inter_agent_ratings_and_rubric_updates(stage="mid", rnd=1, memory=memory)

        # ===== Discussion Round 2 =====
        print(f"\n--- Discussion Round 2/{self.discussion_rounds} ---")
        self._decay_suspicion_start_of_round()  # decay ×0.9 at start (rounded to 1dp)

        alive_agents = self.get_alive_agents()
        day_order = list(alive_agents)
        random.shuffle(day_order)
        alive_info = ", ".join([f"Agent{a.agent_id}" for a in alive_agents])
        alive_context = f"Currently alive agents: {alive_info}."

        for agent in day_order:
            if not agent.alive:
                continue
            style = agent.get_style_instructions()
            style_block = f"\nSTYLE INSTRUCTIONS:\n{style}\n" if style else ""
            prompt = (
                f"{alive_context}\n"
                f"{round_instructions}{style_block}\n"
                "Remember: do not reveal your role."
            )
            response = gemini_respond(agent.chat, agent.agent_id, agent.role, memory, prompt)
            print(f"Agent{agent.agent_id} (Round 2 — {agent.role} — {agent.archetype}) says:\n{response}\n")
            time.sleep(3)

            content_public = self._strip_vote_line(response)
            self._append_event(
                {
                    "timestamp": datetime.now().isoformat(),
                    "phase": "day",
                    "day": self.day_count,
                    "round": 2,
                    "event_type": "discussion_round",
                    "actor": f"Agent{agent.agent_id}",
                    "content": content_public,
                    "spoke": bool(content_public),
                    "agents_alive": [f"Agent{a.agent_id}" for a in self.get_alive_agents()],
                    "archetype": agent.archetype,
                }
            )

        # Update memory after round 2
        with open(self.log_path, "r") as f:
            data = json.load(f)
        memory = self._compose_memory(data["events"], role="villager")

        # ===== RATING (after Round 2, before FINAL VOTE) =====
        print("\n--- Ratings AFTER Round 2 (pre-FINAL VOTE) ---")
        self._collect_inter_agent_ratings_and_rubric_updates(stage="post", rnd=2, memory=memory)

        # ===== FINAL VOTE (uses updated suspicion + momentum) =====
        print("--- Final Voting (after last ratings) ---")
        alive = self.get_alive_agents()
        final_vote_record = {}

        # Ensure vectors have only alive targets
        self._ensure_alive_targets()

        # First-pass vote
        for a in alive:
            name = f"Agent{a.agent_id}"
            susp = self.suspicion_state.get(name, {})
            if a.role == "villager":
                choice, vdebug = self._choose_vote_from_suspicion(susp, rater_name=name)
                final_vote_record[a.agent_id] = choice if choice else "NO VOTE"
            else:
                # Werewolves vote with a short justification; villagers use rubric suspicion.
                alive_ids = [x.agent_id for x in alive]
                allowed_targets_line = (
                    "You may only vote among the currently ALIVE agents: "
                    + ", ".join([f"Agent{i}" for i in alive_ids if i != a.agent_id])
                    + ". If unsure, vote 'NOONE'."
                )
                header = (
                    "[FINAL DAY VOTE]\n"
                    "Give a very short justification (1–2 sentences) for your vote. End with one vote line.\n"
                    f"{allowed_targets_line}\n"
                )
                resp = gemini_vote_with_reason(a.chat, a.agent_id, a.role, memory, header)
                v = extract_vote(resp)
                final_vote_record[a.agent_id] = f"Agent{v}" if v is not None else "NO VOTE"

        # Tally strict majority
        votes_counter = {}
        for voter_id, vote_str in final_vote_record.items():
            if isinstance(vote_str, str) and vote_str.startswith("Agent"):
                tgt_id = int(vote_str.replace("Agent", ""))
                votes_counter[tgt_id] = votes_counter.get(tgt_id, 0) + 1

        num_alive = len(alive)
        majority = num_alive // 2 + 1

        # Persist the final vote event (pre-runoff)
        self._append_event(
            {
                "timestamp": datetime.now().isoformat(),
                "phase": "day",
                "day": self.day_count,
                "event_type": "final_vote_round",
                "votes": {f"Agent{k}": v for k, v in final_vote_record.items()},
                "vote_policy": {
                    "villager_policy": self.villager_vote_policy,
                    "softmax_temp": self.vote_softmax_temp,
                    "threshold": self.vote_threshold,   # interpreted as bias if softmax
                    "rng_seed": self.rng_seed,
                    "momentum_bonus": 0.2,
                },
                "agents_alive": [f"Agent{a.agent_id}" for a in self.get_alive_agents()],
            }
        )

        winners = [aid for aid, c in votes_counter.items() if c >= majority]
        if len(winners) == 1:
            eliminated = self.agents[winners[0]]
            if eliminated.alive:
                eliminated.alive = False
            event = {
                "timestamp": datetime.now().isoformat(),
                "phase": "day",
                "day": self.day_count,
                "event_type": "elimination",
                "target": f"Agent{eliminated.agent_id}",
                "reason": "Voted out by majority (final vote).",
                "agents_alive": [f"Agent{a.agent_id}" for a in self.get_alive_agents()],
            }
            print(f"Agent{eliminated.agent_id} was voted out during Day {self.day_count}.")
            self._append_event(event)
            return eliminated.agent_id, final_vote_record

        # --- Runoff: still require majority; if none, take top-2 and re-softmax to reduce no-elimination days ---
        if votes_counter:
            # Identify top-2 vote-getters (break ties randomly but reproducibly via rng_seed)
            sorted_counts = sorted(votes_counter.items(), key=lambda kv: (-kv[1], kv[0]))
            top2 = [sorted_counts[0][0]]
            if len(sorted_counts) > 1:
                # collect all ties for second place, pick one at random
                second_score = sorted_counts[1][1]
                second_pool = [aid for aid, cnt in sorted_counts[1:] if cnt == second_score]
                if second_pool:
                    pick = random.choice(second_pool)
                    top2.append(pick)
                else:
                    top2.append(sorted_counts[1][0])
            else:
                # Only one candidate had votes; choose any other alive to complete runoff
                others = [a.agent_id for a in alive if a.agent_id != top2[0]]
                if others:
                    top2.append(random.choice(others))

            top2_names = [f"Agent{tid}" for tid in top2]
            print(f"[runoff] No majority. Runoff among {top2_names} using re-softmax.")

            runoff_record = {}
            # Villagers: softmax restricted to top-2 (with τ-bias and momentum)
            # Werewolves: map to whichever of top-2 they scored higher (using their freeform suspicions)
            for a in alive:
                name = f"Agent{a.agent_id}"
                susp = self.suspicion_state.get(name, {})
                # restrict to top2
                restr = {f"Agent{tid}": susp.get(f"Agent{tid}", 0.0) for tid in top2}
                if a.role == "villager":
                    choice, _ = self._choose_vote_from_suspicion(restr, rater_name=name)
                    runoff_record[a.agent_id] = choice if choice else "NO VOTE"
                else:
                    # pick the higher-scored among top2 per werewolf suspicion
                    a_name, b_name = top2_names[0], top2_names[1]
                    choice = a_name if restr.get(a_name, 0.0) >= restr.get(b_name, 0.0) else b_name
                    runoff_record[a.agent_id] = choice

            # Tally runoff
            runoff_counter = {}
            for voter_id, vote_str in runoff_record.items():
                if isinstance(vote_str, str) and vote_str.startswith("Agent"):
                    tgt_id = int(vote_str.replace("Agent", ""))
                    runoff_counter[tgt_id] = runoff_counter.get(tgt_id, 0) + 1

            # Persist the runoff vote event
            self._append_event(
                {
                    "timestamp": datetime.now().isoformat(),
                    "phase": "day",
                    "day": self.day_count,
                    "event_type": "final_vote_runoff",
                    "top2": top2_names,
                    "votes": {f"Agent{k}": v for k, v in runoff_record.items()},
                    "agents_alive": [f"Agent{a.agent_id}" for a in self.get_alive_agents()],
                }
            )

            winners = [aid for aid, c in runoff_counter.items() if c >= majority]
            if len(winners) == 1:
                eliminated = self.agents[winners[0]]
                if eliminated.alive:
                    eliminated.alive = False
                event = {
                    "timestamp": datetime.now().isoformat(),
                    "phase": "day",
                    "day": self.day_count,
                    "event_type": "elimination",
                    "target": f"Agent{eliminated.agent_id}",
                    "reason": "Voted out by majority (runoff).",
                    "agents_alive": [f"Agent{a.agent_id}" for a in self.get_alive_agents()],
                }
                print(f"[runoff] Agent{eliminated.agent_id} was voted out after runoff.")
                self._append_event(event)
                return eliminated.agent_id, runoff_record

        # If still no majority after runoff
        event = {
            "timestamp": datetime.now().isoformat(),
            "phase": "day",
            "day": self.day_count,
            "event_type": "elimination",
            "target": None,
            "reason": "No majority after runoff. No one eliminated.",
            "agents_alive": [f"Agent{a.agent_id}" for a in self.get_alive_agents()],
        }
        print("[runoff] No elimination after runoff.")
        self._append_event(event)
        return None, final_vote_record

    def check_win_condition(self):
        """Return winner if resolved ('villagers' or 'werewolves'); otherwise None, and write winner to log when found."""
        wolves = len(self.get_alive_werewolves())
        villagers = len(self.get_alive_villagers())
        if wolves == 0:
            self._update_winner("villagers")
            return "villagers"
        if wolves >= villagers:
            self._update_winner("werewolves")
            return "werewolves"
        return None
