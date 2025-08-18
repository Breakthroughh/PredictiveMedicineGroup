from agents import Agent
from gemini_wrapper import (
    gemini_respond,
    extract_vote,
    create_agent_chat,
    gemini_vote_with_reason,
    gemini_rate_agents,
    gemini_accuse_left,  # <-- NEW: one-line Day-1 seed utterances
)
import random
import time
import os
import json
from datetime import datetime
import re  # <-- added for vote-line stripping helpers


class WerewolfGame:
    """Core class for a game of Werewolf: 
    assigns roles, runs night/day phases, logs events, and checks win conditions."""

    def __init__(
        self,
        num_agents=5,
        num_werewolves=1,
        log_path=None,
        discussion_rounds=2,
        archetype_mode: str = "default",
        archetype_overrides: dict | None = None,
        rng_seed: int | None = None,
    ):
        """
        Initialize a new game with agents, roles, archetypes, and an output log.

        Args:
            num_agents (int): Total agents.
            num_werewolves (int): Number of werewolves.
            log_path (str | None): JSON log path (created/overwritten).
            discussion_rounds (int): Day rounds before final tally (>=1).
            archetype_mode (str): "default" or "random".
            archetype_overrides (dict|None): {agent_id: archetype}.
            rng_seed (int|None): Seed for reproducibility (random archetypes).

        Notes:
            - Final elimination uses ONLY the last day round's votes.
            - Night kill uses ONLY the last wolf discussion round's votes.
        """
        if rng_seed is not None:
            random.seed(rng_seed)

        self.day_count = 0
        self.logs = []
        self.log_path = log_path or "log/game_log.json"
        self.discussion_rounds = max(1, int(discussion_rounds))
        self.archetype_mode = archetype_mode
        self.archetype_overrides = archetype_overrides or {}

        self.agents = self._assign_roles(num_agents, num_werewolves)
        # A list of Agent objects 

        # Persistent seating order (random permutation) used to define "left" for Day-1 seeding
        self.seating_order = random.sample(range(len(self.agents)), len(self.agents))
        # Flag to ensure we only seed once on Day 1
        self._did_seed_day1 = False

        # Prepare metadata and initialize log file
        self.metadata = {
            "num_agents": num_agents,
            "num_werewolves": num_werewolves,
            "discussion_rounds": self.discussion_rounds,
            "start_time": datetime.now().isoformat(),
            "roles": {f"Agent{a.agent_id}": a.role for a in self.agents},
            "archetypes": {f"Agent{a.agent_id}": a.archetype for a in self.agents},
            "seating_order": [f"Agent{i}" for i in self.seating_order],  # <-- persist the circle
        }
        with open(self.log_path, "w") as f:
            json.dump({"metadata": self.metadata, "events": [], "winner": None}, f, indent=2)

    def _assign_roles(self, num_agents, num_werewolves): #Returns list of Agent objects w/ assigned roles + chooses archetypes per config.
        roles = ["werewolf"] * num_werewolves + ["villager"] * (num_agents - num_werewolves)
        random.shuffle(roles)

        # Allowed archetypes for random mode (exclude 'default' so random is meaningful)
        from agents import WEREWOLF_ARCHETYPES, VILLAGER_ARCHETYPES
        ww_names = [k for k in WEREWOLF_ARCHETYPES.keys() if k != "default"]
        vg_names = [k for k in VILLAGER_ARCHETYPES.keys() if k != "default"]

        agents = []
        for i, role in enumerate(roles):
            if role == "werewolf":
                goal = "Stay alive as long as possible and try to win by killing all other villagers."
            else:
                goal = "You are a villager. Try to deduce who are the werewolves and eliminate them. Also try to minimise villager casualties."
            # Decide archetype
            if i in self.archetype_overrides:
                chosen = self.archetype_overrides[i]
            elif self.archetype_mode == "random":
                chosen = random.choice(ww_names if role == "werewolf" else vg_names)
            else:
                chosen = "default"

            # Pass discussion_rounds so the rules primer can include it
            chat = create_agent_chat(goal, i, role, discussion_rounds=self.discussion_rounds)
            agents.append(Agent(i, role, goal, chat, archetype=chosen))
        return agents


    def _append_event(self, event): #Reads and rewrites whole log file with new event. May need to optimise later   
        with open(self.log_path, "r") as f:
            data = json.load(f)
        data["events"].append(event)
        with open(self.log_path, "w") as f:
            json.dump(data, f, indent=2)

    def _update_winner(self, winner): 
        """Write the winner field in the JSON log."""
        with open(self.log_path, "r") as f:
            data = json.load(f)
        data["winner"] = winner
        with open(self.log_path, "w") as f:
            json.dump(data, f, indent=2)

    def get_alive_agents(self):
        """Return a list of all agents still alive."""
        return [a for a in self.agents if a.alive]

    def get_alive_werewolves(self):
        """Return a list of alive agents with role 'werewolf'."""
        return [a for a in self.get_alive_agents() if a.role == "werewolf"]

    def get_alive_villagers(self):
        """Return a list of alive agents with role 'villager'."""
        return [a for a in self.get_alive_agents() if a.role == "villager"]

    # -------------------- helpers to collect snapshots (daytime) -----------------------

    #Collect vote+reason from all agents (alive + dead) at pre/post of round; targets restricted to alive. Then add event to log file
    def _collect_vote_snapshot(self, stage: str, rnd: int, memory: str):
        alive = self.get_alive_agents()
        alive_ids = {a.agent_id for a in alive}
        alive_names = [f"Agent{a.agent_id}" for a in alive]
        raters = list(self.agents)  # <-- changed: everyone votes in snapshots

        allowed_targets_line = (
            "You may only vote among the currently ALIVE agents: "
            + ", ".join(alive_names)
            + ". If unsure, vote 'NOONE'."
        )

        record = {}
        for agent in raters:
            style = agent.get_style_instructions()
            style_block = f"\nSTYLE INSTRUCTIONS:\n{style}\n" if style else ""
            header = (
                f"[{stage.upper()}-ROUND VOTE SNAPSHOT — Round {rnd}]\n"
                "Give a short justification (1–3 sentences) for your vote RIGHT NOW, "
                "based on current information. Then end with one vote line.\n"
                "Do NOT reveal your role.\n"
                f"{allowed_targets_line}\n"
                f"{style_block}"
            )
            resp = gemini_vote_with_reason(agent.chat, agent.agent_id, agent.role, memory, header)
            vote = extract_vote(resp)

            # Clamp votes to alive-only targets
            if vote is not None and vote not in alive_ids:
                # invalid vote -> treat as abstain for the snapshot
                vote = None

            record[f"Agent{agent.agent_id}"] = {
                "vote": (f"Agent{vote}" if vote is not None else None),
                "reason": resp,
            }

        self._append_event(
            {
                "timestamp": datetime.now().isoformat(),
                "phase": "day",
                "day": self.day_count,
                "round": rnd,
                "event_type": f"{stage}_vote_round",
                "by_agent": record,
                "agents_alive": [f"Agent{a.agent_id}" for a in self.get_alive_agents()],
            }
        )

    def _collect_inter_agent_ratings(self, stage: str, rnd: int, memory: str):
        """Ask each agent (alive + dead) to rate all other ALIVE agents; log per-rater and a matrix snapshot."""
        alive = self.get_alive_agents()
        alive_ids = [a.agent_id for a in alive]

        raters = list(self.agents)  # <-- changed: everyone rates
        matrix = {}

        for agent in raters:
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
                ratings = {}  # keep it empty in the log to distinguish from valid data


            # Persist per-rater event
            self._append_event(
                {
                    "timestamp": datetime.now().isoformat(),
                    "phase": "day",
                    "day": self.day_count,
                    "round": rnd,
                    "event_type": "inter_agent_ratings",
                    "stage": stage,  # 'pre' or 'post'
                    "rater": f"Agent{agent.agent_id}",
                    "ratings": ratings,  # {AgentK: {t,t,i,s}}
                    "agents_alive": [f"Agent{a.agent_id}" for a in self.get_alive_agents()],
                }
            )
            matrix[f"Agent{agent.agent_id}"] = ratings

        # Also store an aggregated matrix snapshot at this timepoint (for convenience)
        snapshot_event = {
            "timestamp": datetime.now().isoformat(),
            "phase": "day",
            "day": self.day_count,
            "round": rnd,
            "event_type": "inter_agent_ratings_snapshot",
            "stage": stage,
            "matrix": matrix,  # {AgentX: {AgentY: {...}}}
            "agents_alive": [f"Agent{a.agent_id}" for a in self.get_alive_agents()],
        }
        self._append_event(snapshot_event)

        # --- Print a compact matrix to terminal for inspection (show all four metrics) ---
        try:
            cols = [f"Agent{x}" for x in alive_ids]
            # Column header
            header = " " * 7 + " | ".join([f"{c:^17}" for c in cols])  # 17 chars fits "( t, w, i, s )"
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
                    # Represent tuple; keep blanks if missing
                    cells.append(f"({str(t):>2},{str(w):>2},{str(i):>2},{str(s):>2})".rjust(17))
                print(f"{rater:>7} " + " | ".join(cells))
            print()
        except Exception:
            # Don't let printing break the game
            pass

    # ------------------------------------------------------------------------------

    @staticmethod
    def _strip_vote_line(text: str) -> str:
        """
        Remove a single trailing 'VOTE: ...' line from a block of text (case-insensitive).
        Keeps everything else verbatim so discussion content remains public, but votes do not leak.
        """
        if not text:
            return text
        lines = [ln for ln in text.splitlines()]
        # Walk back to find the last non-empty line; if it's a vote, drop it
        idx = len(lines) - 1
        # Skip trailing empty lines
        while idx >= 0 and lines[idx].strip() == "":
            idx -= 1
        if idx >= 0:
            last = lines[idx].strip()
            if re.match(r"^VOTE\s*:\s*(?:AGENT\d+|NOONE)\s*$", last, flags=re.I):
                # Remove that single line
                del lines[idx]
                # Also trim any trailing blank lines left behind
                while lines and lines[-1].strip() == "":
                    lines.pop()
        return "\n".join(lines)

    # -------------------- Day-1 accuse-left seeding -----------------------

    def _next_alive_left_neighbor(self, src_id: int, alive_set: set[int]) -> int | None:
        """
        Given a source agent id and the alive set, walk the global seating ring clockwise
        and return the next alive agent id (skipping dead). None if no other alive.
        """
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
        """
        Create a Day-1 'accuse your left' ritual:
        - Uses the persistent seating_order ring.
        - Only ALIVE agents generate a one-line accusation aimed at provoking a reaction.
        - Logs a single 'seed_accusation_round' event with pairs + utterances.
        """
        alive_agents = self.get_alive_agents()
        alive_ids = {a.agent_id for a in alive_agents}
        if len(alive_ids) <= 1:
            return  # trivial / skip

        pairs = {}
        utterances = {}
        order_names = [f"Agent{i}" for i in self.seating_order]

        for a in alive_agents:
            target = self._next_alive_left_neighbor(a.agent_id, alive_ids)
            if target is None:
                continue
            pairs[f"Agent{a.agent_id}"] = f"Agent{target}"

            # One-line, style-aware, provocative (no vote)
            style = a.get_style_instructions()
            msg = gemini_accuse_left(
                a.chat,
                a.agent_id,
                memory_for_public,
                target_name=f"Agent{target}",
                style=style,
            )
            # Keep it one line and short in case the model rambles
            if msg:
                msg = msg.strip().splitlines()[0][:300]
            utterances[f"Agent{a.agent_id}"] = msg

        event = {
            "timestamp": datetime.now().isoformat(),
            "phase": "day",
            "day": self.day_count,
            "round": 0,  # occurs before Round 1
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

    
    def night_phase(self):
        """Run the night phase: 2 private werewolf discussion rounds, log votes, then kill one target by final-round plurality."""
        self.day_count += 1
        werewolves = self.get_alive_werewolves()
        candidates = [a for a in self.get_alive_agents() if a.role != "werewolf"]
        if not candidates:
            return None

        print("\n--- Night Phase ---")
        # Compose memory from previous events (for LLM context)
        with open(self.log_path, "r") as f:
            data = json.load(f)
        memory_for_wolves = self._compose_memory(data["events"], role="werewolf")

        alive_agents = self.get_alive_agents()
        alive_info = ", ".join([f"Agent{a.agent_id}" for a in alive_agents])
        alive_context = f"Currently alive agents: {alive_info}."

        # ===================== Werewolf private discussion (2 rounds) =====================
        wolf_order = list(werewolves)
        random.shuffle(wolf_order)  # give some variety at night as well

        last_round_votes_counter = {}
        num_wolf_rounds = 2  # fixed to 2

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

                # Determine if they actually spoke beyond the vote line
                lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
                spoke = False
                if lines:
                    if len(lines) > 1:
                        spoke = True
                    elif not lines[0].upper().startswith("VOTE:"):
                        spoke = True

                # Log wolf discussion content (private)
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

            # After all wolves speak in this night-round, snapshot their kill-votes for THIS round only
            vote_record = {}
            votes_counter = {}
            for wolf_id, response in responses.items():
                v = extract_vote(response)
                vote_record[wolf_id] = f"Agent{v}" if v is not None else "NO VOTE"
                if v is not None:
                    votes_counter[v] = votes_counter.get(v, 0) + 1

            # Persist a night-round wolf vote event (private)
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

            last_round_votes_counter = votes_counter  # keep only the latest round for the final kill decision
        # =================== end werewolf private discussion ===================

        # Decide the NIGHT kill using ONLY the final night round’s wolf votes
        if not last_round_votes_counter:
            return None  # wolves cast no valid final vote

        target_id = max(set(last_round_votes_counter.keys()), key=last_round_votes_counter.get)
        victim = self.agents[target_id]
        if not victim.alive:
            return None

        # Log individual kill_vote (legacy; visible only to wolves in memory)
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

        # Single kill per night
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
        """
        Build a role-specific textual summary of past events for LLM context (wolves see private wolf items).

        NOTE: Discussion content stored in 'discussion_round' events has had its final 'VOTE:' line removed,
        so public memory will *not* reveal per-round votes.
        """
        lines = []
        for e in events:
            if e["event_type"] == "kill":
                lines.append(f"Night {e['day']}: {e['target']} was killed.")
            elif e["event_type"] == "seed_accusations":
                # legacy / if ever used
                pairs = e.get("pairs", {})
                arrow = ", ".join([f"{k}→{v}" for k, v in pairs.items()])
                lines.append(f"Before Day {e.get('day')} Round 1: ritual seed (treat as noisy) — {arrow}")
            elif e["event_type"] == "seed_accusation_round":
                pairs = e.get("pairs", {})
                # Prefer to show in the ring order for readability
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
            elif e["event_type"] == "discussion":
                # kept for backward compatibility if older logs exist
                if e.get("content"):
                    lines.append(f"{e['actor']} says: {e['content']}")
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
            # IMPORTANT: We intentionally do NOT include daytime vote snapshots in memory
        return "\n".join(lines)

    def day_phase(self):
        """
        Run the day phase:
        - (NEW) First day 'accuse-left' ritual before Round 1.
        - Take ONE baseline snapshot before Round 1 (pre-round vote+reason and ratings).
        - Run R discussion rounds. After each round:
            * log speeches,
            * take a round vote snapshot (per-round),
            * take POST-round vote+reason snapshot,
            * take POST-round inter-agent ratings snapshot.
        - Decide elimination using ONLY the final round's votes.
        """
        print("\n--- Day Phase: Discussion ---")
        with open(self.log_path, "r") as f:
            data = json.load(f)
        memory = self._compose_memory(data["events"], role="villager")

        # ---------- NEW: Day-1 accuse-left seeding (once) ----------
        if self.day_count == 1 and not self._did_seed_day1:
            # Create the seed using the memory so far
            self._seed_accusations(memory_for_public=memory)
            self._did_seed_day1 = True
            # Rebuild memory so the seed appears to all subsequent prompts this day
            with open(self.log_path, "r") as f:
                data = json.load(f)
            memory = self._compose_memory(data["events"], role="villager")

        alive_agents = self.get_alive_agents()

        # Speaking order randomized once per day
        day_order = list(alive_agents)
        random.shuffle(day_order)
        alive_info = ", ".join([f"Agent{a.agent_id}" for a in alive_agents])
        alive_context = f"Currently alive agents: {alive_info}."

        # Track round-by-round responses and votes
        round_vote_records = []  # list of dicts: [{agent_id -> "AgentX" or "NO VOTE"}, ...]
        last_round_votes_counter = {}  # used to decide elimination at the end

        # ===== Baseline before any discussion begins (pre-round 1) =====
        print("\n--- Baseline snapshots BEFORE Round 1 ---")
        self._collect_vote_snapshot(stage="pre", rnd=1, memory=memory)
        self._collect_inter_agent_ratings(stage="pre", rnd=1, memory=memory)

        for rnd in range(1, self.discussion_rounds + 1):
            print(f"\n--- Discussion Round {rnd}/{self.discussion_rounds} ---")

            responses = {}

            round_instructions = (
                f"This is Round {rnd} of {self.discussion_rounds} for today's discussion. "
                "You may choose to say something to the group, or you may remain silent. "
                "If you wish to remain silent, simply respond with your vote line only (e.g., 'VOTE: Agent3' or 'VOTE: NOONE'). "
                "If you wish to speak, share your thoughts first, then end with your vote line.\n"
                "IMPORTANT: End your response with exactly one vote line (e.g., 'VOTE: Agent3' or 'VOTE: NOONE')."
            )

            # Day-1 Round-1 reminder to not overweight the seed
            if self.day_count == 1 and rnd == 1:
                round_instructions += (
                    "\nReminder: the pre-round 'accuse-left' statements are a ritualized starter. "
                    "Treat them as intentionally noisy; use them to test reactions, not as hard evidence."
                )

            # Collect speeches + per-round votes (ALIVE speakers only)
            for agent in day_order:
                if not agent.alive:
                    continue

                style = agent.get_style_instructions()
                style_block = f"\nSTYLE INSTRUCTIONS:\n{style}\n" if style else ""

                prompt = (
                    f"{alive_context}\n"
                    f"{round_instructions}{style_block}\n"
                    "REMEMBER: The discussion is public. Do NOT reveal your role. "
                    "Try to deduce who the werewolf is by looking for clues in what others say."
                )
                response = gemini_respond(agent.chat, agent.agent_id, agent.role, memory, prompt)
                print(f"Agent{agent.agent_id} (Round {rnd} — {agent.role} — {agent.archetype}) says:\n{response}\n")
                responses[agent.agent_id] = response
                time.sleep(3)

                # Extract vote and whether the agent actually spoke beyond the vote line
                vote = extract_vote(response)
                lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
                spoke = False
                if lines:
                    if len(lines) > 1:
                        spoke = True
                    elif not lines[0].upper().startswith("VOTE:"):
                        spoke = True

                # Log the discussion content for this round
                # IMPORTANT: Strip the final 'VOTE: ...' line from public content
                content_public = self._strip_vote_line(response) if spoke else None
                self._append_event(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "phase": "day",
                        "day": self.day_count,
                        "round": rnd,
                        "event_type": "discussion_round",
                        "actor": f"Agent{agent.agent_id}",
                        "content": content_public,  # <-- no vote line in public logs
                        "spoke": spoke,
                        "agents_alive": [f"Agent{a.agent_id}" for a in self.get_alive_agents()],
                        "archetype": agent.archetype,
                    }
                )

            # After all speak in this round, snapshot the votes for THIS round only
            votes = {}
            vote_record = {}
            for agent_id, response in responses.items():
                vote = extract_vote(response)
                vote_record[agent_id] = f"Agent{vote}" if vote is not None else "NO VOTE"
                if vote is not None:
                    votes[vote] = votes.get(vote, 0) + 1

            round_vote_records.append(vote_record)

            # Persist a round-specific vote event so you can analyze vote trajectories
            self._append_event(
                {
                    "timestamp": datetime.now().isoformat(),
                    "phase": "day",
                    "day": self.day_count,
                    "round": rnd,
                    "event_type": "vote_round",
                    "votes": {f"Agent{k}": v for k, v in vote_record.items()},
                    "agents_alive": [f"Agent{a.agent_id}" for a in self.get_alive_agents()],
                }
            )

            # ===== Post-round snapshots (explicit vote+reason and ratings) =====
            self._collect_vote_snapshot(stage="post", rnd=rnd, memory=memory)
            self._collect_inter_agent_ratings(stage="post", rnd=rnd, memory=memory)

            # Update memory for the next round
            with open(self.log_path, "r") as f:
                data = json.load(f)
            memory = self._compose_memory(data["events"], role="villager")

            # Keep the last round's tallies around to decide elimination at the end
            last_round_votes_counter = votes

        # ===== End of all rounds — decide elimination using ONLY the final round's votes =====
        print("--- Final Voting (after last round) ---")
        final_vote_record = round_vote_records[-1] if round_vote_records else {}
        if not last_round_votes_counter:
            # No votes in final round: no elimination
            self._append_event(
                {
                    "timestamp": datetime.now().isoformat(),
                    "phase": "day",
                    "day": self.day_count,
                    "event_type": "elimination",
                    "target": None,
                    "reason": "No votes in final round. No one eliminated.",
                    "agents_alive": [f"Agent{a.agent_id}" for a in self.get_alive_agents()],
                }
            )
            return None, final_vote_record

        # Count total number of living players for majority calculation
        num_alive = len(self.get_alive_agents())
        majority = num_alive // 2 + 1

        # Find if any candidate has strictly more than half the votes in the FINAL round
        majority_candidates = [aid for aid, v in last_round_votes_counter.items() if v >= majority]
        if len(majority_candidates) == 1:
            eliminated = self.agents[majority_candidates[0]]
            if eliminated.alive:
                eliminated.alive = False
            event = {
                "timestamp": datetime.now().isoformat(),
                "phase": "day",
                "day": self.day_count,
                "event_type": "elimination",
                "target": f"Agent{eliminated.agent_id}",
                "reason": "Voted out by majority in the final round.",
                "agents_alive": [f"Agent{a.agent_id}" for a in self.get_alive_agents()],
            }
            print(f"Agent{eliminated.agent_id} was voted out during Day {self.day_count}.")
            self._append_event(event)
            return eliminated.agent_id, final_vote_record
        else:
            event = {
                "timestamp": datetime.now().isoformat(),
                "phase": "day",
                "day": self.day_count,
                "event_type": "elimination",
                "target": None,
                "reason": "No majority in final round. No one eliminated.",
                "agents_alive": [f"Agent{a.agent_id}" for a in self.get_alive_agents()],
            }
            print("No one was eliminated due to lack of majority in the final round.")
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
