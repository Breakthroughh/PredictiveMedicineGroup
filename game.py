from agents import Agent
from gemini_wrapper import gemini_respond, extract_vote, create_agent_chat
import random
import time
import os
import json
from datetime import datetime

class WerewolfGame:
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
        discussion_rounds: number of day discussion passes before the final tally.
        We record a per-round vote snapshot to study how votes evolve across rounds.
        The final elimination is decided using ONLY the last round's votes.

        archetype_mode:
          - "default": all agents use 'default' archetype (no extra style instructions)
          - "random": assign each agent a random archetype appropriate to their role
        archetype_overrides: optional {agent_id: archetype_name} to force a specific style.
        rng_seed: optional seed for reproducible random archetype assignment.
        """
        if rng_seed is not None:
            random.seed(rng_seed)

        self.day_count = 0
        self.logs = []
        self.log_path = log_path or "log/game_log.json"
        self.discussion_rounds = max(1, int(discussion_rounds))  # at least 1
        self.archetype_mode = archetype_mode
        self.archetype_overrides = archetype_overrides or {}

        self.agents = self._assign_roles(num_agents, num_werewolves)

        # Prepare metadata
        self.metadata = {
            "num_agents": num_agents,
            "num_werewolves": num_werewolves,
            "discussion_rounds": self.discussion_rounds,
            "start_time": datetime.now().isoformat(),
            "roles": {f"Agent{a.agent_id}": a.role for a in self.agents},
            "archetypes": {f"Agent{a.agent_id}": a.archetype for a in self.agents},
        }
        # Initialize log file
        with open(self.log_path, 'w') as f:
            json.dump({"metadata": self.metadata, "events": [], "winner": None}, f, indent=2)

    def _assign_roles(self, num_agents, num_werewolves):
        roles = ["werewolf"] * num_werewolves + ["villager"] * (num_agents - num_werewolves)
        random.shuffle(roles)

        # Allowed archetypes for random mode (exclude 'default' so random is meaningful)
        from agents import WEREWOLF_ARCHETYPES, VILLAGER_ARCHETYPES
        ww_names = [k for k in WEREWOLF_ARCHETYPES.keys() if k != "default"]
        vg_names = [k for k in VILLAGER_ARCHETYPES.keys() if k != "default"]

        agents = []
        for i, role in enumerate(roles):
            goal = "Stay alive as long as possible." if role == "werewolf" else "Deduce who the werewolf is."
            # Decide archetype
            if i in self.archetype_overrides:
                chosen = self.archetype_overrides[i]
            elif self.archetype_mode == "random":
                chosen = random.choice(ww_names if role == "werewolf" else vg_names)
            else:
                chosen = "default"

            chat = create_agent_chat(goal, i, role)
            agents.append(Agent(i, role, goal, chat, archetype=chosen))
        return agents

    def _append_event(self, event):
        with open(self.log_path, 'r') as f:
            data = json.load(f)
        data["events"].append(event)
        with open(self.log_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _update_winner(self, winner):
        with open(self.log_path, 'r') as f:
            data = json.load(f)
        data["winner"] = winner
        with open(self.log_path, 'w') as f:
            json.dump(data, f, indent=2)

    def get_alive_agents(self):
        return [a for a in self.agents if a.alive]

    def get_alive_werewolves(self):
        return [a for a in self.get_alive_agents() if a.role == "werewolf"]

    def get_alive_villagers(self):
        return [a for a in self.get_alive_agents() if a.role == "villager"]

    def night_phase(self):
        self.day_count += 1
        werewolves = self.get_alive_werewolves()
        candidates = [a for a in self.get_alive_agents() if a.role != "werewolf"]
        if not candidates:
            return None

        NIGHT1_PROMPT = (
            "It's Night 1. No discussions or accusations have occurred yet. "
            "You are a werewolf. Your main objective is to win the game by eliminating all villagers while making sure you and your fellow werewolves do not get suspected or voted out. "
            "You must always choose a target to eliminate unless you have a very strong reason not to. Voting 'NOONE' should be extremely rare and must be justified. "
            "Never use phrases like 'If I were a werewolf...' or anything that hints at your true role. Always speak and justify your actions as if you are a villager. "
            "Pick a random villager to eliminate tonight. You can justify your choice using their agent ID or name. "
            "Do NOT reference any discussion or suspicion, as none has happened yet. "
            "Be strategic: never reveal your identity as a werewolf, and make your reasoning sound like a villager. "
            "At the end, write only one final line like: 'VOTE: Agent3' or 'VOTE: NOONE' "
            "REMEMBER: Your response will be shared publicly. Do NOT reveal that you are a werewolf."
        )
        NIGHT_PROMPT = (
            "Who should the werewolves kill tonight? Your main objective is to win by eliminating all villagers while making sure you and your fellow werewolves do not get suspected or voted out. "
            "You must always choose a target to eliminate unless you have a very strong reason not to. Voting 'NOONE' should be extremely rare and must be justified. "
            "Never use phrases like 'If I were a werewolf...' or anything that hints at your true role. Always speak and justify your actions as if you are a villager. "
            "Consider everyone who is still alive, what the villagers have discussed and what they suspect. Justify your answer in a way that does not arouse suspicion. "
            "Be strategic: never reveal your identity as a werewolf, and make your reasoning sound like a villager. "
            "At the end, write only one final line like: 'VOTE: Agent3' or 'VOTE: NOONE' "
            "REMEMBER: Your response will be shared publicly. Do NOT reveal that you are a werewolf."
        )
        print("\n--- Night Phase ---")
        # Compose memory from previous events (for LLM context)
        with open(self.log_path, 'r') as f:
            data = json.load(f)
        memory = self._compose_memory(data["events"], role="werewolf")
        alive_agents = self.get_alive_agents()
        alive_info = ", ".join([f"Agent{a.agent_id}" for a in alive_agents])
        alive_context = f"Currently alive agents: {alive_info}."
        votes = []
        for wolf in werewolves:
            # Style add-on for this agent (archetype)
            style = wolf.get_style_instructions()
            style_block = f"\nSTYLE INSTRUCTIONS:\n{style}\n" if style else ""

            if self.day_count == 1:
                agent_ids = [a.agent_id for a in alive_agents if a.agent_id != wolf.agent_id]
                agent_id_range = f"{min(agent_ids)} to {max(agent_ids)}"
                NIGHT1_PROMPT = (
                    f"It's Night 1. No discussions or accusations have occurred yet. "
                    f"You are Agent{wolf.agent_id}. There are {len(self.agents)} agents: Agent0 to Agent{len(self.agents)-1}. "
                    f"You are a werewolf. Your main objective is to win the game by eliminating all villagers while making sure you and your fellow werewolves do not get suspected or voted out. "
                    f"You must always choose a target to eliminate unless you have a very strong reason not to. Voting 'NOONE' should be extremely rare and must be justified. "
                    f"Never use phrases like 'If I were a werewolf...' or anything that hints at your true role. Always speak and justify your actions as if you are a villager. "
                    f"Pick a random agent to eliminate tonight, but do NOT pick yourself (Agent{wolf.agent_id}). "
                    f"Valid choices: Agent IDs {agent_id_range}, excluding yourself. "
                    "You can justify your choice using their agent ID or name. "
                    "Do NOT reference any discussion or suspicion, as none has happened yet. "
                    "Be strategic: never reveal your identity as a werewolf, and make your reasoning sound like a villager. "
                    "At the end, write only one final line like: 'VOTE: Agent3' or 'VOTE: NOONE' "
                    "REMEMBER: Your response will be shared publicly. Do NOT reveal that you are a werewolf."
                )
                prompt = f"{alive_context}\n{NIGHT1_PROMPT}{style_block}"
                memory_for_llm = ""
            else:
                prompt = f"{alive_context}\n{NIGHT_PROMPT}{style_block}"
                memory_for_llm = memory
            response = gemini_respond(wolf.chat, wolf.agent_id, "werewolf", memory_for_llm, prompt)
            print(f"\U0001f43a Agent{wolf.agent_id} (Werewolf — {wolf.archetype}) says:\n{response}\n")
            time.sleep(1)
            vote = extract_vote(response)
            if vote is not None:
                votes.append(vote)
            self._append_event({
                "timestamp": datetime.now().isoformat(),
                "phase": "night",
                "day": self.day_count,
                "event_type": "kill_vote",
                "actor": f"Agent{wolf.agent_id}",
                "justification": response,
                "vote": vote,
                "agents_alive": [f"Agent{a.agent_id}" for a in self.get_alive_agents()],
                "archetype": wolf.archetype,
            })
        if not votes:
            return None
        target_id = max(set(votes), key=votes.count)
        victim = self.agents[target_id]
        if not victim.alive:
            return None
        victim.alive = False
        event = {
            "timestamp": datetime.now().isoformat(),
            "phase": "night",
            "day": self.day_count,
            "event_type": "kill",
            "actor": "Werewolves",  # fix: this used to reference the last 'wolf' variable
            "target": f"Agent{victim.agent_id}",
            "agents_alive": [f"Agent{a.agent_id}" for a in self.get_alive_agents()]
        }
        print(f"Night {self.day_count}: Agent{victim.agent_id} was killed.")
        self._append_event(event)
        return victim.agent_id

    def _compose_memory(self, events, role):
        # Compose a string summary of relevant events for LLM context
        lines = []
        for e in events:
            if e["event_type"] == "kill":
                lines.append(f"Night {e['day']}: {e['target']} was killed.")
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
                lines.append(f"{e['target']} was eliminated." if e['target'] else e.get('reason', 'No one was eliminated.'))
            elif e["event_type"] == "kill_vote":
                if role == "werewolf":
                    # Only werewolves see kill_vote events
                    lines.append(f"Night {e['day']}: {e['actor']} voted to kill Agent{e['vote']}.")
            # IMPORTANT: We intentionally do NOT include any daytime vote snapshots
            # ('vote' or 'vote_round') in memory for either role to avoid degenerate
            # herd behavior purely from echoing a tally.
        return "\n".join(lines)

    def day_phase(self):
        print("\n--- Day Phase: Discussion ---")
        with open(self.log_path, 'r') as f:
            data = json.load(f)
        memory = self._compose_memory(data["events"], role="villager")

        alive_agents = self.get_alive_agents()

        # We keep the speaking order consistent across rounds so vote changes are
        # less confounded by order randomness. (You can randomize per round if you prefer.)
        day_order = list(alive_agents)
        random.shuffle(day_order)  # randomize once per day
        alive_info = ", ".join([f"Agent{a.agent_id}" for a in alive_agents])
        alive_context = f"Currently alive agents: {alive_info}."

        # Track round-by-round responses and votes
        round_vote_records = []  # list of dicts: [{agent_id -> "AgentX" or "NO VOTE"}, ...]
        last_round_votes_counter = {}  # used to decide elimination at the end

        for rnd in range(1, self.discussion_rounds + 1):
            print(f"\n--- Discussion Round {rnd}/{self.discussion_rounds} ---")
            responses = {}

            # Round-specific prompt addendum, ensuring every response ends with a vote.
            round_instructions = (
                f"This is Round {rnd} of {self.discussion_rounds} for today's discussion. "
                "You may choose to say something to the group, or you may remain silent. "
                "If you wish to remain silent, simply respond with your vote line only (e.g., 'VOTE: Agent3' or 'VOTE: NOONE'). "
                "If you wish to speak, share your thoughts first, then end with your vote line.\n"
                "IMPORTANT: End your response with exactly one vote line (e.g., 'VOTE: Agent3' or 'VOTE: NOONE')."
            )

            # Collect speeches + per-round votes
            for agent in day_order:
                if not agent.alive:
                    continue

                # Style block for each agent (villager or werewolf)
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
                time.sleep(1)

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
                self._append_event({
                    "timestamp": datetime.now().isoformat(),
                    "phase": "day",
                    "day": self.day_count,
                    "round": rnd,
                    "event_type": "discussion_round",
                    "actor": f"Agent{agent.agent_id}",
                    "content": response if spoke else None,
                    "spoke": spoke,
                    "agents_alive": [f"Agent{a.agent_id}" for a in self.get_alive_agents()],
                    "archetype": agent.archetype,
                })

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
            self._append_event({
                "timestamp": datetime.now().isoformat(),
                "phase": "day",
                "day": self.day_count,
                "round": rnd,
                "event_type": "vote_round",
                "votes": {f"Agent{k}": v for k, v in vote_record.items()},
                "agents_alive": [f"Agent{a.agent_id}" for a in self.get_alive_agents()]
            })

            # Update memory for the next round (so round t can reference speeches from round t-1)
            with open(self.log_path, 'r') as f:
                data = json.load(f)
            memory = self._compose_memory(data["events"], role="villager")

            # Keep the last round's tallies around to decide elimination at the end
            last_round_votes_counter = votes

        # ===== End of all rounds — decide elimination using ONLY the final round's votes =====
        print("--- Final Voting (after last round) ---")
        final_vote_record = round_vote_records[-1] if round_vote_records else {}
        if not last_round_votes_counter:
            # No votes in final round: no elimination
            self._append_event({
                "timestamp": datetime.now().isoformat(),
                "phase": "day",
                "day": self.day_count,
                "event_type": "elimination",
                "target": None,
                "reason": "No votes in final round. No one eliminated.",
                "agents_alive": [f"Agent{a.agent_id}" for a in self.get_alive_agents()]
            })
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
                "agents_alive": [f"Agent{a.agent_id}" for a in self.get_alive_agents()]
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
                "agents_alive": [f"Agent{a.agent_id}" for a in self.get_alive_agents()]
            }
            print("No one was eliminated due to lack of majority in the final round.")
            self._append_event(event)
            return None, final_vote_record

    def check_win_condition(self):
        wolves = len(self.get_alive_werewolves())
        villagers = len(self.get_alive_villagers())
        if wolves == 0:
            self._update_winner("villagers")
            return "villagers"
        if wolves >= villagers:
            self._update_winner("werewolves")
            return "werewolves"
        return None
