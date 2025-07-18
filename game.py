from agents import Agent
from gemini_wrapper import gemini_respond, extract_vote, create_agent_chat
import random
import time
import os
import json
from datetime import datetime

class WerewolfGame:
    def __init__(self, num_agents=5, num_werewolves=1, log_path=None):
        self.agents = self._assign_roles(num_agents, num_werewolves)
        self.day_count = 0
        self.logs = []
        self.log_path = log_path or "log/game_log.json"
        # Prepare metadata
        self.metadata = {
            "num_agents": num_agents,
            "num_werewolves": num_werewolves,
            "start_time": datetime.now().isoformat(),
            "roles": {f"Agent{a.agent_id}": a.role for a in self.agents}
        }
        # Initialize log file
        with open(self.log_path, 'w') as f:
            json.dump({"metadata": self.metadata, "events": [], "winner": None}, f, indent=2)

    def _assign_roles(self, num_agents, num_werewolves):
        roles = ["werewolf"] * num_werewolves + ["villager"] * (num_agents - num_werewolves)
        random.shuffle(roles)
        agents = []
        for i, role in enumerate(roles):
            goal = "Stay alive as long as possible." if role == "werewolf" else "Deduce who the werewolf is."
            chat = create_agent_chat(goal, i, role)
            agents.append(Agent(i, role, goal, chat))
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
                prompt = f"{alive_context}\n{NIGHT1_PROMPT}"
                memory_for_llm = ""
            else:
                prompt = f"{alive_context}\n{NIGHT_PROMPT}"
                memory_for_llm = memory
            response = gemini_respond(wolf.chat, wolf.agent_id, "werewolf", memory_for_llm, prompt)
            print(f"\U0001f43a Agent{wolf.agent_id} (Werewolf) says:\n{response}\n")
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
                "agents_alive": [f"Agent{a.agent_id}" for a in self.get_alive_agents()]
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
            "actor": f"Agent{wolf.agent_id}",
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
                lines.append(f"{e['actor']} says: {e['content']}")
            elif e["event_type"] == "elimination":
                lines.append(f"{e['target']} was eliminated." if e['target'] else e.get('reason', 'No one was eliminated.'))
            elif e["event_type"] == "kill_vote":
                if role == "werewolf":
                    # Only werewolves see kill_vote events
                    lines.append(f"Night {e['day']}: {e['actor']} voted to kill Agent{e['vote']}.")
            # Do NOT include 'vote' events for any role
        return "\n".join(lines)

    def day_phase(self):
        print("\n--- Day Phase: Discussion ---")
        with open(self.log_path, 'r') as f:
            data = json.load(f)
        memory = self._compose_memory(data["events"], role="villager")
        alive_agents = self.get_alive_agents()
        random.shuffle(alive_agents)  # Randomize speaking order
        alive_info = ", ".join([f"Agent{a.agent_id}" for a in alive_agents])
        alive_context = f"Currently alive agents: {alive_info}."
        responses = {}
        for agent in alive_agents:
            prompt = (
                f"{alive_context}\n"
                "You may choose to say something to the group, or you may remain silent. "
                "If you wish to remain silent, simply respond with your vote line only (e.g., 'VOTE: Agent3' or 'VOTE: NOONE'). "
                "If you wish to speak, share your thoughts first, then end with your vote line. "
                "REMEMBER: The discussion is public. Do NOT reveal your role. Try to deduce who the werewolf is by looking for clues in what others say."
            )
            response = gemini_respond(agent.chat, agent.agent_id, agent.role, memory, prompt)
            responses[agent.agent_id] = response
            print(f"Agent{agent.agent_id} says:\n{response}\n")
            time.sleep(1)
            vote = extract_vote(response)
            # Determine if the agent spoke (said anything besides the vote line)
            lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
            spoke = False
            if lines:
                if len(lines) > 1:
                    spoke = True
                elif not lines[0].upper().startswith("VOTE:"):
                    spoke = True
            self._append_event({
                "timestamp": datetime.now().isoformat(),
                "phase": "day",
                "day": self.day_count,
                "event_type": "discussion",
                "actor": f"Agent{agent.agent_id}",
                "content": response if spoke else None,
                "spoke": spoke,
                "agents_alive": [f"Agent{a.agent_id}" for a in self.get_alive_agents()]
            })
        print("--- Voting ---")
        votes = {}
        vote_record = {}
        for agent_id, response in responses.items():
            vote = extract_vote(response)
            vote_record[agent_id] = f"Agent{vote}" if vote is not None else "NO VOTE"
            if vote is not None:
                votes[vote] = votes.get(vote, 0) + 1
        self._append_event({
            "timestamp": datetime.now().isoformat(),
            "phase": "day",
            "day": self.day_count,
            "event_type": "vote",
            "votes": {f"Agent{k}": v for k, v in vote_record.items()},
            "agents_alive": [f"Agent{a.agent_id}" for a in self.get_alive_agents()]
        })
        if not votes:
            self._append_event({
                "timestamp": datetime.now().isoformat(),
                "phase": "day",
                "day": self.day_count,
                "event_type": "elimination",
                "target": None,
                "reason": "No votes. No one eliminated.",
                "agents_alive": [f"Agent{a.agent_id}" for a in self.get_alive_agents()]
            })
            return None, vote_record
        # Count total number of living players for majority calculation
        num_alive = len(self.get_alive_agents())
        majority = num_alive // 2 + 1
        # Find if any candidate has strictly more than half the votes
        majority_candidates = [aid for aid, v in votes.items() if v >= majority]
        if len(majority_candidates) == 1:
            eliminated = self.agents[majority_candidates[0]]
            eliminated.alive = False
            event = {
                "timestamp": datetime.now().isoformat(),
                "phase": "day",
                "day": self.day_count,
                "event_type": "elimination",
                "target": f"Agent{eliminated.agent_id}",
                "reason": "Voted out by majority.",
                "agents_alive": [f"Agent{a.agent_id}" for a in self.get_alive_agents()]
            }
            print(f"Agent{eliminated.agent_id} was voted out during Day {self.day_count}.")
            self._append_event(event)
            return eliminated.agent_id, vote_record
        else:
            event = {
                "timestamp": datetime.now().isoformat(),
                "phase": "day",
                "day": self.day_count,
                "event_type": "elimination",
                "target": None,
                "reason": "No majority. No one eliminated.",
                "agents_alive": [f"Agent{a.agent_id}" for a in self.get_alive_agents()]
            }
            print("No one was eliminated due to lack of majority.")
            self._append_event(event)
            return None, vote_record

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
