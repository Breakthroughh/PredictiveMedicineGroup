# main.py
# If you want to run game, just run python3 main.py

"""
Rules:
- Players/agents are assigned roles at the start of the game: either a werewolf or villager. 
- There is a moderator/announcer guiding the game, announcing events
- In the night phase: the werewolves discuss amongst each other and choose a target to “kill”/eliminate from the game.
- In the day phase: the villager who was just killed at night is announced. Players discuss who they think the werewolves are, based on the night’s events and player behaviour. 
  Discussion phase: We will go in a circle once, and give each agent a chance to speak. 
  At the end of the discussion, players will vote to eliminate one person. If there is a tie in votes, no one is eliminated. 
- The game continues: alternating between night and day until either all werewolves are dead, or the werewolves outnumber villagers. 
"""

import time
import os
import glob
import re
import json
import random
from game import WerewolfGame
# Import archetype helpers so we can set wolves to chosen personalities post-init
from agents import WEREWOLF_ARCHETYPES, get_archetype_prompt

# --- Experiment parameters ---

agent_counts = [7]    # number of agents per game
num_werewolves = 2    # <-- updated: 2 werewolves
replicates_per_archetype = 1  # adjust as you like

# How many rounds of discussion happen each day before final tally
discussion_rounds = 2

# Output folder for this experiment
# UPDATED: new folder name for 7 agents, 2 werewolves with inter-agent ratings
output_folder = "twoDiscussion_interAgentRating_7agents_2werewolves_2werewolfDiscussionRounds"
os.makedirs(output_folder, exist_ok=True)

def next_run_index(num_agents: int, archetype_name: str) -> int:
    """
    Resume-safe: look for existing files for this (agents, archetype) combo
    and return the next run index.
    """
    # Escape regex special chars in archetype for matching
    safe_arch = re.escape(archetype_name)
    pattern = f'{output_folder}/game_{num_agents}agents_{discussion_rounds}rounds_{archetype_name}_run*.json'
    # Glob can't use regex escapes directly, so do a broad glob then filter
    broad = glob.glob(f'{output_folder}/game_{num_agents}agents_{discussion_rounds}rounds_*_run*.json')
    run_ids = []
    for path in broad:
        fname = os.path.basename(path)
        # Match exact archetype block
        m = re.match(rf'^game_{num_agents}agents_{discussion_rounds}rounds_{safe_arch}_run(\d+)\.json$', fname)
        if m:
            run_ids.append(int(m.group(1)))
    return (max(run_ids) + 1) if run_ids else 1

for archetype_name in WEREWOLF_ARCHETYPES.keys():  # includes "default" baseline
    print(f"\n[experiment] Archetype: {archetype_name}")
    for num_agents in agent_counts:
        start_idx = next_run_index(num_agents, archetype_name)
        for offset in range(replicates_per_archetype):
            run_idx = start_idx + offset

            # Initialize game with a temporary log path (we set the final name below)
            game = WerewolfGame(
                num_agents=num_agents,
                num_werewolves=num_werewolves,   # <-- updated: 2 werewolves
                log_path=f'{output_folder}/TEMP.json',
                discussion_rounds=discussion_rounds
            )

            # --- Set everyone to default except the werewolf(s), who get archetype_name ---
            for wolf in game.get_alive_werewolves():
                wolf.archetype = archetype_name
                wolf.archetype_prompt = get_archetype_prompt(wolf.role, archetype_name)

            # Build final file name: include archetype; keep it short (no timestamp)
            wolf_types_str = "_".join(sorted({wolf.archetype for wolf in game.get_alive_werewolves()}))
            game.log_path = f'{output_folder}/game_{num_agents}agents_{discussion_rounds}rounds_{wolf_types_str}_run{run_idx}.json'

            # Initialize metadata WITH the full structure so the game can append events safely
            try:
                init_payload = {
                    "metadata": {
                        "archetypes": {f"Agent{a.agent_id}": a.archetype for a in game.agents},
                        "num_agents": num_agents,
                        "num_werewolves": num_werewolves,  # <-- updated: use variable (2)
                        "discussion_rounds": discussion_rounds,
                        "werewolf_archetype": archetype_name,
                        "roles": {f"Agent{a.agent_id}": a.role for a in game.agents},
                    },
                    "events": [],   # crucial to avoid KeyError in night_phase/_append_event
                    "winner": None
                }
                with open(game.log_path, 'w') as f:
                    json.dump(init_payload, f, indent=2)
            except Exception as e:
                print(f"[warn] Could not initialize log file: {e}")

            print(f"\n=== Starting game with {num_agents} agents ({num_werewolves} werewolves), {archetype_name}, run {run_idx} ===")
            print("Assigned Roles (with archetypes):")
            for agent in game.agents:
                print(f"{agent}  —  archetype: {agent.archetype}")

            # Default Day 1 (peaceful)
            print("\n========================")
            print(" DAY 1 (Peaceful)")
            print("========================")
            print("No one was eliminated. The village enjoys a peaceful first day.")

            # Game loop
            while True:
                print("\n========================")
                print(f" NIGHT {game.day_count + 1}")
                print("========================")

                victim_id = game.night_phase()
                if victim_id is not None:
                    print(f"Agent{victim_id} was killed during the night.")
                else:
                    print("No one was killed during the night.")

                time.sleep(0.3)

                print("\n========================")
                print(f" DAY {game.day_count + 1}")
                print("========================")
                eliminated_id, vote_record = game.day_phase()

                if eliminated_id is not None:
                    print(f"Agent{eliminated_id} was voted out.")
                else:
                    print("No one was eliminated due to a tie.")

                print("\n--- Final Round Vote Record ---")
                for voter, votee in vote_record.items():
                    print(f"Agent{voter} voted for: {votee}")

                time.sleep(0.3)

                winner = game.check_win_condition()
                if winner:
                    print(f"\n Game Over! {winner.capitalize()} win!")
                    break

            print(f"\n=== Finished run {run_idx} for {num_agents} agents ({archetype_name}) ===\n")
            time.sleep(1)  # ensure file writes complete
