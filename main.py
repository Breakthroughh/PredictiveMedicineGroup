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
from game import WerewolfGame

# --- Experiment parameters ---


agent_counts = [6]   # number of agents with exactly 1 werewolf
replicates   = 1            # number of new replicates/runs for each agent count 


os.makedirs('pilotTest', exist_ok=True)

for num_agents in agent_counts:
    pattern = f'pilotTest/game_{num_agents}agents_run*_*.json'
    existing_files = glob.glob(pattern)
    existing_runs = []
    for path in existing_files:
        fname = os.path.basename(path)
        m = re.search(rf'game_{num_agents}agents_run(\d+)_', fname)
        if m:
            existing_runs.append(int(m.group(1)))
    max_run = max(existing_runs) if existing_runs else 0


    # Running the actual replicate games
    for offset in range(1, replicates + 1):
        run_idx = max_run + offset
        now = time.strftime('%Y%m%d_%H%M%S')
        game_log = f'pilotTest/game_{num_agents}agents_run{run_idx}_{now}.json'

        # Initialize game with log file name
        game = WerewolfGame(num_agents=num_agents,
                             num_werewolves=1,
                             log_path=game_log)

        print(f"\n=== Starting game with {num_agents} agents (1 werewolf), run {run_idx} ===")
        print("Assigned Roles:")
        for agent in game.agents:
            print(agent)

        # Default Day 1 (peaceful)
        print("\n========================")
        print("☀️ Day 1 (Peaceful)")
        print("========================")
        print("No one was eliminated. The village enjoys a peaceful first day.")

        # Game loop
        while True:
            print("\n========================")
            print(f"🌙 Night {game.day_count + 1}")
            print("========================")

            victim_id = game.night_phase()
            if victim_id is not None:
                print(f"Agent{victim_id} was killed during the night.")
            else:
                print("No one was killed during the night.")

            time.sleep(0.3)

            print("\n========================")
            print(f"☀️ Day {game.day_count + 1}")
            print("========================")
            eliminated_id, vote_record = game.day_phase()

            if eliminated_id is not None:
                print(f"Agent{eliminated_id} was voted out.")
            else:
                print("No one was eliminated due to a tie.")

            print("\n--- Vote Record ---")
            for voter, votee in vote_record.items():
                print(f"Agent{voter} voted for: {votee}")

            time.sleep(0.3)

            winner = game.check_win_condition()
            if winner:
                print(f"\n🏆 Game Over! {winner.capitalize()} win!")
                break

        print(f"\n=== Finished run {run_idx} for {num_agents} agents ===\n")
        # Small delay to ensure unique timestamps for each log
        time.sleep(1)
