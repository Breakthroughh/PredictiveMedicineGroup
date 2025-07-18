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
from game import WerewolfGame

# Generate timestamped log file name
now = time.strftime('%Y%m%d_%H%M%S')
game_log = f'log/game_{now}.json'

# Ensure log directory exists
os.makedirs('log', exist_ok=True)

# Initialize game with log file name
game = WerewolfGame(num_agents=5, num_werewolves=1, log_path=game_log)

print("Assigned Roles:")
for agent in game.agents:
    print(agent)

# Default Day 1 (peaceful)
print("\n========================")
print(f"☀️ Day 1 (Peaceful)")
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

# TODO: Store logs in a separate file.
