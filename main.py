# main.py
# If you want to run game, just run:
#   python3 main.py --villager_team "Analyst" --half 1 --run_tag batch4.6_homogeneous_villagerTeam --replicates 20
# and in a second terminal for the same villager_team:
#   python3 main.py --villager_team "Analyst" --half 2 --run_tag batch4.6_homogeneous_villagerTeam --replicates 20

"""
Rules:
- Players/agents are assigned roles at the start of the game: either a werewolf or villager.
- There is a moderator/announcer guiding the game, announcing events
- In the night phase: the werewolves discuss amongst each other and choose a target to “kill”/eliminate from the game.
- In the day phase: ratings → discuss round 1 → ratings → discuss round 2 → ratings → final vote.
- A strict majority is required; if none, a runoff among top-2 is held to reduce no-elimination days.
- The game continues night/day until either all werewolves are dead, or the werewolves outnumber villagers.
"""

import time
import os
import glob
import re
import json
import argparse

from game import WerewolfGame
from agents import WEREWOLF_ARCHETYPES, VILLAGER_ARCHETYPES, get_archetype_prompt


# --- Experiment parameters (defaults; can be extended to CLI later if desired) ---
agent_counts = [5]
num_werewolves = 1
discussion_rounds = 2

# Default output root if --run_tag not provided
DEFAULT_RUN_TAG = "batch4.6_homogeneous_villagerTeam"


def next_run_index(output_folder: str, num_agents: int, wolf_arch: str, villager_team: str) -> int:
    """Find next run id for (agents, wolf_arch, villager_team) combo within output_folder."""
    def tagify(name: str) -> str:
        return re.sub(r"\s+", "_", name.strip())

    safe_w = re.escape(wolf_arch)
    safe_vtag = re.escape(tagify(villager_team))  # match the filename's __vg_{tag}
    pattern = rf'^game_{num_agents}agents_{discussion_rounds}rounds_{safe_w}__vg_{safe_vtag}_run(\d+)\.json$'

    broad = glob.glob(os.path.join(output_folder, f'game_{num_agents}agents_{discussion_rounds}rounds_*__vg_*_run*.json'))
    run_ids = []
    for path in broad:
        fname = os.path.basename(path)
        m = re.match(pattern, fname)
        if m:
            run_ids.append(int(m.group(1)))
    return (max(run_ids) + 1) if run_ids else 1


def main():
    # --- CLI args for parallel sharded runs ---
    parser = argparse.ArgumentParser(description="Run werewolf/villager sweeps in parallel shards.")
    parser.add_argument("--villager_team", type=str, required=True,
                        help="Villager archetype to run (e.g., 'Analyst', 'Archivist', 'Interrogator', 'Closer', 'default', ...).")
    parser.add_argument("--half", type=int, choices=[1, 2], required=True,
                        help="Which half of werewolf archetypes to run: 1 (first half) or 2 (second half).")
    parser.add_argument("--run_tag", type=str, default=DEFAULT_RUN_TAG,
                        help="Root folder name for outputs (will create subfolder per villager team).")
    parser.add_argument("--replicates", type=int, default=20,
                        help="Replicates per (villager, werewolf) pair in this shard.")
    args = parser.parse_args()

    villager_team = args.villager_team
    if villager_team not in VILLAGER_ARCHETYPES:
        raise ValueError(f"Unknown villager_team '{villager_team}'. "
                         f"Valid options include: {list(VILLAGER_ARCHETYPES.keys())}")

    # Deterministic split of werewolf archetypes into two halves
    sorted_wolves = sorted(WEREWOLF_ARCHETYPES.keys())
    # Split as evenly as possible; if odd, second half gets the extra
    mid = len(sorted_wolves) // 2
    if len(sorted_wolves) % 2 != 0:
        mid = (len(sorted_wolves) + 1) // 2
    werewolf_subset = sorted_wolves[:mid] if args.half == 1 else sorted_wolves[mid:]

    output_root = args.run_tag
    os.makedirs(output_root, exist_ok=True)

    # Each villager archetype gets its own subfolder under run_tag
    output_folder = os.path.join(output_root, villager_team.replace(" ", "_"))
    os.makedirs(output_folder, exist_ok=True)

    replicates_per_pair = args.replicates

    # Sharded run: exactly one villager team per process, and either the first or second half of wolves
    for wolf_arch in werewolf_subset:
        print(f"\n[experiment] Wolf={wolf_arch}  |  Villagers (homogeneous)={villager_team}  |  half={args.half}")
        for num_agents in agent_counts:
            start_idx = next_run_index(output_folder, num_agents, wolf_arch, villager_team)
            for offset in range(replicates_per_pair):
                run_idx = start_idx + offset
                rng_seed = run_idx

                game = WerewolfGame(
                    num_agents=num_agents,
                    num_werewolves=num_werewolves,
                    log_path=os.path.join(output_folder, 'TEMP.json'),
                    discussion_rounds=discussion_rounds,
                    villager_vote_policy="softmax",
                    vote_softmax_temp=1.05,
                    vote_threshold=3.18,
                    rng_seed=rng_seed,
                )

                # Force werewolf archetype for this run
                for wolf in game.get_alive_werewolves():
                    wolf.archetype = wolf_arch
                    wolf.archetype_prompt = get_archetype_prompt(wolf.role, wolf_arch)

                # Force homogeneous villager team for this run
                for vg in game.get_alive_villagers():
                    vg.archetype = villager_team
                    vg.archetype_prompt = get_archetype_prompt(vg.role, villager_team)

                wolf_types_str = "_".join(sorted({wolf.archetype for wolf in game.get_alive_werewolves()}))
                vg_team_tag = villager_team.replace(" ", "_")
                game.log_path = os.path.join(
                    output_folder,
                    f'game_{num_agents}agents_{discussion_rounds}rounds_{wolf_types_str}__vg_{vg_team_tag}_run{run_idx}.json'
                )

                # Initialize the log stub (now includes villager team label)
                try:
                    init_payload = {
                        "metadata": {
                            "archetypes": {f"Agent{a.agent_id}": a.archetype for a in game.agents},
                            "num_agents": num_agents,
                            "num_werewolves": num_werewolves,
                            "discussion_rounds": discussion_rounds,
                            "werewolf_archetype": wolf_arch,
                            "villager_team": villager_team,
                            "roles": {f"Agent{a.agent_id}": a.role for a in game.agents},
                            "rng_seed": rng_seed,
                        },
                        "events": [],
                        "winner": None
                    }
                    with open(game.log_path, 'w') as f:
                        json.dump(init_payload, f, indent=2)
                except Exception as e:
                    print(f"[warn] Could not initialize log file: {e}")

                print(f"\n=== Starting game: {num_agents} agents ({num_werewolves} werewolf)"
                      f", wolf={wolf_arch}, villagers={villager_team}, half={args.half}, run {run_idx} ===")
                print("Assigned Roles (with archetypes):")
                for agent in game.agents:
                    print(f"{agent}  —  archetype: {agent.archetype}")

                # Optional peaceful Day 1 preface (unchanged)
                print("\n========================")
                print(" DAY 1 (Peaceful)")
                print("========================")
                print("No one was eliminated. The village enjoys a peaceful first day.")

                # --- Game loop (unchanged) ---
                while True:
                    print("\n========================")
                    print(f" NIGHT {game.day_count + 1}")
                    print("========================")

                    victim_id = game.night_phase()
                    if victim_id is not None:
                        print(f"Agent{victim_id} was killed during the night.")
                    else:
                        print("No one was killed during the night.")

                    time.sleep(1)

                    print("\n========================")
                    print(f" DAY {game.day_count + 1}")
                    print("========================")
                    eliminated_id, vote_record = game.day_phase()

                    if eliminated_id is not None:
                        print(f"Agent{eliminated_id} was voted out.")
                    else:
                        print("No one was eliminated (tie or no majority, even after runoff).")

                    print("\n--- Final Round Vote Record ---")
                    for voter, votee in vote_record.items():
                        print(f"Agent{voter} voted for: {votee}")

                    time.sleep(1)

                    winner = game.check_win_condition()
                    if winner:
                        print(f"\n Game Over! {winner.capitalize()} win!")
                        break

                print(f"\n=== Finished run {run_idx} for villager={villager_team} vs wolf={wolf_arch} ===\n")
                time.sleep(1)


if __name__ == "__main__":
    main()
