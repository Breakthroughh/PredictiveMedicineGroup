#Stores the agent class.
#The memory attribute in the class is not actually used in the game logic

class Agent:
    def __init__(self, agent_id: int, role: str, goal: str = "", chat=None):
        self.agent_id = agent_id
        self.role = role  # 'villager' or 'werewolf'
        self.goal = goal
        self.chat = chat
        self.alive = True

    def __repr__(self):
        status = "Alive" if self.alive else "Dead"
        return f"Agent{self.agent_id} ({status}) [{self.role}]"


    def update_memory(self, new_info: str):
        self.memory += f"\n{new_info}"

    def get_prompt(self, day_log: str) -> str:
        return f"Today, the group is discussing who might be a werewolf. Here is what has happened so far:\n{day_log}\n\nWhat is your opinion?"
