# Stores the agent class.
# The memory attribute in the class is not actually used in the game logic

from typing import Optional

# ---- Archetype prompt libraries ------------------------------------------------
# We expose 9 archetypes for each role + a "default" that adds nothing.
# These strings are appended to the existing prompts at inference time.
# Keep them concise and directive so the LLM actually varies behavior.

WEREWOLF_ARCHETYPES = {
    "default": "",
    "Silent Hunter": (
        "STYLE (Werewolf — Silent Hunter): Keep a low profile. Speak briefly and sparingly. "
        "Avoid initiating accusations. Deflect attention with mild, reasonable doubts. "
        "Never reveal uncertainty about your own alignment."
    ),
    "Chaos Agent": (
        "STYLE (Werewolf — Chaos Agent): Change suspicions frequently. Introduce conflicting theories. "
        "Undermine consensus and keep villagers divided. Avoid letting any single narrative dominate."
    ),
    "Manipulator": (
        "STYLE (Werewolf — Manipulator): Find a villager's theory and amplify it with confident, specific details. "
        "Redirect it to protect werewolves or frame innocents. Sound methodical and certain."
    ),
    "Accuser": (
        "STYLE (Werewolf — Accuser): Push a strong accusation each round with confidence. "
        "Minimize hedging. Drive the discussion toward swift votes before doubts spread to you."
    ),
    "Sympathizer": (
        "STYLE (Werewolf — Sympathizer): Agree with others often to blend in. "
        "Echo popular reasoning and only add small, supportive points. Vote to match momentum that helps you."
    ),
    "Deflector": (
        "STYLE (Werewolf — Deflector): If accused, immediately counter-accuse or shift suspicion with specific reasons. "
        "Project confidence; never sound defensive or uncertain."
    ),
    "Opportunist": (
        "STYLE (Werewolf — Opportunist): Avoid starting theories. Watch the room, then join the most credible accusation. "
        "Your goal is to seem reasonable and non-initiating while steering outcomes subtly."
    ),
    "Gaslighter": (
        "STYLE (Werewolf — Gaslighter): Recast past statements and events to create doubt about others' memories. "
        "Confidently assert misremembered details that make innocents look suspicious."
    ),
    "Fake Detective": (
        "STYLE (Werewolf — Fake Detective): Claim pattern-based deductions. Cite ‘behavioral tells’ or voting patterns. "
        "Fabricate plausible links to justify your chosen target convincingly."
    ),
}

VILLAGER_ARCHETYPES = {
    "default": "",
    "Paranoid Watcher": (
        "STYLE (Villager — Paranoid Watcher): Maintain multiple suspects and update them often. "
        "React strongly to small inconsistencies. Change targets when new red flags appear."
    ),
    "Loyal Follower": (
        "STYLE (Villager — Loyal Follower): Choose one trusted player early. Support their theories consistently. "
        "Prefer alignment over originality unless strong evidence contradicts your leader."
    ),
    "Aggressive Accuser": (
        "STYLE (Villager — Aggressive Accuser): Pick a suspect quickly each round and prosecute the case hard. "
        "Push for decisive votes rather than prolonged discussion."
    ),
    "Cautious Investigator": (
        "STYLE (Villager — Cautious Investigator): Ask clarifying questions. Require evidence before voting. "
        "Prefer consensus and avoid rash eliminations."
    ),
    "Silent Observer": (
        "STYLE (Villager — Silent Observer): Speak minimally. Weigh the room, then vote with prudent justification. "
        "Avoid leading; focus on conservative, low-risk choices."
    ),
    "Pattern Matcher": (
        "STYLE (Villager — Pattern Matcher): Track speech and vote histories. "
        "Cite concrete inconsistencies and repeated behaviors as primary evidence."
    ),
    "Contrarian": (
        "STYLE (Villager — Contrarian): Resist bandwagons. If unconvinced, argue against the majority and propose alternatives. "
        "Prioritize preventing easy manipulation."
    ),
    "Empath": (
        "STYLE (Villager — Empath): Attend to tone, hesitation, and confidence shifts. "
        "Justify suspicions using emotional cues and conversational dynamics."
    ),
    "Detective Storyteller": (
        "STYLE (Villager — Detective Storyteller): Construct a coherent narrative of events and motives. "
        "Use the story to persuade others toward your chosen suspect."
    ),
}


def get_archetype_prompt(role: str, archetype: str) -> str:
    """
    Returns the style instruction snippet for the given role and archetype.
    Unknown archetypes fall back to 'default'.
    """
    role = role.lower().strip()
    name = (archetype or "default").strip()
    if role == "werewolf":
        return WEREWOLF_ARCHETYPES.get(name, WEREWOLF_ARCHETYPES["default"])
    return VILLAGER_ARCHETYPES.get(name, VILLAGER_ARCHETYPES["default"])


class Agent:
    def __init__(self, agent_id: int, role: str, goal: str = "", chat=None,
                 archetype: Optional[str] = "default"):
        self.agent_id = agent_id
        self.role = role  # 'villager' or 'werewolf'
        self.goal = goal
        self.chat = chat
        self.alive = True

        # --- Personality / Archetype ---
        # Archetype names correspond to keys in the *ARCHETYPES dicts above.
        # 'default' adds no extra instructions beyond your existing prompts.
        self.archetype = archetype or "default"
        self.archetype_prompt = get_archetype_prompt(self.role, self.archetype)

    def __repr__(self):
        status = "Alive" if self.alive else "Dead"
        return f"Agent{self.agent_id} ({status}) [{self.role}]"

    def update_memory(self, new_info: str):
        self.memory += f"\n{new_info}"

    def get_prompt(self, day_log: str) -> str:
        return (
            f"Today, the group is discussing who might be a werewolf. "
            f"Here is what has happened so far:\n{day_log}\n\n"
            f"What is your opinion?"
        )

    # Convenience if you want to inspect or re-attach the style string elsewhere
    def get_style_instructions(self) -> str:
        """Returns the archetype prompt to be appended to task prompts."""
        return self.archetype_prompt or ""
