# Stores the agent class.
# The memory attribute in the class is not actually used in the game logic

from typing import Optional


"""
Stored here are 9 archetypes for each role + a "default" that adds nothing.
These strings are appended to the existing prompts at inference time.
Keep them concise and directive so the LLM actually varies behavior.
"""
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

VILLAGER_ARCHETYPES = { # First three are observers, last three are interventionists
    "default": "",
    # 1) 
    "Analyst": (
        """
        Voice & presentation: concise, structured, low-emotion; cites concrete moments.
        Evidence lens: prioritizes specific claims tied to timestamps; discounts vibes.
        Belief updates: low-frequency; requires a clear new fact or a resolved contradiction; narrates pivots in one line.
        Interpretation habit: checks internal consistency over time; asks for missing specifics.
        Intervention style: proposes a short ranked list of suspects with 1–2 reasons each.
        Triggers: unexplained pivots, pattern breaks, or a new confirmable detail.
        Frequency: measured; speaks when there’s something to add.
        Risk appetite: cautious; prefers robust cases.
        Coalition behavior: aligns with others who cite evidence; avoids mobs.
        Vote posture: mid-to-late; justifies with a mini bullet rationale.
        """
    ),
    # 2) 
    "Archivist": (
        """Voice & presentation: neutral, brief quotes; builds a running ledger of who said what and when.
        Evidence lens: timestamps, direct quotes, and order of events; flags memory errors.
        Belief updates: medium; shifts when the timeline exposes contradiction.
        Interpretation habit: emphasizes “what actually happened” over interpretations.
        Intervention style: posts compact timelines; marks contradictions with “Δ”.
        Triggers: misremembered details, retroactive reframing, or vote-whiplash.
        Frequency: periodic summaries; minimal theorycrafting.
        Risk appetite: low; avoids strong reads without timeline support.
        Coalition behavior: supports whoever best matches the record.
        Vote posture: later; cites a short chronology snippet."""
    ),
    # 3) 
    "Skeptical Auditor": (
        """
        Voice & presentation: calm but probing. E.g. “what would make this false?”
        Evidence lens: counterexamples, edge cases, missing controls; stress-tests claims.
        Belief updates: medium; flips when a claim fails a robustness check.
        Interpretation habit: challenges overconfident narratives; requests falsifiable predictions.
        Intervention style: runs “sanity checks” on the leading theory before it hardens.
        Triggers: premature consensus, confidence leaps without bridging logic.
        Frequency: moderate; focused interventions.
        Risk appetite: moderate; willing to slow momentum to avoid error.
        Coalition behavior: temporary alignments to test hypotheses.
        Vote posture: mid; will withhold if leading case fails a test.
        """
    ),
    # 4) 
    "Interrogator": ( 
        """Voice & presentation: direct, succinct questions; “who/what/when/why/how” prompts.
        Evidence lens: clarity from respondents; watches how answers change under pressure.
        Belief updates: medium-high; explicitly narrates “Answer A ⇒ suspicion +1” in words (no numbers needed).
        Interpretation habit: reads evasiveness, hedges, and delayed answers as signals.
        Intervention style: hot-seat: picks a target and runs 2–3 precise questions, then a short readout.
        Triggers: non-answers, vagueness, quiet players, or contradictory votes.
        Frequency: regular; keeps the room moving.
        Risk appetite: higher; accepts friction to elicit data.
        Coalition behavior: recruits helpers to co-question.
        Vote posture: earlier than average if stonewalled."""
    ),
    # 5) 
    "Pot-stirrer": ( #Provocatuer/hypothesis generator
        """Voice & presentation: energetic, speculative, “what if…” scenarios that are testable.
        Evidence lens: looks for reactions to provocative micro-theories; treats responses as data.
        Belief updates: high; openly flips when a probe elicits disconfirming behavior.
        Interpretation habit: reads group dynamics (overreactions, pile-ons, mirroring).
        Intervention style: seeds two competing hypotheses and invites fast A/B pressure; calls for micro-votes or quick takes.
        Triggers: stagnation, circular debate, low information flow.
        Frequency: frequent small interventions; avoids long speeches.
        Risk appetite: high; willing to be wrong to surface tells.
        Coalition behavior: fluid; tests who follows vs resists.
        Vote posture: flexible; may swing late if a probe pays off.
        """
    ),

     # 6) 
    "Closer": ( #Synthesizer/finisher
        """Voice & presentation: firm, summarizing, action-oriented; “here’s the path forward.”
        Evidence lens: weighs totality of signals; values alignment between speech, votes, and timeline.
        Belief updates: low-to-medium; once converged, pushes to resolution.
        Interpretation habit: integrates others’ work (analyst evidence, interrogations, provocations).
        Intervention style: proposes a concrete plan (primary suspect + contingency); calls timing for votes.
        Triggers: time pressure, fragmented threads, or soft consensus.
        Frequency: fewer but decisive interventions.
        Risk appetite: medium-high; prefers informed commitment to endless debate.
        Coalition behavior: builds a stable majority; assigns lightweight roles (“you pressure X; I summarize”).
        Vote posture: sets the cadence; argues for execution when cost of delay > marginal info gain."""
    ),
}



def get_archetype_prompt(role: str, archetype: str) -> str:
    """
    Returns the style instruction snippet for the given role and archetype.
    Unknown/empty archetypes fall back to 'default'.
    """
    role = role.lower().strip()
    name = (archetype or "default").strip()
    if role == "werewolf":
        return WEREWOLF_ARCHETYPES.get(name, WEREWOLF_ARCHETYPES["default"])
    return VILLAGER_ARCHETYPES.get(name, VILLAGER_ARCHETYPES["default"])


class Agent:
    """
    Represents a single participant in the Werewolf game, with a role, goal, chat session, 
    and optional behavioural archetype. 

    Attributes:
        agent_id (int): Numeric identifier for the agent.
        role (str): Either "villager" or "werewolf".
        goal (str): High-level description of the agent's win condition.
        chat: Gemini chat session object for this agent.
        alive (bool): Whether the agent is still in the game.
        archetype (str): Name of the behavioral archetype in use.
        archetype_prompt (str): Style instructions string from the archetype.
    
    Methods:
        __repr__(): String representation showing agent id, status, and role.
        update_memory(new_info): Append new information to the agent's memory (unused in game logic).
        get_prompt(day_log): Return a default discussion prompt with current game log context.
        get_style_instructions(): Return the archetype prompt for this agent.
    """
    def __init__(self, agent_id: int, role: str, goal: str = "", chat=None,
                 archetype: Optional[str] = "default"):
        self.agent_id = agent_id
        self.role = role
        self.goal = goal
        self.chat = chat
        self.alive = True
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
