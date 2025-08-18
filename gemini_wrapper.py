# gemini_wrapper.py
# Essentially provides the gemini related operations for the game
"""
Thin wrapper around google.generativeai for the Werewolf experiments.
Provides:
- chat session creation per agent,
- prompting helpers with simple rate-limit retries,
- vote parsing,
- inter-agent rating collection (prefers strict JSON),
- (NEW) one-line Day-1 accuse-left utterances.
"""

import os
import time
import re
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Model can be swapped here if needed.
model = genai.GenerativeModel("gemini-2.5-flash-lite-preview-06-17")


def _strip_code_fences(s: str) -> str:
    """
    If the model returns fenced code (e.g., ```json ... ```), remove the fences.
    Keeps inner content intact.
    """
    s = s.strip()
    # Match ```json ... ``` or ``` ... ```
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, flags=re.S | re.I)
    return m.group(1).strip() if m else s


def create_agent_chat(goal, agent_id, role, discussion_rounds: int | None = None):
    """
    Create and return a fresh Gemini chat session for one agent.

    Args:
        goal (str): The agent's objective text.
        agent_id (int): Numeric id used in prompts.
        role (str): "villager" or "werewolf".
        discussion_rounds (int|None): If provided, included in the Rules Primer.

    Returns:
        genai.ChatSession: Session primed with a short persona message + Rules Primer.
    """
    r_text = f"{discussion_rounds}" if discussion_rounds is not None else "multiple"
    system_message = (
        f"You are Agent {agent_id}, a {role} in a Werewolf game.\n"
        f"Your goal: {goal}\n\n"
        "RULES PRIMER (concise):\n"
        f"- Daytime has {r_text} discussion round(s); ONLY the final round's votes decide elimination.\n"
        "- A strict majority is required (floor(alive/2)+1). If no majority, no one is eliminated.\n"
        "- Night: werewolves discuss privately and choose a victim; ONLY the final night round counts.\n"
        "- Win: Villagers win when ALL werewolves are dead. Werewolves win when werewolves >= villagers.\n"
        "- Public vs private: Public discussions are visible to everyone; individual votes are logged but NOT broadcast as public content.\n"
        "- When asked to vote, end with exactly one line: 'VOTE: AgentX' or 'VOTE: NOONE'.\n"
        "You will receive additional context and prompts throughout the game."
    )
    return model.start_chat(history=[{"role": "user", "parts": [system_message]}])


def _rules_reminder_short() -> str:
    """A tiny reminder we prepend to prompts to reduce drift without bloating tokens."""
    return (
        "RULES REMINDER (short): Do not reveal your role. Only the FINAL round's votes count. "
        "Majority (floor(alive/2)+1) required; otherwise no elimination. "
        "Public discussions are logged without the final 'VOTE:' line.\n"
    )


def gemini_respond(chat, agent_id, role, memory, prompt, delay=1) -> str:
    """
    Send a discussion prompt (with memory/context) and get the agent's reply.

    Ensures the reply ends with a single 'VOTE: AgentX' or 'VOTE: NOONE' line.
    Retries on 429/quota errors with a backoff parsed from the error, default 60s.

    Args:
        chat: genai chat session from create_agent_chat.
        agent_id (int): For logging/error messages.
        role (str): Agent role (not used for logic here, but kept for parity).
        memory (str): Prior events summary string.
        prompt (str): Round-specific instructions.
        delay (int): Sleep seconds before sending to avoid rate limits.

    Returns:
        str: LLM text (stripped). On non-rate-limit error, a short '[ERROR] ...' string.
    """
    time.sleep(delay)
    system_prompt = (
        f"{_rules_reminder_short()}"
        f"Here is the context of what has happened so far:\n{memory}\n\n"
        f"PROMPT:\n{prompt}\n\n"
        "IMPORTANT: At the end of your response, state your vote for who should be eliminated today.\n"
        "Write only one final line like: 'VOTE: Agent3' or 'VOTE: NOONE'"
    )

    # Retry loop on rate limits (HTTP 429 / quota errors)
    while True:
        try:
            response = chat.send_message(system_prompt)
            return response.text.strip()
        except Exception as e:
            err_str = str(e)
            # Detect rate limit errors by status code or "quota" keyword
            if "429" in err_str or "quota" in err_str.lower():
                # Attempt to extract suggested retry_delay.seconds from the error string
                m = re.search(r'retry_delay\s*[:{]\s*seconds\s*[:=]?\s*(\d+)', err_str)
                wait = int(m.group(1)) if m else 60
                print(f"[gemini_wrapper] Rate limit hit for Agent{agent_id}; waiting {wait}s before retry.")
                time.sleep(wait)
                continue
            # Non-rate-limit errors: return as an error message (visible in logs)
            print(f"[gemini_wrapper] Non-429 error for Agent{agent_id}: {err_str}")
            return f"[ERROR] Agent {agent_id} failed: {err_str}"


def extract_vote(response: str):
    """
    Extract a vote from the last 'VOTE:' line in an LLM response.

    Args:
        response (str): LLM free-form text ending with a vote line.

    Returns:
        int | None: Agent id (e.g., 3 for 'Agent3') or None for 'NOONE' / not found.
    """
    lines = response.strip().splitlines()
    for line in reversed(lines):
        if "VOTE:" in line:
            parts = line.strip().upper().split("VOTE:")
            if len(parts) > 1:
                vote_str = parts[1].strip()
                if vote_str == "NOONE":
                    return None
                if vote_str.startswith("AGENT"):
                    try:
                        return int(vote_str.replace("AGENT", "").strip())
                    except:
                        return None
    return None


# --- Explicit vote+reason and inter-agent ratings helpers ----------------

def gemini_vote_with_reason(chat, agent_id, role, memory, header_prompt, delay=1) -> str:
    """
    Ask for a short justification plus a single vote line.

    Args:
        chat: genai chat session.
        agent_id (int): For logging/error messages.
        role (str): Agent role (unused here but kept for symmetry).
        memory (str): Context string (prior events).
        header_prompt (str): Header/directives to prepend.
        delay (int): Pre-send sleep to soften rate spikes.

    Returns:
        str: Raw LLM text containing a short justification and a 'VOTE:' line.
    """
    time.sleep(delay)
    system_prompt = (
        f"{_rules_reminder_short()}"
        f"Here is the context of what has happened so far:\n{memory}\n\n"
        "TASK: Provide a short justification for who you think should be eliminated right now, "
        "then end with exactly one vote line. Recall your role and goals. \n"
        f"{header_prompt}\n\n"
        "IMPORTANT FORMAT:\n"
        "- A short justification paragraph (1-3 sentences max)\n"
        "- Final line: 'VOTE: AgentX' or 'VOTE: NOONE'\n"
    )
    while True:
        try:
            response = chat.send_message(system_prompt)
            return response.text.strip()
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "quota" in err_str.lower():
                m = re.search(r'retry_delay\s*[:{]\s*seconds\s*[:=]?\s*(\d+)', err_str)
                wait = int(m.group(1)) if m else 60
                print(f"[gemini_wrapper] Rate limit (vote_with_reason) Agent{agent_id}; waiting {wait}s.")
                time.sleep(wait)
                continue
            print(f"[gemini_wrapper] Non-429 error (vote_with_reason) Agent{agent_id}: {err_str}")
            return f"[ERROR] Agent {agent_id} failed: {err_str}"


def gemini_rate_agents(chat, agent_id, role, memory, alive_agent_ids, delay=1):
    """
    Ask an agent to rate all other *alive* agents on four scales in [-10, 10].

    Requests strict JSON and returns a dict like:
        {
          "AgentK": {"truthfulness": int, "trustworthiness": int,
                     "influence": int, "suspicion": int},
          ...
        }
    Falls back to a permissive line parser if JSON parse fails. Retries on 429/quota.

    Args:
        chat: genai chat session.
        agent_id (int): Id of the rater (excluded from targets).
        role (str): Agent role (unused here but kept for symmetry).
        memory (str): Context string (prior events).
        alive_agent_ids (list[int]): Agent ids currently alive.
        delay (int): Pre-send sleep to soften rate spikes.

    Returns:
        dict[str, dict] | {"__ERROR__": str}: Ratings per target agent, or an error payload on non-429 failure.
    """
    time.sleep(delay)
    others = [f"Agent{x}" for x in alive_agent_ids if x != agent_id]
    targets_line = ", ".join(others) if others else "(no others)"
    prompt = (
        f"Here is the context so far:\n{memory}\n\n"
        "TASK: Rate each of the OTHER alive agents on four scales from -10 (worst) to 10 (best):\n"
        "1) truthfulness {how accurate you think their statements are}\n"
        "2) trustworthiness {how much weight you would give their statements}\n"
        "3) influence {how much you think the group values their statements}\n"
        "4) suspicion {how likely you think this agent is the werewolf}\n\n"
        f"OTHER AGENTS TO RATE: {targets_line}\n\n"
        "OUTPUT STRICT JSON ONLY with this schema (no prose):\n"
        "{\n"
        '  "AgentK": {"truthfulness": int, "trustworthiness": int, "influence": int, "suspicion": int},\n'
        '  "AgentM": {"truthfulness": int, "trustworthiness": int, "influence": int, "suspicion": int}\n'
        "}\n"
        "All integers must be between -10 and 10. Do not include yourself."
    )
    while True:
        try:
            raw = chat.send_message(prompt).text.strip()
            raw = _strip_code_fences(raw)

            # First try strict JSON parse
            try:
                obj = json.loads(raw)
                if not isinstance(obj, dict):
                    # If it isn't a dict, force fallback parsing.
                    raise ValueError("Non-dict JSON for ratings")
                return obj
            except Exception:
                # Fallback: permissive line parser like "Agent3: T=1, W=2, I=-3, S=4"
                parsed = {}
                for line in raw.splitlines():
                    m = re.match(r"^\s*(Agent\d+)\s*[:\-]\s*(.*)$", line.strip())
                    if not m:
                        continue
                    who = m.group(1)
                    rest = m.group(2)
                    t = re.search(r"truthfulness\s*=?\s*(-?\d+)|\bT\s*=?\s*(-?\d+)", rest, re.I)
                    w = re.search(r"trustworthiness\s*=?\s*(-?\d+)|\bW\s*=?\s*(-?\d+)", rest, re.I)
                    i = re.search(r"influence\s*=?\s*(-?\d+)|\bI\s*=?\s*(-?\d+)", rest, re.I)
                    s = re.search(r"suspicion\s*=?\s*(-?\d+)|\bS\s*=?\s*(-?\d+)", rest, re.I)

                    def pick(mo):
                        """Pick first present capture group as int; default 0."""
                        if not mo:
                            return 0
                        for g in mo.groups():
                            if g is not None:
                                try:
                                    return int(g)
                                except:
                                    return 0
                        return 0

                    parsed[who] = {
                        "truthfulness": max(-10, min(10, pick(t))),
                        "trustworthiness": max(-10, min(10, pick(w))),
                        "influence": max(-10, min(10, pick(i))),
                        "suspicion": max(-10, min(10, pick(s))),
                    }
                return parsed
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "quota" in err_str.lower():
                m = re.search(r'retry_delay\s*[:{]\s*seconds\s*[:=]?\s*(\d+)', err_str)
                wait = int(m.group(1)) if m else 60
                print(f"[gemini_wrapper] Rate limit (rate_agents) Agent{agent_id}; waiting {wait}s.")
                time.sleep(wait)
                continue
            # Non-429 error: surface the exact message to the caller for debugging
            print(f"[gemini_wrapper] Non-429 error (rate_agents) Agent{agent_id}: {err_str}")
            return {"__ERROR__": err_str}


# --- NEW: one-line Day-1 accuse-left utterance ---

def gemini_accuse_left(chat, agent_id, memory, target_name: str, style: str | None = None, delay=1) -> str:
    """
    Produce a single, provocative accusation aimed at the 'left' neighbor (target_name).
    One sentence only, no vote, no role reveal; used to stir initial noise on Day 1.
    Retries on 429/quota with backoff; on other errors returns a short '[seed-error] ...' message.
    """
    time.sleep(delay)
    style_block = f"\nSTYLE INSTRUCTIONS:\n{style}\n" if style else ""
    prompt = (
        f"{_rules_reminder_short()}"
        f"Context (public information so far):\n{memory}\n\n"
        f"RITUAL WARM-UP (treat as noisy): Publicly address {target_name} with a ONE-SENTENCE accusation "
        f"aimed at provoking a reaction. This is a starter spark, not evidence.\n"
        "CONSTRAINTS: one sentence, no vote line, do NOT reveal your role, no private info.\n"
        "Tone should match your style/personality: you may be playful, skeptical, or sharp."
        f"{style_block}"
    )

    while True:
        try:
            return chat.send_message(prompt).text.strip()
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "quota" in err_str.lower():
                m = re.search(r'retry_delay\s*[:{]\s*seconds\s*[:=]?\s*(\d+)', err_str)
                wait = int(m.group(1)) if m else 60
                print(f"[gemini_wrapper] Rate limit (accuse_left) Agent{agent_id}; waiting {wait}s.")
                time.sleep(wait)
                continue
            print(f"[gemini_wrapper] Non-429 error (accuse_left) Agent{agent_id}: {err_str}")
            return f"[seed-error] {err_str}"
