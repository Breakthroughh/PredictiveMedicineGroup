# gemini_wrapper.py

import os
import time
import re
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash-lite-preview-06-17") 

def create_agent_chat(goal, agent_id, role):
    system_message = (
        f"You are Agent {agent_id}, a {role} in a Werewolf game.\n"
        f"Your goal: {goal}\n"
        "You will receive context and prompts throughout the game."
    )
    return model.start_chat(history=[{"role": "user", "parts": [system_message]}])

def gemini_respond(chat, agent_id, role, memory, prompt, delay=1) -> str:
    """Send a prompt to Gemini chat with a small delay to avoid rate limiting."""
    time.sleep(delay)
    system_prompt = (
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
            # Non-rate-limit errors: return as an error message
            return f"[ERROR] Agent {agent_id} failed: {err_str}"

def extract_vote(response: str):
    """Parses the last line of the LLM response to extract the vote."""
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

# --- New helpers for explicit vote+reason and inter-agent ratings ----------------

def gemini_vote_with_reason(chat, agent_id, role, memory, header_prompt, delay=1) -> str:
    """
    Ask for a concise justification PLUS a vote line.
    Returns the raw text (we reuse extract_vote to parse the vote).
    """
    time.sleep(delay)
    system_prompt = (
        f"Here is the context of what has happened so far:\n{memory}\n\n"
        "TASK: Provide a short justification for who should be eliminated right now, "
        "then end with exactly one vote line.\n"
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
            return f"[ERROR] Agent {agent_id} failed: {err_str}"

def gemini_rate_agents(chat, agent_id, role, memory, alive_agent_ids, delay=1):
    """
    Ask an agent to rate all other alive agents on 4 dimensions in [-10,10].
    Request STRICT JSON. Returns a dict: { 'AgentK': {'truthfulness':int,'trustworthiness':int,'influence':int,'suspicion':int}, ... }
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
            response = chat.send_message(prompt).text.strip()
            # First try strict JSON parse
            try:
                obj = json.loads(response)
                return obj
            except Exception:
                # Fallback: permissive line parser like "Agent3: T=1, W=2, I=-3, S=4"
                parsed = {}
                for line in response.splitlines():
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
                        if not mo: return 0
                        for g in mo.groups():
                            if g is not None: return int(g)
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
            return {}
