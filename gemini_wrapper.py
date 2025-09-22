# gemini_wrapper.py
# Essentially provides the gemini related operations for the game
"""
Thin wrapper around google.generativeai for the Werewolf experiments.
Provides:
- chat session creation per agent,
- prompting helpers with simple rate-limit retries,
- vote parsing,
- inter-agent rating collection (T/Tw/I only),
- Day-1 accuse-left utterances,
- rubric-driven suspicion update with per-target audit lines (no extra prose rationale) for villagers,
- freeform suspicion scoring for werewolves (no rubric).
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
    """If the model returns fenced code (```json ... ```), remove the fences."""
    s = s.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, flags=re.S | re.I)
    return m.group(1).strip() if m else s


def create_agent_chat(goal, agent_id, role, discussion_rounds: int | None = None):
    """
    Create and return a fresh Gemini chat session for one agent.

    Briefly explain the rubric-updates + softmax voting rule to align agent expectations.
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
        "- When asked to vote, end with exactly one line: 'VOTE: AgentX' or 'VOTE: NOONE'.\n\n"
        "Decision protocol:\n"
        "- Villagers: Suspicion is updated from a RULE RUBRIC over public events.\n"
        "- Werewolves: Suspicion is your own free assessment in [-10,10] (no rubric).\n"
        "- Villager voting uses a shifted-softmax over alive opponents: p(j) ∝ exp((s_j − τ)/T). τ acts as a bias.\n"
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
    Day discussions don't require a vote line; night/explicit vote prompts do.
    """
    time.sleep(delay)
    system_prompt = (
        f"{_rules_reminder_short()}"
        f"Here is the context of what has happened so far:\n{memory}\n\n"
        f"PROMPT:\n{prompt}\n"
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
                print(f"[gemini_wrapper] Rate limit hit for Agent{agent_id}; waiting {wait}s before retry.")
                time.sleep(wait)
                continue
            print(f"[gemini_wrapper] Non-429 error for Agent{agent_id}: {err_str}")
            return f"[ERROR] Agent {agent_id} failed: {err_str}"


def extract_vote(response: str):
    """Extract 'AgentN' or None from a trailing 'VOTE:' line."""
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


def gemini_vote_with_reason(chat, agent_id, role, memory, header_prompt, delay=1) -> str:
    """
    Ask for a short justification plus a single vote line.
    Used at NIGHT and for werewolves at FINAL DAY VOTE.
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


# ---------------- Inter-agent ratings (T/Tw/I only) ----------------

def gemini_rate_agents(chat, agent_id, role, memory, alive_agent_ids, delay=1):
    """
    Ask an agent to rate all other *alive* agents on THREE scales in [-10, 10]:
        1) truthfulness
        2) trustworthiness
        3) influence

    NOTE: Suspicion is NOT requested here. Suspicion is updated separately via the rubric (villagers)
          or freeform (werewolves).
    Returns dict like:
        {
          "AgentK": {"truthfulness": int, "trustworthiness": int, "influence": int},
          ...
        }
    """
    time.sleep(delay)
    others = [f"Agent{x}" for x in alive_agent_ids if x != agent_id]
    targets_line = ", ".join(others) if others else "(no others)"
    prompt = (
        f"Here is the context so far:\n{memory}\n\n"
        "TASK: Rate each of the OTHER alive agents on three scales from -10 (worst) to 10 (best):\n"
        "1) truthfulness {how accurate you think their statements are}\n"
        "2) trustworthiness {how much weight you would give their statements}\n"
        "3) influence {how much you think the group values their statements}\n\n"
        f"OTHER AGENTS TO RATE: {targets_line}\n\n"
        "OUTPUT STRICT JSON ONLY with this schema (no prose):\n"
        "{\n"
        '  "AgentK": {"truthfulness": int, "trustworthiness": int, "influence": int},\n'
        '  "AgentM": {"truthfulness": int, "trustworthiness": int, "influence": int}\n'
        "}\n"
        "All integers must be between -10 and 10. Do not include yourself."
    )
    while True:
        try:
            raw = chat.send_message(prompt).text.strip()
            raw = _strip_code_fences(raw)
            try:
                obj = json.loads(raw)
                if not isinstance(obj, dict):
                    raise ValueError("Non-dict JSON for ratings")
                return obj
            except Exception:
                # Fallback permissive line parser like: "Agent3: T=1, W=2, I=-3"
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
                    def pick(mo):
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
            print(f"[gemini_wrapper] Non-429 error (rate_agents) Agent{agent_id}: {err_str}")
            return {"__ERROR__": err_str}


# --------------- Villager rubric-driven suspicion update ----------------

def gemini_update_suspicion(
    chat,
    agent_id: int,
    role: str,
    memory: str,
    rubric_text: str,
    alive_agent_ids: list[int],
    previous_suspicion: dict[str, float] | None = None,
    delay: int = 1
) -> dict:
    """
    Hardened for JSON correctness:
      1) Ask in JSON mode.
      2) If parse fails, ask once to REFORMAT to strict JSON.
      3) Return ok/error flags so callers can log status.
    """
    time.sleep(delay)
    others = [f"Agent{x}" for x in alive_agent_ids if x != agent_id]
    targets_line = ", ".join(others) if others else "(no others)"
    prev_json = json.dumps(previous_suspicion or {}, ensure_ascii=False)

    prompt = (
        f"{_rules_reminder_short()}"
        f"Context (public information only):\n{memory}\n\n"
        "TASK: Apply the following RULES RUBRIC to the LATEST public discussion/events only and UPDATE your suspicion "
        "scores for each OTHER alive agent. Start from your previous suspicion values (already decayed externally) "
        "and add small adjustments according to the rubric. Finally, clamp all values to the range [-10, 10]. "
        "Keep numbers mild in a 5-player game.\n\n"
        f"{rubric_text}\n\n"
        "Clarification: Day-1 'seed' accusations ARE valid accusations for Rule 1 (they COUNT), "
        "but they do not grant any extra credit/penalty beyond Rule 1.\n\n"
        f"OTHER ALIVE AGENTS TO SCORE: {targets_line}\n"
        f"YOUR PREVIOUS SUSPICION (JSON): {prev_json}\n\n"
        "OUTPUT STRICT JSON ONLY with this schema (no prose outside the JSON):\n"
        "{\n"
        '  "suspicion": {\n'
        '    "AgentK": float,\n'
        '    "AgentM": float\n'
        "  },\n"
        '  "rationale": "2–3 sentences explaining which rules you applied and which events triggered them.",\n'
        '  "rationale_lines": [\n'
        '     "Suspicion score AgentK: PREV -> NEW (because of rule <#>: <short reason>).",\n'
        '     "Suspicion score AgentM: PREV -> NEW (because of rule <#>: <short reason>)."\n'
        "  ]\n"
        "}\n"
        "Formatting rules:\n"
        "- Output ONLY valid JSON (no markdown fences, no comments, no extra text).\n"
        "- Use double quotes for all strings.\n"
        "- Provide numeric values (floats) with at most 1 decimal place.\n"
        "- Include an entry in 'rationale_lines' for each OTHER alive agent.\n"
    )

    def _try_parse(raw_str: str):
        raw_str = _strip_code_fences(raw_str).strip()
        obj = json.loads(raw_str)
        if not isinstance(obj, dict):
            raise ValueError("Suspicion update: non-dict JSON")
        if not isinstance(obj.get("suspicion"), dict):
            obj["suspicion"] = {}
        if not isinstance(obj.get("rationale_lines"), list):
            obj["rationale_lines"] = []
        obj["ok"] = True
        return obj

    genconf = {
        "response_mime_type": "application/json",
        "temperature": 0.2
    }

    # --- Attempt 1: ask in JSON mode directly ---
    try:
        resp = chat.send_message(prompt, generation_config=genconf)
        raw = (resp.text or "").strip()
        return _try_parse(raw)
    except Exception as e1:
        err1 = str(e1)

    # Handle rate limit separately
    if "429" in err1 or "quota" in err1.lower():
        m = re.search(r'retry_delay\s*[:{]\s*seconds\s*[:=]?\s*(\d+)', err1)
        wait = int(m.group(1)) if m else 60
        print(f"[gemini_wrapper] Rate limit (update_suspicion A) Agent{agent_id}; waiting {wait}s.")
        time.sleep(wait)
        try:
            resp = chat.send_message(prompt, generation_config=genconf)
            raw = (resp.text or "").strip()
            return _try_parse(raw)
        except Exception as e1b:
            err1 = str(e1b)

    # --- Attempt 2: ask the model to REFORMAT prior content to strict JSON ---
    repair_prompt = (
        "Your previous reply was not valid JSON. Reformat it into STRICT JSON ONLY, "
        "matching the exact schema previously specified. No prose, no code fences."
    )
    try:
        resp2 = chat.send_message(repair_prompt, generation_config=genconf)
        raw2 = (resp2.text or "").strip()
        return _try_parse(raw2)
    except Exception as e2:
        err2 = str(e2)
        if "429" in err2 or "quota" in err2.lower():
            m = re.search(r'retry_delay\s*[:{]\s*seconds\s*[:=]?\s*(\d+)', err2)
            wait = int(m.group(1)) if m else 60
            print(f"[gemini_wrapper] Rate limit (update_suspicion B) Agent{agent_id}; waiting {wait}s.")
            time.sleep(wait)
            try:
                resp2 = chat.send_message(repair_prompt, generation_config=genconf)
                raw2 = (resp2.text or "").strip()
                return _try_parse(raw2)
            except Exception as e2b:
                err2 = str(e2b)

    # --- Final fallback (keep previous suspicion); mark error for logs ---
    print(f"[gemini_wrapper] JSON repair failed (update_suspicion) Agent{agent_id}: {err1} | {err2}")
    return {
        "suspicion": previous_suspicion or {},
        "rationale": "[ERROR] Using previous suspicion due to update failure.",
        "rationale_lines": [],
        "ok": False,
        "error": f"A: {err1} | B: {err2}"
    }


# --------------- Werewolf freeform suspicion (no rubric) ----------------

def gemini_score_suspicion_freeform(
    chat,
    agent_id: int,
    role: str,
    memory: str,
    alive_agent_ids: list[int],
    previous_suspicion: dict[str, float] | None = None,
    delay: int = 1
) -> dict:
    """
    Werewolves assess suspicion freeform in [-10, 10], based on public info.
    No rules rubric is applied. Keep mild magnitudes in 5-player settings.
    """
    time.sleep(delay)
    others = [f"Agent{x}" for x in alive_agent_ids if x != agent_id]
    targets_line = ", ".join(others) if others else "(no others)"
    prev_json = json.dumps(previous_suspicion or {}, ensure_ascii=False)

    prompt = (
        f"{_rules_reminder_short()}"
        f"Context (public information only):\n{memory}\n\n"
        "TASK: Update (or set) your suspicion scores in [-10, 10] for each OTHER alive agent based on your own free assessment. "
        "Consider consistency, timing, and strategic behavior, but DO NOT use any explicit external rules rubric. "
        "Start from your previous values (already decayed externally) and make mild adjustments. Clamp to [-10,10].\n\n"
        f"OTHER ALIVE AGENTS TO SCORE: {targets_line}\n"
        f"YOUR PREVIOUS SUSPICION (JSON): {prev_json}\n\n"
        "OUTPUT STRICT JSON ONLY with this schema (no prose outside the JSON):\n"
        "{\n"
        '  "suspicion": {\n'
        '    "AgentK": float,\n'
        '    "AgentM": float\n'
        "  },\n"
        '  "rationale_lines": [\n'
        '     "Suspicion score AgentK: PREV -> NEW (short reason).",\n'
        '     "Suspicion score AgentM: PREV -> NEW (short reason)."\n'
        "  ]\n"
        "}\n"
        "Formatting rules:\n"
        "- Output ONLY valid JSON (no markdown fences, no comments, no extra text).\n"
        "- Use double quotes for all strings.\n"
        "- Provide numeric values (floats) with at most 1 decimal place.\n"
        "- Include an entry in 'rationale_lines' for each OTHER alive agent.\n"
    )

    def _try_parse(raw_str: str):
        raw_str = _strip_code_fences(raw_str).strip()
        obj = json.loads(raw_str)
        if not isinstance(obj, dict):
            raise ValueError("Freeform suspicion: non-dict JSON")
        if not isinstance(obj.get("suspicion"), dict):
            obj["suspicion"] = {}
        if not isinstance(obj.get("rationale_lines"), list):
            obj["rationale_lines"] = []
        obj["ok"] = True
        return obj

    genconf = {
        "response_mime_type": "application/json",
        "temperature": 0.3
    }

    try:
        resp = chat.send_message(prompt, generation_config=genconf)
        raw = (resp.text or "").strip()
        return _try_parse(raw)
    except Exception as e1:
        err1 = str(e1)

    if "429" in err1 or "quota" in err1.lower():
        m = re.search(r'retry_delay\s*[:{]\s*seconds\s*[:=]?\s*(\d+)', err1)
        wait = int(m.group(1)) if m else 60
        print(f"[gemini_wrapper] Rate limit (freeform A) Agent{agent_id}; waiting {wait}s.")
        time.sleep(wait)
        try:
            resp = chat.send_message(prompt, generation_config=genconf)
            raw = (resp.text or "").strip()
            return _try_parse(raw)
        except Exception as e1b:
            err1 = str(e1b)

    # Try a single repair pass
    repair_prompt = (
        "Your previous reply was not valid JSON. Reformat it into STRICT JSON ONLY, "
        "matching the exact schema previously specified. No prose, no code fences."
    )
    try:
        resp2 = chat.send_message(repair_prompt, generation_config=genconf)
        raw2 = (resp2.text or "").strip()
        return _try_parse(raw2)
    except Exception as e2:
        err2 = str(e2)

    print(f"[gemini_wrapper] JSON repair failed (freeform suspicion) Agent{agent_id}: {err1} | {err2}")
    return {
        "suspicion": previous_suspicion or {},
        "rationale_lines": [],
        "ok": False,
        "error": f"A: {err1} | B: {err2}"
    }


# --- One-line Day-1 accuse-left utterance ---

def gemini_accuse_left(chat, agent_id, memory, target_name: str, style: str | None = None, delay=1) -> str:
    """
    Produce a single, provocative accusation aimed at the 'left' neighbor (target_name).
    One sentence only, no vote, no role reveal; used to stir initial noise on Day 1.
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
