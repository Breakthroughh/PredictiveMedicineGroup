# gemini_wrapper.py

import os
import time
import re
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
