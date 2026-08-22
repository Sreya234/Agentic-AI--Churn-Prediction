from dotenv import load_dotenv
load_dotenv()

"""
CrewAI + Groq (LiteLLM) — cache_breakpoint fix
Root cause: crewai/agents/crew_agent_executor.py calls
mark_cache_breakpoint() unconditionally on every message. Only the
Anthropic adapter strips it back out — Groq's API rejects the extra key.
Apply this patch BEFORE importing/instantiating any Agent/Crew objects.
"""

import os
import copy
import logging

logger = logging.getLogger(__name__)

# --- STEP 1: stop CrewAI from injecting the flag in the first place ---
try:
    import crewai.llms.cache as _crewai_cache

    def _noop_mark_cache_breakpoint(msg):
        return msg

    _crewai_cache.mark_cache_breakpoint = _noop_mark_cache_breakpoint
except ImportError:
    logger.warning("crewai.llms.cache not found — relying on Step 2 only.")

# --- STEP 2: defense-in-depth — strip it from every LiteLLM request ---
import litellm


def _strip_cache_breakpoint(messages):
    if not messages:
        return messages
    cleaned = []
    for msg in messages:
        if isinstance(msg, dict):
            m = copy.deepcopy(msg)
            m.pop("cache_breakpoint", None)
            content = m.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        block.pop("cache_breakpoint", None)
            cleaned.append(m)
        else:
            cleaned.append(msg)
    return cleaned


_original_completion = litellm.completion
_original_acompletion = litellm.acompletion


def _patched_completion(*args, **kwargs):
    if "messages" in kwargs:
        kwargs["messages"] = _strip_cache_breakpoint(kwargs["messages"])
    elif args:
        args = list(args)
        args[0] = _strip_cache_breakpoint(args[0])
        args = tuple(args)
    return _original_completion(*args, **kwargs)


async def _patched_acompletion(*args, **kwargs):
    if "messages" in kwargs:
        kwargs["messages"] = _strip_cache_breakpoint(kwargs["messages"])
    elif args:
        args = list(args)
        args[0] = _strip_cache_breakpoint(args[0])
        args = tuple(args)
    return await _original_acompletion(*args, **kwargs)


litellm.completion = _patched_completion
litellm.acompletion = _patched_acompletion
litellm.drop_params = True

# --- Your CrewAI code ---
from crewai import Agent, Task, Crew, LLM

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY environment variable is not set.")

llm = LLM(
    model="groq/openai/gpt-oss-120b",   # <- was groq/llama-3.3-70b-versatile
    api_key=api_key,
    temperature=0.7,
)


retention_specialist = Agent(
    role="Customer Retention Specialist",
    goal="Prevent customer churn with tailored offers.",
    backstory="You are a customer success expert.",
    verbose=True,
    llm=llm,
)

task = Task(
    description=(
        "A customer named Alex has been inactive for 30 days and is at "
        "risk of churning. Draft a short, tailored retention offer for them."
    ),
    expected_output="A 3-4 sentence personalized retention message.",
    agent=retention_specialist,
)

crew = Crew(
    agents=[retention_specialist],
    tasks=[task],
    verbose=True,
)

if __name__ == "__main__":
    result = crew.kickoff()
    print("\n--- CREW RESULT ---")
    print(result)