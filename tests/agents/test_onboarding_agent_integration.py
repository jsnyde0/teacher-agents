# tests/agents/test_onboarding_agent_integration.py
import os
from typing import List

import pytest
from dotenv import load_dotenv
from pydantic_ai.messages import ModelMessage  # Import for history type hint

# Removed local Pydantic import, will import from src
# from pydantic import BaseModel, Field
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# Import the actual implementation
from src.agents.onboarding_agent import OnboardingData, create_onboarding_agent

# Removed local OnboardingData definition
# class OnboardingData(BaseModel):
#     ...

# Load environment variables (where OPENROUTER_API_KEY should be)
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Using a capable model on OpenRouter for testing
TEST_MODEL_NAME = "google/gemini-2.0-flash-lite-001"  # Updated model ID

# --- Test Setup (Model Configuration) ---

# Configure OpenRouter model - this remains the same
# Reference: https://ai.pydantic.dev/models/#openrouter
openrouter_model = OpenAIModel(
    TEST_MODEL_NAME,
    provider=OpenAIProvider(  # Correct import path used here
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    ),
)

# Removed the placeholder agent definition
# onboarding_agent_placeholder = Agent(...)

# --- Integration Test for Conversational Onboarding ---


@pytest.mark.asyncio
@pytest.mark.skipif(
    not OPENROUTER_API_KEY, reason="OPENROUTER_API_KEY not set in environment"
)
async def test_onboarding_agent_conversational_flow():
    """Verify the onboarding agent handles a multi-turn conversation."""

    agent = create_onboarding_agent(openrouter_model)
    history: List[ModelMessage] = []

    # --- Turn 1: User gives partial info (Goal only) ---
    prompt1 = "Hi, my learning goal is to learn FastAPI."
    print(f"\nUser Turn 1: {prompt1}")
    result1 = await agent.run(prompt1, message_history=history)
    history.extend(result1.all_messages())

    print(f"Agent Turn 1: {result1.data}")
    assert isinstance(result1.data, str), (
        "Agent should ask a clarifying question (string response) after turn 1"
    )
    question1_lower = result1.data.lower()
    assert (
        "know" in question1_lower
        or "start" in question1_lower
        or "experience" in question1_lower
        or "current" in question1_lower
    ), (
        f"Agent's first question ('{result1.data}') should ask about current \
            knowledge (Point A)"
    )

    # --- Turn 2: User provides Point A ---
    prompt2 = "I know basic Python pretty well."
    print(f"\nUser Turn 2: {prompt2}")
    result2 = await agent.run(prompt2, message_history=history)
    history.extend(result2.all_messages())

    print(f"Agent Turn 2: {result2.data}")
    assert isinstance(result2.data, str), (
        "Agent should ask a clarifying question (string response) after turn 2"
    )
    question2_lower = result2.data.lower()
    assert (
        "prefer" in question2_lower
        or "learn" in question2_lower
        or "style" in question2_lower
    ), f"Agent's second question ('{result2.data}') should ask about preferences"

    # --- Turn 3: User provides Preferences ---
    prompt3 = "I prefer learning with code examples."
    print(f"\nUser Turn 3: {prompt3}")
    result3 = await agent.run(prompt3, message_history=history)
    # Don't extend history here if we expect final output, Pydantic AI includes
    # it anyway

    print(f"Agent Turn 3: {result3.data}")
    assert isinstance(result3.data, OnboardingData), (
        "Agent should return OnboardingData after turn 3"
    )

    # --- Final Assertions on Structured Data ---
    final_data = result3.data
    assert final_data.point_a, "Final Point A should not be empty"
    assert "basic python" in final_data.point_a.lower(), (
        "Final Point A not captured correctly"
    )

    assert final_data.point_b, "Final Point B should not be empty"
    assert "fastapi" in final_data.point_b.lower(), (
        "Final Point B not captured correctly"
    )

    assert final_data.preferences, "Final Preferences should not be empty"
    assert "code examples" in final_data.preferences.lower(), (
        "Final Preferences not captured correctly"
    )

    print("\n--- Test Result --- ")
    print(f"Final Data: {final_data}")
    print(f"Final Usage: {result3.usage()}")
    print("-------------------")
