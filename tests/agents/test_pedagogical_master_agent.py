# tests/agents/test_pedagogical_master_agent.py
import os

import pytest
from dotenv import load_dotenv
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# Import the actual implementation
from src.agents.onboarding_agent import OnboardingData  # Need this for input
from src.agents.pedagogical_master_agent import (
    PedagogicalGuidelines,
    create_pedagogical_master_agent,
)

# Load environment variables
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
TEST_MODEL_NAME = "google/gemini-flash-1.5"

# --- Test Setup (Model Configuration) ---
openrouter_model = OpenAIModel(
    TEST_MODEL_NAME,
    provider=OpenAIProvider(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    ),
)

# --- Integration Test ---


@pytest.mark.asyncio
@pytest.mark.skipif(
    not OPENROUTER_API_KEY, reason="OPENROUTER_API_KEY not set in environment"
)
async def test_pedagogical_master_agent_produces_guidelines():
    """Verify the PMA generates structured guidelines from OnboardingData."""

    # 1. Create sample input data
    sample_onboarding_data = OnboardingData(
        point_a="Knows basic Python syntax and data types.",
        point_b="Wants to learn FastAPI for web APIs.",
        preferences="Prefers learning by building small projects and seeing \
            code examples.",
    )

    # 2. Instantiate the actual agent
    agent = create_pedagogical_master_agent(openrouter_model)

    # 3. Construct the input prompt for the agent
    # Note: The agent's system prompt guides it to analyze this profile.
    input_prompt = (
        f"Determine pedagogical guidelines based on the following student \
            profile:\n"
        f"Current Knowledge (Point A): {sample_onboarding_data.point_a}\n"
        f"Learning Goal (Point B): {sample_onboarding_data.point_b}\n"
        f"Learning Preferences: {sample_onboarding_data.preferences}"
    )

    # 4. Run the agent
    print(f"\nRunning PMA with prompt:\n{input_prompt}")
    result = await agent.run(input_prompt)
    print(f"Agent Raw Result Data: {result.data}")  # Log raw result

    # 5. --- Assertions ---
    assert isinstance(result.data, PedagogicalGuidelines), (
        f"Expected result.data to be PedagogicalGuidelines, got {type(result.data)}"
    )

    assert result.data.guideline, "Guideline should not be empty"
    assert isinstance(result.data.guideline, str), "Guideline should be a string"

    # Optional: More specific checks on guideline content could be added,
    # but might be brittle due to LLM variability.
    # e.g., assert "project" in result.data.guideline.lower()

    print("\n--- Test Result --- ")
    print(f"Guideline: {result.data.guideline}")
    print(f"Usage: {result.usage()}")
    print("-------------------")
