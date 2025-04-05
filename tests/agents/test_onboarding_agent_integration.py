# tests/agents/test_onboarding_agent_integration.py
import os

import pytest
from dotenv import load_dotenv

# Removed local Pydantic import, will import from src
# from pydantic import BaseModel, Field
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider  # Correct import path used here

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

# --- Integration Test ---


@pytest.mark.asyncio
@pytest.mark.skipif(
    not OPENROUTER_API_KEY, reason="OPENROUTER_API_KEY not set in environment"
)
async def test_onboarding_agent_produces_structured_data():
    """Verify the onboarding agent returns the expected OnboardingData structure."""

    # Instantiate the actual agent using the function from src
    agent = create_onboarding_agent(openrouter_model)

    test_prompt = "Hi! I know basic Python functions and loops, but I want to \
        learn how to build APIs with FastAPI. I prefer learning by seeing code \
            examples first."

    # Run the *actual* agent
    result = await agent.run(test_prompt)

    # --- Assertions ---
    # (Assertions remain the same, but now test the result from the actual agent)
    # 1. Check if the result data is the correct type
    assert isinstance(result.data, OnboardingData), (
        f"Expected result.data to be OnboardingData, got {type(result.data)}"
    )

    # 2. Check if the fields are populated (non-empty strings)
    assert result.data.point_a, "Point A should not be empty"
    assert isinstance(result.data.point_a, str), "Point A should be a string"

    assert result.data.point_b, "Point B should not be empty"
    assert isinstance(result.data.point_b, str), "Point B should be a string"

    assert result.data.preferences, "Preferences should not be empty"
    assert isinstance(result.data.preferences, str), "Preferences should be a string"

    # Optional: Print for verification during test development
    print("\n--- Test Result ---")
    print(f"Prompt: {test_prompt}")
    print(f"Result Data: {result.data}")
    print(f"Usage: {result.usage()}")
    print("-------------------")
