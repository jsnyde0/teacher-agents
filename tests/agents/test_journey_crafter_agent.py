# tests/agents/test_journey_crafter_agent.py
import os

import pytest
from dotenv import load_dotenv

# Remove explicit client import - we won't create it manually
# from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider  # Need the provider again

from src.agents.journey_crafter_agent import (
    LearningPlan,
    create_journey_crafter_agent,
)

# Import agent implementation and data models
from src.agents.onboarding_agent import OnboardingData
from src.agents.pedagogical_master_agent import PedagogicalGuidelines

# Load environment variables
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
TEST_MODEL_NAME = "google/gemini-2.0-flash-lite-001"

# --- Test Setup (Model Configuration) ---

# Remove explicit client creation
# openai_client = AsyncOpenAI(...)

# Configure the Pydantic AI model directly, passing provider args
openrouter_model = OpenAIModel(
    model_name=TEST_MODEL_NAME,  # Use model_name parameter
    provider=OpenAIProvider(  # Use the provider to specify the target
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    ),
)

# --- Integration Test ---


@pytest.mark.asyncio
@pytest.mark.skipif(
    not OPENROUTER_API_KEY, reason="OPENROUTER_API_KEY not set in environment"
)
async def test_journey_crafter_agent_produces_learning_plan():
    """Verify the JCA generates a structured learning plan (list of strings)."""

    # 1. Create sample input data
    sample_onboarding_data = OnboardingData(
        point_a="Knows basic Python syntax, variables, loops, functions.",
        point_b="Wants to understand and use asynchronous programming in Python (async/await).",
        preferences="Prefers conceptual explanations first, then simple code examples.",
    )
    sample_guidelines = PedagogicalGuidelines(
        guideline="Introduce async concepts gradually, starting with the 'why' and core ideas before showing complex code. Use simple analogies where possible."
    )

    # 2. Instantiate the agent
    agent = create_journey_crafter_agent(openrouter_model)

    # 3. Construct the input prompt
    input_prompt = (
        f"Create a learning plan based on the following profile and guidelines:\n\n"
        f"**Student Profile:**\n"
        f"- Current Knowledge (Point A): {sample_onboarding_data.point_a}\n"
        f"- Learning Goal (Point B): {sample_onboarding_data.point_b}\n"
        f"- Learning Preferences: {sample_onboarding_data.preferences}\n\n"
        f"**Pedagogical Guideline:** {sample_guidelines.guideline}"
    )

    # 4. Run the agent
    print("\nRunning JCA with prompt...\n")  # Log prompt for debugging
    result = await agent.run(input_prompt)
    print(f"Agent Raw Result Data: {result.data}")  # Log raw result

    # 5. --- Assertions (Simplified) ---
    assert isinstance(result.data, LearningPlan), (
        f"Expected result.data to be LearningPlan, got {type(result.data)}"
    )

    learning_plan = result.data
    assert learning_plan.steps, "Learning plan should contain steps"
    assert isinstance(learning_plan.steps, list), "Steps should be a list"
    assert len(learning_plan.steps) > 0, "Learning plan should have at least one step"

    print(f"\n--- Generated Plan ({len(learning_plan.steps)} steps) --- ")
    for i, step_text in enumerate(learning_plan.steps):
        print(f"  Step {i + 1}: '{step_text}'")
        assert isinstance(step_text, str), f"Item {i} in steps should be a string"
        assert step_text, f"Step text {i + 1} should not be empty"

    print(f"\nUsage: {result.usage()}")
    print("---------------------------")
