# tests/agents/test_teacher_agent.py
import os

import pytest
from dotenv import load_dotenv
from pydantic_ai.agent import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# Import the agent creation function
from src.agents.teacher_agent import create_teacher_agent

# --- Test Setup ---


@pytest.fixture(scope="module")
def openai_model() -> OpenAIModel:
    """Fixture to provide a configured OpenAIModel instance for tests."""
    load_dotenv()
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    MODEL_NAME = "google/gemini-2.0-flash-lite-001"  # Or your preferred model

    if not OPENROUTER_API_KEY:
        pytest.skip("OPENROUTER_API_KEY not found in environment variables.")

    # Configure OpenRouter
    return OpenAIModel(
        MODEL_NAME,
        provider=OpenAIProvider(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
            # No timeout specified here, rely on defaults for now
        ),
    )


# --- Tests ---


@pytest.mark.asyncio
async def test_teacher_agent_mvp_output(openai_model: OpenAIModel):
    """Tests the Teacher Agent's ability to generate initial step guidance."""
    # Create the agent
    agent: Agent = create_teacher_agent(openai_model)

    # Sample input data based on task requirements
    sample_guideline = "Explain the concept first, then provide a simple code example."
    sample_plan_step = (
        "Introduce the concept of Python lists and how to create an empty list."
    )

    # Construct the prompt according to the agent's expected format
    input_prompt = (
        f"Start teaching the student according to these instructions:\n\n"
        f"Pedagogical Guideline: {sample_guideline}\n"
        f"Current Learning Step: {sample_plan_step}"
    )

    # Run the agent
    result = await agent.run(input_prompt)

    # Assertions based on Acceptance Criteria
    assert result.data is not None, "Agent should produce some output."
    assert isinstance(result.data, str), "Agent output should be a string."
    assert len(result.data.strip()) > 0, "Agent output string should not be empty."

    # Basic content check: Does the output mention key terms?
    # Note: This is a basic check. More sophisticated checks might involve
    # asserting the structure based on the guideline (e.g., explanation first).
    output_lower = result.data.lower()
    assert "list" in output_lower, "Output should mention 'list'."
    assert "python" in output_lower, "Output should mention 'Python'."
    # Check for alignment with guideline (concept first)
    # This is harder to assert robustly, but we can check for keywords related to explanation
    # assert "concept" in output_lower or "idea" in output_lower or "what is" in output_lower, "Output should likely start with conceptual explanation based on guideline."
    # For now, let's keep assertions simple.

    print("\n--- Test Teacher Agent MVP Output ---")
    print(f"Guideline: {sample_guideline}")
    print(f"Plan Step: {sample_plan_step}")
    print(f"Agent Output:\n{result.data}")
    print(f"Usage: {result.usage()}")
    print("------------------------------------")
