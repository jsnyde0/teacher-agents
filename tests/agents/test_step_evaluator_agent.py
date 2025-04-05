# tests/agents/test_step_evaluator_agent.py
import os
from typing import Literal

import pytest
from dotenv import load_dotenv
from pydantic_ai.agent import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# Import the agent creation function
from src.agents.step_evaluator_agent import create_step_evaluator_agent

# --- Test Setup ---


@pytest.fixture(scope="module")
def openai_model() -> OpenAIModel:
    """Fixture to provide a configured OpenAIModel instance for tests."""
    load_dotenv()
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    MODEL_NAME = "google/gemini-flash-1.5"  # Fast model suitable for classification

    if not OPENROUTER_API_KEY:
        pytest.skip("OPENROUTER_API_KEY not found in environment variables.")

    return OpenAIModel(
        MODEL_NAME,
        provider=OpenAIProvider(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
        ),
    )


# --- Test Cases ---


# Parameterize the test function to cover different scenarios
@pytest.mark.parametrize(
    "student_message, expected_outcome",
    [
        # PROCEED cases
        ("Okay, got it! Let's move on.", "PROCEED"),
        ("next please", "PROCEED"),
        ("I understand now.", "PROCEED"),
        ("ok", "PROCEED"),
        ("Continue", "PROCEED"),
        ("Ready for the next step.", "PROCEED"),
        # STAY cases
        ("I'm not sure I understand the part about dictionaries.", "STAY"),
        ("Why does that code work?", "STAY"),
        ("Can you explain that again?", "STAY"),
        ("Hmm, let me try that out first.", "STAY"),  # Indicates needing time
        ("Wait, what happens if I change this?", "STAY"),
        ("I have a question.", "STAY"),
        # UNCLEAR cases
        ("What's for lunch?", "UNCLEAR"),
        ("Thanks!", "UNCLEAR"),  # Ambiguous gratitude
        ("Hello there.", "UNCLEAR"),
        ("Interesting.", "UNCLEAR"),  # Doesn't signal readiness
        ("Python is cool.", "UNCLEAR"),
    ],
)
@pytest.mark.asyncio
async def test_step_evaluator_agent_scenarios(
    openai_model: OpenAIModel,
    student_message: str,
    expected_outcome: Literal["PROCEED", "STAY", "UNCLEAR"],
):
    """Tests the Step Evaluator Agent with various student message inputs."""
    agent: Agent = create_step_evaluator_agent(openai_model)

    # Run the agent with the student message
    # No prior history needed for this agent usually
    result = await agent.run(student_message)

    # Assertions
    assert result.data is not None, "Agent should produce a result."
    assert result.data == expected_outcome, (
        f"Expected {expected_outcome} but got {result.data} for message: '{student_message}'"
    )

    print(f"\n--- Test Step Evaluator: '{student_message}' ---")
    print(f"Expected: {expected_outcome}")
    print(f"Actual:   {result.data}")
    print(f"Usage:    {result.usage()}")
    print("--------------------------------------------------")
