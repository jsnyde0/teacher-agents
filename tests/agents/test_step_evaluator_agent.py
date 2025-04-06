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
    MODEL_NAME = (
        "google/gemini-2.0-flash-lite-001"  # Fast model suitable for classification
    )

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
    "teacher_message, student_message, expected_outcome",
    [
        # === PROCEED cases ===
        # Explicit readiness after generic info/question
        (
            "Here is the info on lists. Let me know your thoughts.",
            "Ok, got it, next step please",
            "PROCEED",
        ),
        (
            "Can you try writing a simple list?",
            "Done! Ready for the next topic.",
            "PROCEED",
        ),
        ("We will cover loops now.", "continue", "PROCEED"),  # User takes initiative
        # === STAY cases ===
        # Asking question related to teacher message
        ("Lists store items in order.", "Why is order important?", "STAY"),
        ("You can add items using .append()", "What is append?", "STAY"),
        # Providing answer/attempt related to teacher question
        ("How would you create an empty list?", "my_list = []", "STAY"),
        ("Can you give an example of a variable?", "x = 5", "STAY"),
        # Expressing confusion
        ("Next we use loops.", "I'm confused about the previous step.", "STAY"),
        # Simple acknowledgement of instruction/info (NOT proceed)
        ("Try running this code: print('hello')", "ok", "STAY"),
        ("Here is an example: x=1", "sure", "STAY"),
        ("Let's talk about functions.", "Okay", "STAY"),  # Ambiguous 'Okay' means stay
        # Needing more time
        (
            "Think about how you would use a dictionary here.",
            "hmm, let me think",
            "STAY",
        ),
        # === UNCLEAR cases ===
        # Off-topic
        ("We are discussing variables.", "What's for lunch?", "UNCLEAR"),
        # Simple pleasantries
        ("Here is how loops work.", "Thanks!", "UNCLEAR"),
        ("Let's start with data types.", "Hello!", "UNCLEAR"),
        # Ambiguous one-word answers not clearly related
        ("Think about the structure.", "Cool.", "UNCLEAR"),
    ],
)
@pytest.mark.llm
@pytest.mark.asyncio
async def test_step_evaluator_agent_scenarios(
    openai_model: OpenAIModel,
    teacher_message: str,
    student_message: str,
    expected_outcome: Literal["PROCEED", "STAY", "UNCLEAR"],
):
    """Tests the Step Evaluator Agent with various teacher/student message pairs."""
    agent: Agent = create_step_evaluator_agent(openai_model)

    # Construct the combined input prompt for the evaluator
    evaluator_input_prompt = (
        f"Teacher's Last Message: {teacher_message}\n"
        f"Student's Response: {student_message}"
    )

    # Run the agent with the combined prompt
    result = await agent.run(evaluator_input_prompt)

    # Assertions
    assert result.data is not None, "Agent should produce a result."
    assert result.data == expected_outcome, (
        f"Mismatch for Teacher: '{teacher_message}' | Student: '{student_message}'. Expected {expected_outcome}, got {result.data}"
    )

    print(
        f"\n--- Test Step Evaluator --- \n  Teacher: '{teacher_message}'\n  Student: '{student_message}'\n  Expected: {expected_outcome}\n  Actual:   {result.data}\n  Usage:    {result.usage()}\n-----------------------------"
    )
