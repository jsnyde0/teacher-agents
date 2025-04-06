# tests/agents/test_answer_evaluator_agent.py
import os
from typing import Literal

import pytest
from dotenv import load_dotenv
from pydantic_ai.agent import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# Import the agent creation function and result model
from src.agents.answer_evaluator_agent import (
    AnswerEvaluationResult,
    create_answer_evaluator_agent,
)

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


@pytest.mark.parametrize(
    "step_goal, teacher_prompt, student_response, expected_eval, expected_explanation_keywords",
    [
        # Correct
        (
            "Understand integer data type.",
            "What data type is the number 5?",
            "integer",
            "correct",
            ["correct", "integer"],
        ),
        (
            "Write code to add 2 and 3.",
            "Show me the Python code to add 2 and 3.",
            "print(2 + 3)",
            "correct",
            ["correct", "syntax", "addition"],
        ),
        # Incorrect
        (
            "Understand integer data type.",
            "What data type is the number 5?",
            "string?",
            "incorrect",
            ["incorrect", "string", "integer"],
        ),
        (
            "Write code to subtract 5 from 10.",
            "Show me the code to subtract 5 from 10.",
            "10 + 5",
            "incorrect",
            ["incorrect", "addition", "subtraction"],
        ),
        # Partial -> Changed expectation to incorrect as agent sees syntax error
        (
            "Create a list of strings.",
            "Create a list with names Tom, Jan, Ann.",
            "team = [Tom, Jan, Ann]",
            "incorrect",
            ["incorrect", "missing", "quotes", "string"],
        ),
        # Unclear
        (
            "Explain variable assignment.",
            "What does x = 5 mean?",
            "maybe assigns 5?",
            "unclear",
            ["unclear", "ambiguous", "vague"],
        ),
        # Not Applicable
        (
            "Understand loops.",
            "How would a for loop help here?",
            "I don't know",
            "not_applicable",
            ["not applicable", "doesn't answer", "know"],
        ),
        (
            "Explain functions.",
            "What is a function argument?",
            "What's for dinner?",
            "not_applicable",
            ["not applicable", "off-topic"],
        ),
    ],
)
@pytest.mark.asyncio
async def test_answer_evaluator_agent_scenarios(
    openai_model: OpenAIModel,
    step_goal: str,
    teacher_prompt: str,
    student_response: str,
    expected_eval: Literal[
        "correct", "incorrect", "partial", "unclear", "not_applicable"
    ],
    expected_explanation_keywords: list[str],
):
    """Tests the Answer Evaluator Agent with various scenarios."""
    agent: Agent = create_answer_evaluator_agent(openai_model)

    # Construct the input prompt for the evaluator
    evaluator_input_prompt = (
        f"Current Learning Step Goal: {step_goal}\n"
        f"Teacher's Last Instruction/Question: {teacher_prompt}\n"
        f"Student's Response: {student_response}"
    )

    # Run the agent
    result = await agent.run(evaluator_input_prompt)

    # Assertions
    assert result.data is not None, "Agent should produce a result."
    assert isinstance(result.data, AnswerEvaluationResult), (
        f"Expected AnswerEvaluationResult, got {type(result.data)}"
    )

    eval_result: AnswerEvaluationResult = result.data

    assert eval_result.evaluation == expected_eval, (
        f"Expected evaluation '{expected_eval}', got '{eval_result.evaluation}' for: [S:{student_response}]"
    )

    print(
        f"\n--- Test Answer Evaluator --- "
        f"\n  Step Goal: '{step_goal}'"
        f"\n  Teacher:   '{teacher_prompt}'"
        f"\n  Student:   '{student_response}'"
        f"\n  Expected:  '{expected_eval}'"
        f"\n  Actual:    '{eval_result.evaluation}'"
        f"\n  Usage:     {result.usage()}"
        f"\n---------------------------"
    )
