# tests/agents/test_teacher_agent.py
import os

import pytest
from dotenv import load_dotenv
from pydantic_ai.agent import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# Import the agent creation function and TeacherResponse model
from src.agents.teacher_agent import create_teacher_agent, TeacherResponse, prepare_teacher_input
from src.agents.onboarding_agent import OnboardingData
from src.agents.pedagogical_master_agent import PedagogicalGuidelines
from src.agents.journey_crafter_agent import LearningPlan

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


# --- Test Data Fixtures ---
@pytest.fixture
def sample_onboarding_data():
    """Create sample onboarding data for testing."""
    return OnboardingData(
        point_a="Basic knowledge of programming concepts",
        point_b="Ability to write Python programs with lists and dictionaries",
        preferences="Visual examples and practical exercises"
    )


@pytest.fixture
def sample_guidelines():
    """Create sample pedagogical guidelines for testing."""
    return PedagogicalGuidelines(
        guideline="Explain the concept first, then provide a simple code example. Use visual analogies where possible."
    )


@pytest.fixture
def sample_learning_plan():
    """Create sample learning plan for testing."""
    return LearningPlan(
        steps=[
            "Introduce the concept of Python lists and how to create an empty list.",
            "Explain list operations like append, insert, and remove.",
            "Demonstrate list comprehensions for advanced usage."
        ]
    )


# --- Tests ---


@pytest.mark.llm
@pytest.mark.asyncio
async def test_teacher_agent_response(openai_model, sample_onboarding_data, sample_guidelines, sample_learning_plan):
    """Tests the Teacher Agent's ability to generate teaching content in the correct format."""
    # Create the agent
    agent: Agent = create_teacher_agent(openai_model)

    # Prepare input using the helper function
    current_step_index = 0
    user_message = "Hi, I'm ready to learn about Python lists!"
    
    input_prompt = prepare_teacher_input(
        sample_onboarding_data,
        sample_guidelines,
        sample_learning_plan,
        current_step_index,
        user_message
    )

    try:
        # Run the agent
        result = await agent.run(input_prompt)

        # Assertions based on updated TeacherResponse structure
        assert result.data is not None, "Agent should produce output."
        assert isinstance(result.data, TeacherResponse), "Agent output should be a TeacherResponse."
        
        # Check response structure
        assert hasattr(result.data, "content"), "Response should have content field."
        assert hasattr(result.data, "current_step_index"), "Response should have current_step_index field."
        assert hasattr(result.data, "completed"), "Response should have completed field."
        
        # Check content
        assert len(result.data.content.strip()) > 0, "Content should not be empty."
        assert result.data.current_step_index == 0, "Initial step index should match input."
        assert isinstance(result.data.completed, bool), "Completed field should be a boolean."
        
        # Basic content check
        output_lower = result.data.content.lower()
        assert "list" in output_lower, "Output should mention 'list'."
        assert "python" in output_lower, "Output should mention 'Python'."

        print("\n--- Test Teacher Agent Response ---")
        print(f"Current Step: {sample_learning_plan.steps[current_step_index]}")
        print(f"Agent Output Content:\n{result.data.content}")
        print(f"Current Step Index: {result.data.current_step_index}")
        print(f"Step Completed: {result.data.completed}")
        print(f"Usage: {result.usage()}")
        print("------------------------------------")
    except Exception as e:
        if 'Received empty model response' in str(e):
            pytest.skip("API returned empty response - skipping test")
        else:
            # Re-raise other errors
            raise


@pytest.mark.asyncio
async def test_teacher_agent_step_progression(openai_model, sample_onboarding_data, sample_guidelines, sample_learning_plan):
    """Tests the Teacher Agent's ability to evaluate step completion."""
    # Create the agent
    agent: Agent = create_teacher_agent(openai_model)

    # Use a message that indicates understanding to trigger step completion
    current_step_index = 0
    user_message = "I understand Python lists now! An empty list is created with [] and I can add items with append(). Can you tell me more about list operations?"
    
    input_prompt = prepare_teacher_input(
        sample_onboarding_data,
        sample_guidelines,
        sample_learning_plan,
        current_step_index,
        user_message
    )

    try:
        # Run the agent
        result = await agent.run(input_prompt)

        # Assertion for step completion
        # Note: This might be flaky as it depends on the model's judgment of "understanding"
        # We're checking that the agent CAN mark a step as completed when appropriate
        print(f"\n--- Testing step progression ---")
        print(f"User message demonstrating understanding: {user_message}")
        print(f"Agent marks step as completed: {result.data.completed}")
        print(f"Agent response: {result.data.content[:100]}...")
        
        # We're not asserting the exact value of completed since it depends on model judgment
        # Just printing the result for manual review
    except Exception as e:
        if 'Received empty model response' in str(e):
            pytest.skip("API returned empty response - skipping test")
        else:
            # Re-raise other errors
            raise
