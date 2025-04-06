# tests/agents/debug_teacher_agent.py
import os
import traceback
import json

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
    
    # Use a well-supported model
    MODEL_NAME = "openai/gpt-3.5-turbo"
    
    print(f"API Key available: {bool(OPENROUTER_API_KEY)}")
    print(f"API Key value: {OPENROUTER_API_KEY[:5]}...{OPENROUTER_API_KEY[-5:]}")
    print(f"API Base URL: {OPENROUTER_BASE_URL}")
    print(f"Model name: {MODEL_NAME}")

    if not OPENROUTER_API_KEY:
        pytest.skip("OPENROUTER_API_KEY not found in environment variables.")

    # Configure OpenRouter without timeout parameter
    return OpenAIModel(
        MODEL_NAME,
        provider=OpenAIProvider(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
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


# --- Debug Test ---

@pytest.mark.asyncio
async def test_debug_teacher_agent(openai_model, sample_onboarding_data, sample_guidelines, sample_learning_plan):
    """Debug test to print detailed error information when calling the API."""
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
    
    print("\n--- Debug Teacher Agent Test ---")
    print(f"Input prompt: {input_prompt[:200]}...")
    
    try:
        # Run the agent
        result = await agent.run(input_prompt)
        
        # If successful, print the result
        print(f"Success! Response received.")
        print(f"Content: {result.data.content[:200]}...")
        print(f"Current Step Index: {result.data.current_step_index}")
        print(f"Step Completed: {result.data.completed}")
        print(f"Usage: {result.usage()}")
        
    except Exception as e:
        # Print detailed error information
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print(f"Traceback:")
        traceback.print_exc()
        
        # If it's an HTTP error, try to extract more information
        if hasattr(e, 'response'):
            try:
                print(f"Status code: {e.response.status_code}")
                print(f"Response headers: {json.dumps(dict(e.response.headers), indent=2)}")
                print(f"Response content: {e.response.text}")
            except:
                print("Could not extract response details")
        
        # Re-raise the exception to fail the test
        raise
    
    print("------------------------------------") 