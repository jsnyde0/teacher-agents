# src/agents/onboarding_agent.py
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import (
    OpenAIModel,  # Using OpenAIModel interface for OpenRouter
)
from pydantic_ai.providers.openai import OpenAIProvider


# --- Data Model ---
class OnboardingData(BaseModel):
    """Data collected during onboarding."""

    point_a: str = Field(..., description="Student's current knowledge/state.")
    point_b: str = Field(..., description="Student's desired learning outcome.")
    preferences: str = Field(
        ..., description="Student's explicitly stated learning preferences."
    )


# --- Agent Definition ---
def create_onboarding_agent(model: OpenAIModel) -> Agent:
    """Creates the Onboarding Agent instance."""
    # For now, the agent logic is simple: rely on the system prompt and response_model
    return Agent(
        model=model,
        result_type=OnboardingData,
        system_prompt=(
            "You are an onboarding agent. Your goal is to understand the user's "
            "current knowledge (Point A), their learning goal (Point B), and their "
            "learning preferences. Extract these three pieces of information."
            "Respond *only* with the structured data."
        ),
    )


# Example Usage (can be run directly for basic check)
if __name__ == "__main__":
    import asyncio
    import os

    from dotenv import load_dotenv

    load_dotenv()
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

    if not OPENROUTER_API_KEY:
        print("Error: OPENROUTER_API_KEY not found in .env file.")
    else:
        # Configure OpenRouter
        openrouter_model = OpenAIModel(
            "google/gemini-2.0-flash-lite-001",  # Or another model
            provider=OpenAIProvider(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY,
            ),
        )

        # Create agent
        agent = create_onboarding_agent(openrouter_model)

        # Example prompt
        test_prompt = "I understand Python basics like variables and lists. \
            I want to learn data analysis with Pandas. I like practical examples."

        async def run_agent():
            print(f"Running agent with prompt: '{test_prompt}'...")
            result = await agent.run(test_prompt)
            print("\n--- Agent Result ---")
            print(f"Data: {result.data}")
            print(f"Usage: {result.usage()}")
            print("------------------")

        asyncio.run(run_agent())
