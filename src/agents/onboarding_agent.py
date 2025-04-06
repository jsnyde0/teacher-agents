# src/agents/onboarding_agent.py
from typing import Union

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
    return Agent(
        model=model,
        # Agent can now return either a string (for conversation) or the final data
        result_type=Union[str, OnboardingData],
        system_prompt=(
            "You are a friendly and helpful onboarding agent. Your goal is to \
                understand the user's "
            "current knowledge (Point A), their learning goal (Point B), and \
                their learning preferences. "
            "Engage in a natural conversation to gather this information. "
            "Ask clarifying questions one at a time ONLY if information for Point A, Point B, or Preferences is clearly missing or ambiguous. "
            "Check the conversation history to avoid repeating questions. "
            "**CRITICAL: Once you are confident you have gathered reasonable answers for all three pieces of information (Point A, Point B, Preferences), you MUST stop asking questions and respond *only* with the structured OnboardingData object.** Do not ask for further details or refinements if you already have the core information. Do not include any conversational text in the final structured response. "
            "If you still need information for Point A, B, or Preferences, respond with a conversational question as a string."
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
            "google/gemini-2.0-flash-lite-001",  # Updated model
            provider=OpenAIProvider(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY,
            ),
        )

        # Create agent
        agent = create_onboarding_agent(openrouter_model)

        # --- Simulate Conversation ---
        async def run_conversation():
            print("--- Starting Simulated Conversation ---")
            history = []

            # Turn 1: User gives partial info
            prompt1 = "Hi there! I want to learn FastAPI."
            print(f"\nUser: {prompt1}")
            result1 = await agent.run(prompt1, message_history=history)
            print(f"Agent: {result1.data}")
            history.extend(result1.all_messages())

            # Turn 2: User gives more info
            if isinstance(result1.data, str):  # Check if agent asked a question
                prompt2 = "I know basic Python syntax and data types."
                print(f"\nUser: {prompt2}")
                result2 = await agent.run(prompt2, message_history=history)
                print(f"Agent: {result2.data}")
                history.extend(result2.all_messages())

                # Turn 3: User gives final info
                if isinstance(result2.data, str):
                    prompt3 = "I like learning by building small projects."
                    print(f"\nUser: {prompt3}")
                    result3 = await agent.run(prompt3, message_history=history)
                    print(f"Agent: {result3.data}")
                    history.extend(result3.all_messages())

                    if isinstance(result3.data, OnboardingData):
                        print("\n--- Final Structured Data Received ---")
                        print(result3.data)
                    else:
                        print("\nAgent did not return structured data after 3 turns.")
                else:
                    print(
                        "\nAgent returned structured data prematurely (after turn 2)."
                    )
                    print(result2.data)
            else:
                print("\nAgent returned structured data prematurely (after turn 1).")
                print(result1.data)

            print("--------------------------------------")

        asyncio.run(run_conversation())
