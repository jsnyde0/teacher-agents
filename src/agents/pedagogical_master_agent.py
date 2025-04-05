# src/agents/pedagogical_master_agent.py
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

# Import the input data model from the onboarding agent
from .onboarding_agent import OnboardingData


# --- Data Model for Output ---
class PedagogicalGuidelines(BaseModel):
    """Defines the pedagogical guidelines for the Teacher Agent."""

    guideline: str = Field(
        ..., description="The core pedagogical instruction or style to follow."
    )
    # We could add more fields later, like suggested activity types, etc.


# --- Agent Definition ---
def create_pedagogical_master_agent(model: OpenAIModel) -> Agent:
    """Creates the Pedagogical Master Agent instance."""
    return Agent(
        model=model,
        result_type=PedagogicalGuidelines,
        system_prompt=(
            "You are the Pedagogical Master Agent. Your role is to determine the \
                best initial teaching approach "
            "based on the student's profile. Analyze the provided OnboardingData \
                (Point A, Point B, Preferences) "
            "and generate a concise, actionable pedagogical guideline for the \
                Teacher Agent. Focus on how to approach explanations, examples, \
                    or interactions based on the student's preferences and goals."
            "Respond *only* with the structured PedagogicalGuidelines object."
        ),
    )


# Example Usage (for basic check)
if __name__ == "__main__":
    import asyncio
    import os

    from dotenv import load_dotenv
    from pydantic_ai.providers.openai import OpenAIProvider

    load_dotenv()
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

    if not OPENROUTER_API_KEY:
        print("Error: OPENROUTER_API_KEY not found in .env file.")
    else:
        # Configure OpenRouter
        openrouter_model = OpenAIModel(
            "google/gemini-flash-1.5",
            provider=OpenAIProvider(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY,
            ),
        )

        # Create agent
        agent = create_pedagogical_master_agent(openrouter_model)

        # Sample input data (replace with actual OnboardingData instance)
        sample_onboarding_data = OnboardingData(
            point_a="Knows basic Python syntax and data types.",
            point_b="Wants to learn FastAPI for web APIs.",
            preferences="Prefers learning by building small projects and \
                seeing code examples.",
        )

        # Construct prompt including the onboarding data
        input_prompt = (
            f"Determine pedagogical guidelines based on the following student \
                profile:\n"
            f"Current Knowledge (Point A): {sample_onboarding_data.point_a}\n"
            f"Learning Goal (Point B): {sample_onboarding_data.point_b}\n"
            f"Learning Preferences: {sample_onboarding_data.preferences}"
        )

        async def run_agent():
            print("Running agent with input profile...\n")
            # print(input_prompt) # Optional: print the full prompt
            result = await agent.run(input_prompt)
            print("\n--- Agent Result ---")
            if isinstance(result.data, PedagogicalGuidelines):
                print(f"Guideline: {result.data.guideline}")
            else:
                print(f"Unexpected result type: {type(result.data)}")
                print(f"Data: {result.data}")
            print(f"Usage: {result.usage()}")
            print("------------------")

        asyncio.run(run_agent())
