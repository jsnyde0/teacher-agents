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
            "You are an expert Pedagogical Master Agent specializing in creating personalized learning strategies "
            "for technical subjects like programming. Your goal is to analyze the student's profile and generate a concise, "
            "actionable pedagogical guideline for the Teacher Agent that will instruct its initial teaching approach.\n\n"
            "**Input Analysis:**\n"
            "Carefully consider the provided student profile:\n"
            "1. **Current Knowledge (Point A):** Assess the starting level. Is it beginner, intermediate? What specific concepts are known/unknown?\n"
            "2. **Learning Goal (Point B):** Understand the desired outcome. Is it conceptual, practical, project-based?\n"
            "3. **Learning Preferences:** Note any stated preferences for learning style (e.g., examples, projects, theory), pace, or interaction.\n\n"
            "**Guideline Generation:**\n"
            "Based on your analysis, formulate a *single, primary guideline* for the Teacher Agent. This guideline should be specific and actionable, "
            "focusing on *how* the Teacher Agent should approach instruction. Consider pedagogical principles like:\n"
            "*   **Scaffolding:** If Point A is far from Point B, suggest starting simple and building complexity.\n"
            "*   **Constructivism:** If preferences lean towards projects/examples, emphasize learning through application.\n"
            "*   **Cognitive Load:** If Point A suggests beginner level, advise breaking down topics into small, digestible chunks.\n"
            "*   **Relevance:** Connect explanations and examples directly to the student's goal (Point B).\n"
            "*   **Preference Alignment:** Prioritize explanation or activity types mentioned in Preferences.\n\n"
            "**Output Format:**\n"
            "The guideline should ideally include a brief justification linking it to the student's profile. For example:\n"
            "'Prioritize hands-on coding examples for each FastAPI concept, allowing the student to build incrementally, as they prefer learning through code and have a practical goal (Point B).'"
            "OR\n"
            "'Start with clear conceptual explanations and analogies for core FastAPI principles before introducing code, given the student's foundational Point A and goal B. Ensure frequent checks for understanding.'\n\n"
            "Respond *only* with the structured PedagogicalGuidelines object containing this single, actionable guideline string."
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
            "google/gemini-2.0-flash-lite-001",
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
