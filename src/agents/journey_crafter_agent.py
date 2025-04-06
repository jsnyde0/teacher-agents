# src/agents/journey_crafter_agent.py
from typing import List

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

# Import necessary input data models
from .onboarding_agent import OnboardingData
from .pedagogical_master_agent import PedagogicalGuidelines

# --- Data Models for Output ---

# Removed PlanStep model
# class PlanStep(BaseModel): ...


class LearningPlan(BaseModel):
    """Defines the overall structure of the learning journey as a list of steps."""

    # Simplified structure: List of strings for MVP
    steps: List[str] = Field(
        ..., description="An ordered list of descriptive learning steps."
    )


# --- Agent Definition ---


def create_journey_crafter_agent(model: OpenAIModel) -> Agent:
    """Creates the Journey Crafter Agent instance."""
    return Agent(
        model=model,
        result_type=LearningPlan,  # Target is the simplified LearningPlan
        system_prompt=(
            "You are an expert Journey Crafter Agent. Your task is to create a step-by-step Learning Plan "
            "to help a student progress from their current knowledge (Point A) to their learning goal (Point B), "
            "adhering to the provided pedagogical guidelines.\n\n"
            "**Input Analysis:**\n"
            "Analyze the student's `OnboardingData` (Point A, Point B, Preferences) and the `PedagogicalGuidelines`.\n\n"
            "**Plan Generation:**\n"
            "1. **You MUST generate** a sequence of logical, small, manageable steps to bridge the gap from Point A to Point B.\n"
            "2. Each step **MUST** be a concise string describing the learning objective or activity for that step.\n"
            "3. The sequence of steps **MUST** be comprehensive enough to reach the learning goal (Point B).\n"
            "4. You **MUST** consider the `PedagogicalGuidelines` when deciding the content and style of each step's description.\n\n"
            "**Output Format:**\n"
            "Respond *only* with a single, valid JSON object matching the `LearningPlan` schema. "
            "The value for the `steps` key MUST be a JSON array `[]` containing strings `"
            "`, where each string describes one learning step. "
            "Ensure the `steps` array is not empty if a plan is possible.\n"
            "Example of correct structure:"
            "```json"
            "{"
            '  "steps": ['
            '    "Understand the concept of concurrency vs parallelism.",'
            '    "Learn the basics of Python\'s `asyncio` library.",'
            '    "Practice writing simple `async def` functions and using `await`.'
            "  ]"
            "}"
            "```"
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
                client_kwargs={"timeout": 60.0},  # Keep increased timeout
            ),
        )

        # Create agent
        agent = create_journey_crafter_agent(openrouter_model)

        # Sample input data (same as before)
        sample_onboarding_data = OnboardingData(
            point_a="Knows basic Python syntax, variables, loops, functions.",
            point_b="Wants to understand and use asynchronous programming in Python (async/await).",
            preferences="Prefers conceptual explanations first, then simple code examples.",
        )
        sample_guidelines = PedagogicalGuidelines(
            guideline="Introduce async concepts gradually, starting with the 'why' and core ideas before showing complex code. Use simple analogies where possible."
        )

        # Construct prompt including the input data
        input_prompt = (
            f"Create a learning plan based on the following profile and guidelines:\n\n"
            f"**Student Profile:**\n"
            f"- Current Knowledge (Point A): {sample_onboarding_data.point_a}\n"
            f"- Learning Goal (Point B): {sample_onboarding_data.point_b}\n"
            f"- Learning Preferences: {sample_onboarding_data.preferences}\n\n"
            f"**Pedagogical Guideline:** {sample_guidelines.guideline}"
        )

        async def run_agent():
            print("Running Journey Crafter Agent...\n")
            result = await agent.run(input_prompt)
            print("\n--- Agent Result ---")
            if isinstance(result.data, LearningPlan):
                print(f"Generated Learning Plan ({len(result.data.steps)} steps):")
                for i, step_text in enumerate(result.data.steps):
                    print(f"  {i + 1}. {step_text}")  # Print the string directly
            else:
                print(f"Unexpected result type: {type(result.data)}")
                print(f"Data: {result.data}")
            print(f"Usage: {result.usage()}")
            print("------------------")

        asyncio.run(run_agent())
