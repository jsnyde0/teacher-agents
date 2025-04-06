# src/agents/teacher_agent.py
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

# Import necessary input data models (though we might just pass strings)
# from .onboarding_agent import OnboardingData
# from .pedagogical_master_agent import PedagogicalGuidelines
# from .journey_crafter_agent import LearningPlan, PlanStep # If passing structured plan

# --- Agent Definition ---


def create_teacher_agent(model: OpenAIModel) -> Agent:
    """Creates the Teacher Agent instance."""
    # For MVP, the agent just initiates the conversation for the *first* step.
    # It receives the step description and pedagogical guideline.
    # It needs to output conversational text (string).
    return Agent(
        model=model,
        result_type=str,  # Outputs conversational text
        system_prompt=(
            "You are a friendly and encouraging Teacher Agent. Your goal is to guide a student through a single learning step, "
            "following specific pedagogical guidelines.\n\n"
            "**Your Task:**\n"
            "You will receive the `Pedagogical Guideline` for how to teach, and the `Current Learning Step` description. "
            "Your job is to generate the *initial* conversational text to present this step to the student. "
            'Critically, after introducing the topic, you MUST immediately transition into the first piece of instruction, a relevant example, or a guiding question *about the topic itself* to encourage engagement. Do NOT simply ask "Are you ready?" or similar generic readiness questions. '
            "Make sure your response directly addresses the learning step and follows the spirit of the guideline "
            "(e.g., if the guideline suggests examples first, start with an example; if it suggests concepts first, start with an explanation). "
            "Keep your tone supportive and engaging.\n\n"
            "**Input Format (within user prompt):**\n"
            "- Pedagogical Guideline: [Guideline text]\n"
            "- Current Learning Step: [Step description text]\n\n"
            "**Output Format:**\n"
            "Respond *only* with the conversational text string to present the step and begin the interaction."
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
        agent = create_teacher_agent(openrouter_model)

        # Sample input data
        sample_guideline = "Focus on practical examples first to illustrate the concept, then explain the theory."
        sample_step = "Introduce Python dictionaries and how to create them."

        # Construct prompt including the input data
        input_prompt = (
            f"Start teaching the student according to these instructions:\n\n"
            f"Pedagogical Guideline: {sample_guideline}\n"
            f"Current Learning Step: {sample_step}"
        )

        async def run_agent():
            print("Running Teacher Agent...\n")
            print(f"Input Prompt:\n{input_prompt}\n")  # Optional
            result = await agent.run(input_prompt)
            print("\n--- Agent Result ---")
            if isinstance(result.data, str):
                print(f"Agent Response:\n{result.data}")
            else:
                print(f"Unexpected result type: {type(result.data)}")
                print(f"Data: {result.data}")
            print(f"Usage: {result.usage()}")
            print("------------------")

        asyncio.run(run_agent())
