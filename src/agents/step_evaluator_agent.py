# src/agents/step_evaluator_agent.py
from typing import Literal

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

# --- Agent Definition ---


def create_step_evaluator_agent(model: OpenAIModel) -> Agent:
    """Creates the Step Evaluator Agent instance."""
    return Agent(
        model=model,
        # The agent must return one of these exact strings
        result_type=Literal["PROCEED", "STAY", "UNCLEAR"],
        system_prompt="""You are a Step Evaluator assistant. Your sole task is to analyze the student's most recent message and determine if they are ready to proceed to the next learning step. Focus *only* on their readiness to move on.

Analyze the student's message and respond with *only* one of the following words:

- 'PROCEED': If the student explicitly states they are ready, understands, or wants to move on (e.g., 'ok', 'next', 'got it', 'continue', 'I understand').
- 'STAY': If the student asks a question about the current topic, expresses confusion, asks for clarification, indicates they need more time, or is actively discussing the current step's content (e.g., 'why does it do that?', 'I'm confused about...', 'hang on', 'what if...').
- 'UNCLEAR': If the message is ambiguous, off-topic, a simple greeting, or doesn't provide a clear signal about their readiness regarding the current learning step.

Do not add any explanation or conversational text. Your response must be *exactly* one of: PROCEED, STAY, or UNCLEAR.
""",
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
            # Using a fast model as this is a simple classification task
            "google/gemini-flash-1.5",
            provider=OpenAIProvider(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY,
            ),
        )

        # Create agent
        agent = create_step_evaluator_agent(openrouter_model)

        # Sample student messages
        test_messages = [
            "Okay, got it! Let's move on.",
            "next please",
            "I'm not sure I understand the part about dictionaries.",
            "Why does that code work?",
            "Hmm, let me try that out.",
            "What's for lunch?",
            "Thanks!",
            "ok",
        ]

        async def run_evaluations():
            print("Running Step Evaluator Agent on test messages...\n")
            for message in test_messages:
                print(f"Student Message: '{message}'")
                result = await agent.run(message)
                print(f"  -> Agent Result: {result.data} (Usage: {result.usage()})")
            print("\n------------------")

        asyncio.run(run_evaluations())
