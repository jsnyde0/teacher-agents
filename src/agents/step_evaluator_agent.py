# src/agents/step_evaluator_agent.py
from typing import Literal

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

# --- Agent Definition ---


def create_step_evaluator_agent(model: OpenAIModel) -> Agent:
    """Creates the Step Evaluator Agent instance.

    This agent analyzes a student's response in the context of the teacher's
    last message to determine if the student is ready to proceed to the next
    learning step. It outputs a constrained literal: PROCEED, STAY, or UNCLEAR.

    Args:
        model: The PydanticAI model instance (e.g., OpenAIModel configured for OpenRouter).

    Returns:
        An initialized Agent instance for the Step Evaluator.
    """
    return Agent(
        model=model,
        # The agent must return one of these exact strings
        result_type=Literal["PROCEED", "STAY", "UNCLEAR"],
        system_prompt="""You are a Step Evaluator assistant. Your task is to analyze the student's response in the context of the teacher's last message to determine if the student is ready to proceed to the next learning step.

**Context:**
- Teacher's Last Message: [The message the teacher sent just before the student replied]
- Student's Response: [The student's latest message]

**Instructions:**
Analyze the student's response *in relation to* the teacher's last message.

Respond with *only* one of the following words:

- 'PROCEED': If the student *explicitly* states they are ready, understand, want to move on (e.g., 'next', 'ok let's continue', 'got it, what's next?') AND their message doesn't indicate confusion or questions related to the teacher's last message.

- 'STAY': If the student asks a question about the topic, expresses confusion, provides an answer related to the teacher's query/instruction, indicates they need more time, or is otherwise actively engaging with the *current* step prompted by the teacher (e.g., 'why?', 'I'm confused', 'team = [a,b,c]', 'hmm', 'ok' if teacher just gave info/asked to try something).

- 'UNCLEAR': If the message is ambiguous, off-topic, a simple greeting/thanks, or doesn't provide a clear signal about their progress or readiness *in the context of the teacher's message* (e.g., 'cool', 'thanks', 'hello').

**Crucial:** An acknowledgement like 'ok' or 'sure' in direct response to a teacher's question or instruction to try something usually means 'STAY', not 'PROCEED'. Only signal 'PROCEED' if the user clearly indicates finishing the current step and wanting the *next* one.

Your response must be *exactly* one of: PROCEED, STAY, or UNCLEAR.
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
            "google/gemini-2.0-flash-lite-001",
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
