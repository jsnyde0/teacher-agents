# app_chainlit.py
import os

import chainlit as cl
from dotenv import load_dotenv

# Import agent creation logic and model setup
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from src.agents.onboarding_agent import OnboardingData, create_onboarding_agent

# Load environment variables (for OpenRouter API Key)
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "google/gemini-flash-1.5"  # Or your preferred model

# Global variable to hold the agent instance
onboarding_agent = None


@cl.on_chat_start
async def on_chat_start():
    """
    Initializes the agent when a new chat session starts.
    """
    global onboarding_agent
    if not OPENROUTER_API_KEY:
        await cl.Message(
            content="Error: OPENROUTER_API_KEY not found in environment variables."
        ).send()
        return

    # Configure the model
    model = OpenAIModel(
        MODEL_NAME,
        provider=OpenAIProvider(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
        ),
    )

    # Create the agent instance
    onboarding_agent = create_onboarding_agent(model)

    await cl.Message(
        content="Onboarding Agent initialized. Please tell me about your \
            current knowledge, your learning goals, and your learning preferences."
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """
    Handles incoming user messages and runs the agent.
    """
    global onboarding_agent
    if not onboarding_agent:
        await cl.Message(
            content="Agent not initialized. Please restart the chat."
        ).send()
        return

    # Display a thinking indicator
    msg = cl.Message(content="")  # Empty message to show activity
    await msg.send()

    try:
        # Run the agent with the user's message content
        result = await onboarding_agent.run(message.content)

        # Process the result
        if isinstance(result.data, OnboardingData):
            # Format the structured data for display
            response_content = (
                f"Okay, I understand:\n"
                f"- **Current Knowledge (Point A):** {result.data.point_a}\n"
                f"- **Learning Goal (Point B):** {result.data.point_b}\n"
                f"- **Preferences:** {result.data.preferences}"
            )
            # Add usage info (optional)
            # response_content += f"\n\n_(Usage: {result.usage().total_tokens} tokens)_"

            msg.content = response_content
            await msg.update()
        else:
            # Handle unexpected result format
            msg.content = f"Sorry, I couldn't extract the information in the \
                expected format. Received: {result.data}"
            await msg.update()

    except Exception as e:
        # Handle errors during agent execution
        msg.content = f"An error occurred: {str(e)}"
        await msg.update()
