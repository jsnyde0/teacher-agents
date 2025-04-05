# app_chainlit.py
import os
from typing import List

import chainlit as cl
from dotenv import load_dotenv
from pydantic_ai.messages import ModelMessage

# Import agent creation logic and model setup
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from src.agents.onboarding_agent import OnboardingData, create_onboarding_agent

# Load environment variables (for OpenRouter API Key)
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "google/gemini-flash-1.5"  # Or your preferred model


@cl.on_chat_start
async def on_chat_start():
    """
    Initializes the agent and conversation history when a new chat session starts.
    """
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
    agent = create_onboarding_agent(model)

    # Store the agent and message history in the user session
    cl.user_session.set("agent", agent)
    cl.user_session.set("message_history", [])  # Initialize history

    await cl.Message(
        content="Hello! I'm here to help onboard you. To start, could you tell \
            me a bit about what you already know and what you'd like to learn?"
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """
    Handles incoming user messages, runs the agent with history, and displays \
        responses.
    """
    agent = cl.user_session.get("agent")  # Retrieve agent from session
    message_history: List[ModelMessage] = cl.user_session.get(
        "message_history"
    )  # Retrieve history

    if not agent:
        await cl.Message(
            content="Agent not initialized. Please restart the chat."
        ).send()
        return

    # Append the user's message to the history manually *before* the run
    # Note: Pydantic AI's run method often adds the current user message automatically,
    # but managing history explicitly via Chainlit session is clearer for UI logic.
    # We'll let agent.run add the *latest* user message to its internal processing,
    # but we update *our* session history after the run.

    # Display a thinking indicator
    msg = cl.Message(content="")
    await msg.send()

    try:
        # Run the agent with the user's message content and the existing history
        result = await agent.run(message.content, message_history=message_history)

        # Update the history *after* the run with all messages from the exchange
        message_history.extend(result.all_messages())
        cl.user_session.set("message_history", message_history)  # Store updated history

        # Process the result based on its type
        if isinstance(result.data, OnboardingData):
            # Final structured data received
            response_content = (
                f"Great, thank you! Based on our conversation, here's what I've \
                    gathered:\n"
                f"- **Current Knowledge (Point A):** {result.data.point_a}\n"
                f"- **Learning Goal (Point B):** {result.data.point_b}\n"
                f"- **Preferences:** {result.data.preferences}\n\n"
                f"*(Onboarding complete!)*"
            )
            msg.content = response_content
            await msg.update()
            # Optionally, disable input now that onboarding is done
            # await cl.ChatSettings(inputs=[cl.TextInput(id="chat_input",
            # label="Onboarding complete", initial="")]).send()

        elif isinstance(result.data, str):
            # Conversational response (agent asking for more info)
            msg.content = result.data
            await msg.update()

        else:
            # Handle unexpected result format
            msg.content = f"Sorry, I received an unexpected response type: \
                {type(result.data)}. Content: {result.data}"
            await msg.update()

    except Exception as e:
        # Handle errors during agent execution
        print(f"Error during agent run: {e}")  # Log the error details
        msg.content = (
            "An error occurred while processing your message. Please try again."
        )
        await msg.update()
