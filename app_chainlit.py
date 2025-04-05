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

# Import PMA components
from src.agents.pedagogical_master_agent import (
    PedagogicalGuidelines,
    create_pedagogical_master_agent,
)

# Load environment variables (for OpenRouter API Key)
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "google/gemini-flash-1.5"  # Or your preferred model


@cl.on_chat_start
async def on_chat_start():
    """
    Initializes agents and conversation history when a new chat session starts.
    """
    if not OPENROUTER_API_KEY:
        await cl.Message(
            content="Error: OPENROUTER_API_KEY not found in environment variables."
        ).send()
        return

    # Configure the model (can be shared by agents)
    model = OpenAIModel(
        MODEL_NAME,
        provider=OpenAIProvider(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
        ),
    )

    # Create the agent instances
    onboarding_agent = create_onboarding_agent(model)
    pedagogical_master_agent = create_pedagogical_master_agent(model)

    # Store agents and message history in the user session
    cl.user_session.set("onboarding_agent", onboarding_agent)
    cl.user_session.set("pedagogical_master_agent", pedagogical_master_agent)
    cl.user_session.set("message_history", [])  # Initialize history
    cl.user_session.set("onboarding_complete", False)  # Flag to track state

    await cl.Message(
        content="Hello! I'm here to help onboard you. To start, could you tell \
            me a bit about what you already know and what you'd like to learn?"
    ).send()


async def run_pedagogical_master(onboarding_data: OnboardingData) -> str:
    """Helper function to run the PMA and return the guideline string or error."""
    pma = cl.user_session.get("pedagogical_master_agent")
    if not pma:
        return "Error: Pedagogical Master Agent not initialized."

    print("\n--- Triggering Pedagogical Master Agent ---")
    print(f"Input Onboarding Data: {onboarding_data}")

    # Construct the input prompt for the PMA
    pma_input_prompt = (
        f"Determine pedagogical guidelines based on the following student profile:\n"
        f"Current Knowledge (Point A): {onboarding_data.point_a}\n"
        f"Learning Goal (Point B): {onboarding_data.point_b}\n"
        f"Learning Preferences: {onboarding_data.preferences}"
    )

    try:
        pma_result = await pma.run(pma_input_prompt)
        if isinstance(pma_result.data, PedagogicalGuidelines):
            print(f"PMA Result: {pma_result.data.guideline}")
            print("----------------------------------------")
            return pma_result.data.guideline
        else:
            print(f"PMA Error: Unexpected result type {type(pma_result.data)}")
            print("----------------------------------------")
            return "Error: Could not determine pedagogical guidelines."
    except Exception as e:
        print(f"PMA Error during run: {e}")
        print("----------------------------------------")
        return f"Error during pedagogical guideline generation: {e}"


@cl.on_message
async def on_message(message: cl.Message):
    """
    Handles incoming user messages, runs the appropriate agent based on state,
    and displays responses.
    """
    onboarding_agent = cl.user_session.get("onboarding_agent")
    message_history: List[ModelMessage] = cl.user_session.get("message_history")
    onboarding_complete = cl.user_session.get("onboarding_complete", False)

    if not onboarding_agent:
        await cl.Message(
            content="Agent not initialized. Please restart the chat."
        ).send()
        return

    # If onboarding is already done, don't run OA again (basic state handling)
    if onboarding_complete:
        await cl.Message(
            content="Onboarding is complete. Next steps would involve the Journey Crafter/Teacher agents (not implemented yet)."
        ).send()
        return

    # Display a thinking indicator
    ui_msg = cl.Message(content="")
    await ui_msg.send()

    try:
        # Run the ONBOARDING agent
        oa_result = await onboarding_agent.run(
            message.content, message_history=message_history
        )

        # Update the history
        message_history.extend(oa_result.all_messages())
        cl.user_session.set("message_history", message_history)

        # Process the result from OA
        if isinstance(oa_result.data, OnboardingData):
            # --- Onboarding Complete - HANDOFF TO PMA ---
            cl.user_session.set("onboarding_complete", True)
            onboarding_data = oa_result.data

            # Update UI temporarily before running PMA
            ui_msg.content = (
                f"Great, thank you! Based on our conversation, here's what I've gathered:\n"
                f"- **Point A:** {onboarding_data.point_a}\n"
                f"- **Point B:** {onboarding_data.point_b}\n"
                f"- **Preferences:** {onboarding_data.preferences}\n\n"
                f"Now, let me determine the best teaching approach..."
            )
            await ui_msg.update()

            # Run PMA (can take a moment)
            guideline = await run_pedagogical_master(onboarding_data)

            # Final UI update after PMA completes
            final_response = (
                f"Great, thank you! Based on our conversation, here's what I've gathered:\n"
                f"- **Point A:** {onboarding_data.point_a}\n"
                f"- **Point B:** {onboarding_data.point_b}\n"
                f"- **Preferences:** {onboarding_data.preferences}\n\n"
                f"**Suggested Teaching Guideline:** {guideline}\n\n"
                f"*(Onboarding complete! Next step: Journey Crafter)*"
            )
            ui_msg.content = final_response
            await ui_msg.update()

        elif isinstance(oa_result.data, str):
            # --- Onboarding In Progress ---
            ui_msg.content = oa_result.data  # Display agent's question
            await ui_msg.update()

        else:
            # Handle unexpected result format from OA
            ui_msg.content = f"Sorry, I received an unexpected response type from the onboarding agent: {type(oa_result.data)}. Content: {oa_result.data}"
            await ui_msg.update()

    except Exception as e:
        # Handle errors during agent execution
        print(f"Error during agent run: {e}")  # Log the error details
        ui_msg.content = (
            "An error occurred while processing your message. Please try again."
        )
        await ui_msg.update()
