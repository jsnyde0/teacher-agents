# app_chainlit.py
import os
from typing import List

import chainlit as cl
from dotenv import load_dotenv
from pydantic_ai.messages import ModelMessage

# Import agent creation logic and model setup
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# Import JCA components
from src.agents.journey_crafter_agent import (
    LearningPlan,
    create_journey_crafter_agent,
)
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
    journey_crafter_agent = create_journey_crafter_agent(model)  # Create JCA

    # Store agents and message history in the user session
    cl.user_session.set("onboarding_agent", onboarding_agent)
    cl.user_session.set("pedagogical_master_agent", pedagogical_master_agent)
    cl.user_session.set("journey_crafter_agent", journey_crafter_agent)  # Store JCA
    cl.user_session.set("message_history", [])  # Initialize history
    cl.user_session.set("onboarding_data", None)  # To store OA result
    cl.user_session.set("pedagogical_guidelines", None)  # To store PMA result
    cl.user_session.set("current_stage", "onboarding")  # Track current flow stage

    await cl.Message(
        content="Hello! I'm here to help onboard you. To start, could you tell \
            me a bit about what you already know and what you'd like to learn?"
    ).send()


async def run_pedagogical_master(
    onboarding_data: OnboardingData,
) -> PedagogicalGuidelines | str:
    """Helper function to run the PMA and return the Guideline object or error string."""
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
            return pma_result.data  # Return the object
        else:
            error_msg = f"PMA Error: Unexpected result type {type(pma_result.data)}"
            print(error_msg)
            print("----------------------------------------")
            return error_msg
    except Exception as e:
        error_msg = f"Error during pedagogical guideline generation: {e}"
        print(f"PMA Error during run: {e}")
        print("----------------------------------------")
        return error_msg


async def run_journey_crafter(
    onboarding_data: OnboardingData, guidelines: PedagogicalGuidelines
) -> LearningPlan | str:
    """Helper function to run the JCA and return the LearningPlan object or error string."""
    jca = cl.user_session.get("journey_crafter_agent")
    if not jca:
        return "Error: Journey Crafter Agent not initialized."

    print("\n--- Triggering Journey Crafter Agent ---")
    print(f"Input Onboarding Data: {onboarding_data}")
    print(f"Input Guidelines: {guidelines.guideline}")

    jca_input_prompt = (
        f"Create a learning plan based on the following profile and guidelines:\n\n"
        f"**Student Profile:**\n"
        f"- Current Knowledge (Point A): {onboarding_data.point_a}\n"
        f"- Learning Goal (Point B): {onboarding_data.point_b}\n"
        f"- Learning Preferences: {onboarding_data.preferences}\n\n"
        f"**Pedagogical Guideline:** {guidelines.guideline}"
    )

    try:
        jca_result = await jca.run(jca_input_prompt)
        if isinstance(jca_result.data, LearningPlan):
            print(f"JCA Result: {len(jca_result.data.steps)} steps generated.")
            print("--------------------------------------")
            return jca_result.data  # Return the LearningPlan object
        else:
            error_msg = f"JCA Error: Unexpected result type {type(jca_result.data)}"
            print(error_msg)
            print("--------------------------------------")
            return error_msg
    except Exception as e:
        error_msg = f"Error during learning plan generation: {e}"
        print(f"JCA Error during run: {e}")
        print("--------------------------------------")
        return error_msg


@cl.on_message
async def on_message(message: cl.Message):
    """
    Handles incoming user messages, runs the appropriate agent based on state,
    and displays responses.
    """
    current_stage = cl.user_session.get("current_stage", "onboarding")
    message_history: List[ModelMessage] = cl.user_session.get("message_history")

    # If planning is already done, stop
    if current_stage == "complete":
        await cl.Message(
            content="Planning is complete. The next step would be the Teacher Agent."
        ).send()
        return

    # --- Stage: Onboarding ---
    if current_stage == "onboarding":
        onboarding_agent = cl.user_session.get("onboarding_agent")
        if not onboarding_agent:
            await cl.Message(
                content="Onboarding Agent not initialized. Please restart."
            ).send()
            return

        ui_msg = cl.Message(content="")
        await ui_msg.send()

        try:
            oa_result = await onboarding_agent.run(
                message.content, message_history=message_history
            )
            message_history.extend(oa_result.all_messages())
            cl.user_session.set("message_history", message_history)

            if isinstance(oa_result.data, OnboardingData):
                # --- Onboarding Complete -> Store data, transition stage ---
                onboarding_data = oa_result.data
                cl.user_session.set("onboarding_data", onboarding_data)
                cl.user_session.set("current_stage", "pedagogy")  # Move to next stage

                ui_msg.content = (
                    f"Great, thank you! Onboarding complete:\n"
                    f"- **Point A:** {onboarding_data.point_a}\n"
                    f"- **Point B:** {onboarding_data.point_b}\n"
                    f"- **Preferences:** {onboarding_data.preferences}\n\n"
                    f"Now determining teaching approach..."
                )
                await ui_msg.update()

                # --- Stage: Pedagogy (Auto-triggered after Onboarding) ---
                pma_result = await run_pedagogical_master(onboarding_data)

                if isinstance(pma_result, PedagogicalGuidelines):
                    cl.user_session.set(
                        "pedagogical_guidelines", pma_result
                    )  # Store guidelines
                    cl.user_session.set(
                        "current_stage", "journey_crafting"
                    )  # Move to next stage

                    ui_msg.content = (
                        f"Great, thank you! Onboarding complete:\n"
                        f"- **Point A:** {onboarding_data.point_a}\n"
                        f"- **Point B:** {onboarding_data.point_b}\n"
                        f"- **Preferences:** {onboarding_data.preferences}\n\n"
                        f"**Suggested Guideline:** {pma_result.guideline}\n\n"
                        f"Now crafting the learning plan..."
                    )
                    await ui_msg.update()

                    # --- Stage: Journey Crafting (Auto-triggered after Pedagogy) ---
                    jca_result = await run_journey_crafter(onboarding_data, pma_result)

                    if isinstance(jca_result, LearningPlan):
                        plan_steps_text = "\n".join(
                            [
                                f"  {i + 1}. {step}"
                                for i, step in enumerate(jca_result.steps)
                            ]
                        )
                        ui_msg.content = (
                            f"Great, thank you! Onboarding complete:\n"
                            f"- **Point A:** {onboarding_data.point_a}\n"
                            f"- **Point B:** {onboarding_data.point_b}\n"
                            f"- **Preferences:** {onboarding_data.preferences}\n\n"
                            f"**Suggested Guideline:** {pma_result.guideline}\n\n"
                            f"**Proposed Learning Plan:**\n{plan_steps_text}\n\n"
                            f"*(Planning complete! Next: Teacher Agent)*"
                        )
                        await ui_msg.update()
                        cl.user_session.set(
                            "current_stage", "complete"
                        )  # Final stage for now
                    else:  # JCA failed
                        ui_msg.content = f"Onboarding and guideline generation complete, but failed to create learning plan: {jca_result}"
                        await ui_msg.update()
                else:  # PMA failed
                    ui_msg.content = f"Onboarding complete, but failed to determine pedagogical guidelines: {pma_result}"
                    await ui_msg.update()

            elif isinstance(oa_result.data, str):
                # --- Onboarding In Progress ---
                ui_msg.content = oa_result.data  # Display agent's question
                await ui_msg.update()

            else:  # OA returned unexpected type
                ui_msg.content = f"Unexpected response from onboarding agent: {type(oa_result.data)}. Content: {oa_result.data}"
                await ui_msg.update()

        except Exception as e:
            print(f"Error during Onboarding stage run: {e}")
            ui_msg.content = "An error occurred during onboarding. Please try again."
            await ui_msg.update()
    else:
        # Handle cases where stage is somehow invalid (shouldn't happen)
        await cl.Message(
            content=f"Error: Unknown application stage '{current_stage}'. Please restart."
        ).send()
