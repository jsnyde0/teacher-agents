# app_chainlit.py
# --> Import logging <---
import logging
import os
from typing import List, Union

import chainlit as cl
from dotenv import load_dotenv

# --> Import Agent <---
from pydantic_ai import Agent
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
from src.agents.step_evaluator_agent import create_step_evaluator_agent

# Import Teacher and Step Evaluator agents
from src.agents.teacher_agent import create_teacher_agent

# Load environment variables (for OpenRouter API Key)
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "google/gemini-2.0-flash-lite-001"

# --> Configure logging <---
logging.basicConfig(level=logging.INFO)  # Log INFO and higher messages
logger = logging.getLogger(__name__)


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
    teacher_agent = create_teacher_agent(model)
    step_evaluator_agent = create_step_evaluator_agent(model)

    # Store agents and message history in the user session
    cl.user_session.set("onboarding_agent", onboarding_agent)
    cl.user_session.set("pedagogical_master_agent", pedagogical_master_agent)
    cl.user_session.set("journey_crafter_agent", journey_crafter_agent)  # Store JCA
    cl.user_session.set("teacher_agent", teacher_agent)
    cl.user_session.set("step_evaluator_agent", step_evaluator_agent)
    cl.user_session.set("message_history", [])  # Initialize history
    cl.user_session.set("onboarding_data", None)  # To store OA result
    cl.user_session.set("pedagogical_guidelines", None)  # To store PMA result
    cl.user_session.set("learning_plan", None)
    cl.user_session.set("current_step_index", -1)
    cl.user_session.set("current_stage", "onboarding")  # Track current flow stage

    await cl.Message(
        content="Hello! I'm here to help onboard you. To start, could you tell \
            me a bit about what you already know and what you'd like to learn?"
    ).send()


async def run_pedagogical_master(
    onboarding_data: OnboardingData,
) -> Union[PedagogicalGuidelines, str]:
    """Runs the Pedagogical Master Agent and returns guidelines or error."""
    pma_agent: Agent = cl.user_session.get("pedagogical_master_agent")
    history: List[ModelMessage] = cl.user_session.get("message_history")

    input_prompt = (
        f"Based on the following student onboarding information:\n"
        f"- Current Knowledge (Point A): {onboarding_data.point_a}\n"
        f"- Learning Goal (Point B): {onboarding_data.point_b}\n"
        f"- Preferences: {onboarding_data.preferences}\n\n"
        f"Generate concise pedagogical guidelines for teaching this student."
    )

    try:
        result = await pma_agent.run(input_prompt, message_history=history)
        history.extend(result.all_messages())
        cl.user_session.set("message_history", history)
        if isinstance(result.data, PedagogicalGuidelines):
            return result.data
        else:
            logger.error(f"PMA returned unexpected data type: {type(result.data)}")
            return "Error: Could not generate pedagogical guidelines."
    except Exception as e:
        logger.error(f"Error running PMA: {e}")
        return f"An error occurred while generating pedagogical guidelines: {e}"


async def run_journey_crafter(
    onboarding_data: OnboardingData,
    guidelines: PedagogicalGuidelines,
) -> Union[LearningPlan, str]:
    """Runs the Journey Crafter Agent and returns a learning plan or error."""
    jca_agent: Agent = cl.user_session.get("journey_crafter_agent")
    history: List[ModelMessage] = cl.user_session.get("message_history")

    # Use the actual guideline string from the object
    guideline_str = guidelines.guideline

    input_prompt = (
        f"Based on the student profile and pedagogical guidelines:\n"
        f"- Point A: {onboarding_data.point_a}\n"
        f"- Point B: {onboarding_data.point_b}\n"
        f"- Preferences: {onboarding_data.preferences}\n"
        f"- Pedagogical Guideline: {guideline_str}\n\n"
        f"Create a concise, step-by-step learning plan (as a list of strings, max 5 steps) to get the student from Point A to Point B."
    )

    try:
        result = await jca_agent.run(input_prompt, message_history=history)
        history.extend(result.all_messages())
        cl.user_session.set("message_history", history)
        if isinstance(result.data, LearningPlan):
            return result.data
        else:
            logger.error(f"JCA returned unexpected data type: {type(result.data)}")
            return "Error: Could not generate learning plan."
    except Exception as e:
        logger.error(f"Error running JCA: {e}")
        return f"An error occurred while generating the learning plan: {e}"


# --> Add Helper Function for Teacher Agent <---
async def run_teacher_for_current_step(is_follow_up: bool = False) -> str:
    """Runs the Teacher Agent for the current step and returns the message.

    Args:
        is_follow_up: If True, use a prompt designed for re-engaging the student
                      on the same step after a STAY signal.
    Returns:
        The teaching message string or an error message.
    """
    teacher_agent: Agent = cl.user_session.get("teacher_agent")
    # --> Comment out unused history variable <---
    # history: List[ModelMessage] = cl.user_session.get("message_history") # Not using history for teacher yet
    learning_plan: List[str] = cl.user_session.get("learning_plan")
    current_step_index: int = cl.user_session.get("current_step_index")
    guidelines_obj: PedagogicalGuidelines = cl.user_session.get(
        "pedagogical_guidelines"
    )

    if (
        not all([teacher_agent, learning_plan, guidelines_obj])
        or current_step_index < 0
    ):
        logger.error("Teacher Agent called with missing session data.")
        return "Error: Could not retrieve necessary data to proceed with teaching."

    if current_step_index >= len(learning_plan):
        logger.error("Teacher Agent called with invalid step index.")
        return "Error: Invalid step index."

    # Get the current step description and guideline string
    current_step_description = learning_plan[current_step_index]
    guideline_str = guidelines_obj.guideline

    # --- Construct prompt conditionally ---
    if not is_follow_up:
        # Standard prompt for introducing a step
        input_prompt = (
            f"Start teaching the student according to these instructions:\\n\\n"
            f"Pedagogical Guideline: {guideline_str}\\n"
            f"Current Learning Step: {current_step_description}"
            # Note: We don't provide the full plan or index to the teacher agent itself
            # It only needs to focus on generating the output for the *current* step.
        )
        logger.info(
            f"Running Teacher Agent for step {current_step_index} (initial prompt)."
        )
    else:
        # Follow-up prompt after a STAY signal
        input_prompt = (
            f"The student seems to need more time or has questions about the current learning step.\\n"
            f"Current Learning Step: {current_step_description}\\n"
            f"Pedagogical Guideline: {guideline_str}\\n\\n"
            f"Your Task: Re-engage the student on this *same* step. Do NOT simply repeat the initial introduction. "
            f"Instead, try one of the following based on the guideline: "
            f"ask a different clarifying question, provide a simpler example, offer a focused explanation, "
            f"or gently prompt them to share their thoughts or current attempt. Keep the tone encouraging."
        )
        logger.info(
            f"Running Teacher Agent for step {current_step_index} (follow-up prompt)."
        )

    # --- Run agent ---
    try:
        result = await teacher_agent.run(input_prompt)  # Run without history for now
        if isinstance(result.data, str):
            return result.data
        else:
            logger.error(
                f"Teacher Agent returned unexpected data type: {type(result.data)}"
            )
            return "Error: Could not generate teaching content for this step."
    except Exception as e:
        logger.error(f"Error running Teacher Agent: {e}")
        return f"An error occurred while preparing the teaching content: {e}"


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

    # Append user message to history
    # message_history.append(UserPromptPart(content=message.content)) # REMOVED - run method handles current message

    # --- Stage-Based Logic ---

    if current_stage == "onboarding":
        logger.info("Processing message in onboarding stage...")
        # --> Retrieve onboarding_agent from session <---
        onboarding_agent: Agent = cl.user_session.get("onboarding_agent")
        if not onboarding_agent:
            logger.error("Onboarding Agent not found in session.")
            await cl.Message(
                content="Error: Cannot process onboarding. Please restart chat."
            ).send()
            cl.user_session.set("current_stage", "complete")
            return

        await cl.Message(content="Processing your information...").send()  # Feedback

        # --> Add try/except around agent run <---
        try:
            result = await onboarding_agent.run(
                message.content, message_history=message_history
            )
            # Use the message_history list fetched at the start
            message_history.extend(result.all_messages())

            # --> Log the type of result.data <---
            logger.info(f"Onboarding agent returned type: {type(result.data)}")

            if isinstance(result.data, OnboardingData):
                logger.info("Onboarding complete. Moving to pedagogy stage.")
                onboarding_data = result.data
                cl.user_session.set("onboarding_data", onboarding_data)
                cl.user_session.set("current_stage", "pedagogy")
                await cl.Message(
                    content=(
                        "Thanks! I have your onboarding details. Now generating personalized pedagogical guidelines..."
                    )
                ).send()
                # --- Trigger PMA ---
                guidelines_result = await run_pedagogical_master(onboarding_data)
                if isinstance(guidelines_result, PedagogicalGuidelines):
                    logger.info("PMA successful. Moving to journey crafting stage.")
                    cl.user_session.set("pedagogical_guidelines", guidelines_result)
                    cl.user_session.set("current_stage", "journey_crafting")
                    await cl.Message(
                        content=(
                            "Guidelines created! Now crafting your learning plan..."
                        )
                    ).send()
                    # --- Trigger JCA ---
                    plan_result = await run_journey_crafter(
                        onboarding_data, guidelines_result
                    )
                    if isinstance(plan_result, LearningPlan):
                        logger.info("JCA successful. Moving to teaching stage.")
                        cl.user_session.set("learning_plan", plan_result.steps)
                        cl.user_session.set("current_step_index", 0)
                        cl.user_session.set("current_stage", "teaching")
                        await cl.Message(
                            content=(
                                "Here is your learning plan:\\n"
                                + "\\n".join([f"- {s}" for s in plan_result.steps])
                                + "\\n\\nLet's start with the first step!"
                            )
                        ).send()
                        # --- Trigger Teacher for Step 0 ---
                        teaching_message = await run_teacher_for_current_step()
                        await cl.Message(content=teaching_message).send()
                    else:  # JCA Error
                        logger.error(f"JCA failed: {plan_result}")
                        await cl.Message(
                            content=f"Sorry, I couldn't create your learning plan: {plan_result}"
                        ).send()
                        cl.user_session.set(
                            "current_stage", "complete"
                        )  # End flow on error
                else:  # PMA Error
                    logger.error(f"PMA failed: {guidelines_result}")
                    await cl.Message(
                        content=f"Sorry, I couldn't generate pedagogical guidelines: {guidelines_result}"
                    ).send()
                    cl.user_session.set(
                        "current_stage", "complete"
                    )  # End flow on error
            elif isinstance(result.data, str):
                # Onboarding agent asking for more info
                logger.info("Onboarding agent needs more info.")
                await cl.Message(content=result.data).send()
            else:
                # Handle unexpected onboarding result type
                logger.error(
                    f"Onboarding returned unexpected data type: {type(result.data)}"
                )
                await cl.Message(
                    content="Sorry, something went wrong during onboarding."
                ).send()
                cl.user_session.set("current_stage", "complete")  # End flow

        except Exception as e:
            logger.error(
                f"Error running Onboarding Agent: {e}", exc_info=True
            )  # Add traceback
            await cl.Message(
                content=f"An error occurred during onboarding processing: {e}"
            ).send()
            cl.user_session.set("current_stage", "complete")

    # --- Teaching Stage Logic ---
    elif current_stage == "teaching":
        logger.info(
            f"Processing message in teaching stage (step {cl.user_session.get('current_step_index')})..."
        )
        # --> Retrieve Step Evaluator Agent from session <---
        step_evaluator_agent: Agent = cl.user_session.get("step_evaluator_agent")
        # Use Step Evaluator to decide next action
        if not step_evaluator_agent:
            logger.error("Step Evaluator Agent not found in session.")
            await cl.Message(
                content="Error: Cannot evaluate progress. Please restart the chat."
            ).send()
            cl.user_session.set("current_stage", "complete")
            return

        try:
            eval_result = await step_evaluator_agent.run(message.content)
            evaluation = eval_result.data
            logger.info(f"Step Evaluator result: {evaluation}")

            if evaluation == "PROCEED":
                current_index = cl.user_session.get("current_step_index")
                learning_plan = cl.user_session.get("learning_plan")
                next_index = current_index + 1
                cl.user_session.set("current_step_index", next_index)

                if next_index < len(learning_plan):
                    logger.info(f"Proceeding to step {next_index}")
                    await cl.Message(
                        content="Great! Let's move to the next step."
                    ).send()
                    teaching_message = await run_teacher_for_current_step()
                    await cl.Message(content=teaching_message).send()
                else:
                    logger.info("Learning plan complete.")
                    await cl.Message(
                        content="Congratulations! You've completed the learning plan."
                    ).send()
                    cl.user_session.set("current_stage", "complete")

            elif evaluation == "STAY":
                logger.info(
                    "Student indicates STAY. Re-running teacher for current step."
                )
                # Send acknowledgement and re-run teacher for the *same* step
                await cl.Message(
                    content="Okay, seems like you're still working on this or have questions."
                ).send()
                teaching_message = await run_teacher_for_current_step(is_follow_up=True)
                await cl.Message(content=teaching_message).send()

            elif evaluation == "UNCLEAR":
                logger.info("Student readiness UNCLEAR.")
                # Ask for clarification
                await cl.Message(
                    content="I couldn't quite tell if you're ready to proceed. Please let me know if you have questions, or say 'next' when you're ready to continue."
                ).send()

            else:
                logger.error(f"Step Evaluator returned unexpected value: {evaluation}")
                await cl.Message(
                    content="Sorry, I had trouble understanding your response regarding progression."
                ).send()

        except Exception as e:
            logger.error(f"Error running Step Evaluator Agent: {e}")
            await cl.Message(
                content=f"An error occurred while evaluating your progress: {e}"
            ).send()
            # Optionally decide whether to halt or retry

    elif current_stage == "complete":
        logger.info("Processing message in complete stage.")
        await cl.Message(
            content="Our current learning session is complete. Feel free to start a new chat to learn something else!"
        ).send()

    else:  # Should not happen if stages are managed correctly
        logger.error(f"Reached unexpected stage: {current_stage}")
        await cl.Message(
            content="Sorry, I've encountered an unexpected state. Please restart the chat."
        ).send()

    # Update history at the end of the turn
    cl.user_session.set("message_history", message_history)
