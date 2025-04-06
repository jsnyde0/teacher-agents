# app_chainlit.py
# --> Import logging <---
import logging
import os

import chainlit as cl

# --> Import httpx <---
from dotenv import load_dotenv

# --> Import Agent <---
# --> Import orchestration functions <---
from src import orchestration

# --> Import AnswerEvaluationResult model <---
# Import JCA components

# Import PMA components

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
    Initializes agents and conversation history using the orchestration module.
    """
    logger.info("Chat start: Initializing session...")
    if not OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY not found.")
        await cl.Message(
            content="Error: OPENROUTER_API_KEY not found in environment variables."
        ).send()
        return

    # --> Call orchestration.initialize_agents <---
    agents = await orchestration.initialize_agents(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        model_name=MODEL_NAME,
    )

    if not agents:
        logger.error("Agent initialization failed.")
        await cl.Message(content="Error: Failed to initialize agents.").send()
        return

    # Store agents dictionary and initial state in the user session
    # --> Store the whole agents dictionary <---
    cl.user_session.set("agents", agents)
    logger.info(f"Stored agent dictionary in session with keys: {list(agents.keys())}")

    cl.user_session.set("message_history", [])  # Initialize history
    cl.user_session.set("onboarding_data", None)  # To store OA result
    cl.user_session.set("pedagogical_guidelines", None)  # To store PMA result
    cl.user_session.set("learning_plan", None)  # Stores List[str] of steps
    cl.user_session.set("current_step_index", -1)
    cl.user_session.set("current_stage", "onboarding")  # Track current flow stage
    cl.user_session.set("last_teacher_message", None)  # Add last_teacher_message state

    await cl.Message(
        content="Hello! I'm here to help onboard you. To start, could you tell \
            me a bit about what you already know and what you'd like to learn?"
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """
    Handles incoming user messages by calling the central orchestration logic.
    Retrieves current state, calls handle_message, updates session, and replies.
    """
    # 1. Retrieve all necessary current state from cl.user_session
    current_session_state = {
        "agents": cl.user_session.get("agents", {}),
        "current_stage": cl.user_session.get("current_stage", "onboarding"),
        "message_history": cl.user_session.get("message_history", []),
        "onboarding_data": cl.user_session.get("onboarding_data"),
        "pedagogical_guidelines": cl.user_session.get("pedagogical_guidelines"),
        "learning_plan": cl.user_session.get("learning_plan"),
        "current_step_index": cl.user_session.get("current_step_index", -1),
        "last_teacher_message": cl.user_session.get("last_teacher_message"),
    }

    # Check if agents are missing (critical error from initialization)
    if not current_session_state.get("agents"):
        logger.error(
            "Agents dictionary not found in session. Aborting message processing."
        )
        await cl.Message(
            content="Critical Error: Session agents missing. Please restart."
        ).send()
        return

    logger.info(
        f"on_message: Passing state to handle_message (Stage: {current_session_state['current_stage']})"
    )

    # 2. Call the central orchestrator
    reply_message, new_session_state = await orchestration.handle_message(
        session_state=current_session_state, user_message=message.content
    )

    # 3. Update the Chainlit session with the new state returned by the orchestrator
    logger.info(
        f"on_message: Updating session with new state (New Stage: {new_session_state.get('current_stage')}) "
    )
    for key, value in new_session_state.items():
        cl.user_session.set(key, value)

    # 4. Send the reply to the user
    await cl.Message(content=reply_message).send()

    # --- Remove old stage-based logic --- #
    # if current_stage == "onboarding":
    #    ... (old logic removed)
    # elif current_stage == "teaching":
    #    ... (old logic removed)
    # elif current_stage == "complete":
    #    ... (old logic removed)
    # else:
    #    ... (old logic removed)
