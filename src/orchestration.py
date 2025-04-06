# src/orchestration.py
"""Core logic for orchestrating agent interactions and managing conversation flow.

This module contains state-independent functions that can be called by different
application entry points (e.g., Chainlit app, FastAPI API) to ensure consistent
behavior.
"""

import logging
from typing import Any, Dict, List, Tuple, Union

# Import necessary Pydantic models and types
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# Import agent creation functions and models
from .agents.answer_evaluator_agent import (
    AnswerEvaluationResult,
    create_answer_evaluator_agent,
)
from .agents.journey_crafter_agent import create_journey_crafter_agent
from .agents.onboarding_agent import OnboardingData, create_onboarding_agent
from .agents.pedagogical_master_agent import (
    PedagogicalGuidelines,
    create_pedagogical_master_agent,
)
from .agents.step_evaluator_agent import create_step_evaluator_agent
from .agents.teacher_agent import create_teacher_agent

# Placeholder for type hinting the session state dictionary
# In a real application, this might be a Pydantic model for better validation
SessionState = Dict[str, Any]

logger = logging.getLogger(__name__)

# --- Implemented Functions ---


async def initialize_agents(
    api_key: str, base_url: str, model_name: str
) -> Dict[str, Agent]:
    """Creates and returns all necessary agent instances."""
    if not api_key:
        logger.error("API key not provided for agent initialization.")
        # In a real app, might raise an error or return an empty dict
        # For now, log and return empty to signal failure upstream
        return {}

    logger.info(f"Initializing agents with model: {model_name}")
    try:
        model = OpenAIModel(
            model_name,
            provider=OpenAIProvider(base_url=base_url, api_key=api_key),
        )

        agents = {
            "onboarding_agent": create_onboarding_agent(model),
            "pedagogical_master_agent": create_pedagogical_master_agent(model),
            "journey_crafter_agent": create_journey_crafter_agent(model),
            "teacher_agent": create_teacher_agent(model),
            "step_evaluator_agent": create_step_evaluator_agent(model),
            "answer_evaluator_agent": create_answer_evaluator_agent(model),
        }
        logger.info("All agents initialized successfully.")
        return agents
    except Exception as e:
        logger.error(f"Failed to initialize agents: {e}", exc_info=True)
        return {}


async def run_step_evaluation(
    step_evaluator_agent: Agent,
    last_teacher_message: str | None,
    user_message: str,
) -> Union[str, None]:
    """Runs the step evaluator.

    Args:
        step_evaluator_agent: The initialized Step Evaluator agent instance.
        last_teacher_message: The previous message sent by the teacher.
        user_message: The student's latest message.

    Returns:
        The evaluation string ("PROCEED", "STAY", "UNCLEAR") or None on error.
    """
    if not step_evaluator_agent:
        logger.error("Step Evaluator agent instance not provided.")
        return None

    evaluator_input_prompt = (
        f"Teacher's Last Message: {last_teacher_message or '(No previous teacher message)'}\\n"
        f"Student's Response: {user_message}"
    )
    logger.info(
        f"Running Step Evaluator with context: [Teacher: '{last_teacher_message}', Student: '{user_message}']"
    )

    try:
        eval_result = await step_evaluator_agent.run(evaluator_input_prompt)
        evaluation = eval_result.data

        if evaluation in ["PROCEED", "STAY", "UNCLEAR"]:
            logger.info(f"Step Evaluator returned: {evaluation}")
            return evaluation
        else:
            logger.error(f"Step Evaluator returned unexpected value: {evaluation}")
            return None  # Indicate error/unexpected value
    except Exception as e:
        logger.error(f"Error running Step Evaluator Agent: {e}", exc_info=True)
        return None


async def run_answer_evaluation(
    answer_evaluator_agent: Agent,
    learning_plan_steps: List[str],  # Pass only steps list
    current_step_index: int,
    last_teacher_message: str | None,
    user_message: str,
) -> Union[AnswerEvaluationResult, None]:
    """Runs the answer evaluator.

    Args:
        answer_evaluator_agent: The initialized Answer Evaluator agent instance.
        learning_plan_steps: The list of steps in the learning plan.
        current_step_index: The index of the current learning step.
        last_teacher_message: The previous message sent by the teacher.
        user_message: The student's latest message.

    Returns:
        The AnswerEvaluationResult object or None on error.
    """
    if not answer_evaluator_agent:
        logger.error("Answer Evaluator agent instance not provided.")
        return None
    if not learning_plan_steps or not (
        0 <= current_step_index < len(learning_plan_steps)
    ):
        logger.error(
            f"Invalid learning plan or step index for Answer Evaluator. Index: {current_step_index}, Plan length: {len(learning_plan_steps) if learning_plan_steps else 0}"
        )
        return None

    current_step_description = learning_plan_steps[current_step_index]

    answer_eval_prompt = (
        f"Current Learning Step Goal: {current_step_description}\\n"
        f"Teacher's Last Instruction/Question: {last_teacher_message or '(None)'}\\n"
        f"Student's Response: {user_message}"
    )
    logger.info("Running Answer Evaluator...")

    try:
        answer_eval_result_obj = await answer_evaluator_agent.run(answer_eval_prompt)

        if isinstance(answer_eval_result_obj.data, AnswerEvaluationResult):
            logger.info(
                f"Answer Evaluation Result: {answer_eval_result_obj.data.evaluation} - {answer_eval_result_obj.data.explanation}"
            )
            return answer_eval_result_obj.data
        else:
            logger.error(
                f"Answer Evaluator returned unexpected type: {type(answer_eval_result_obj.data)}"
            )
            return None  # Indicate error/unexpected type
    except Exception as e:
        logger.error(f"Error running Answer Evaluator Agent: {e}", exc_info=True)
        return None


# --- Placeholder Functions (Still To Be Implemented) ---


async def run_onboarding_step(
    onboarding_agent: Agent,
    user_message: str,
    message_history: List[ModelMessage],
) -> Tuple[Union[OnboardingData, str, None], List[ModelMessage]]:
    """Runs a single step of the onboarding process.

    Args:
        onboarding_agent: The initialized Onboarding agent instance.
        user_message: The student's latest message.
        message_history: The current conversation history.

    Returns:
        A tuple containing:
        - The result: OnboardingData if complete, str if more info needed, None on error.
        - The updated message history.
    """
    if not onboarding_agent:
        logger.error("Onboarding agent instance not provided.")
        return None, message_history  # Return None for data, original history

    logger.info("Running Onboarding Agent step...")
    try:
        result = await onboarding_agent.run(
            user_message, message_history=message_history
        )
        updated_history = message_history + result.all_messages()

        if isinstance(result.data, OnboardingData):
            logger.info("Onboarding complete.")
            return result.data, updated_history
        elif isinstance(result.data, str):
            logger.info("Onboarding agent requires more info.")
            return result.data, updated_history
        else:
            logger.error(
                f"Onboarding agent returned unexpected data type: {type(result.data)}"
            )
            return None, updated_history  # Return None for data, updated history

    except Exception as e:
        logger.error(f"Error running Onboarding Agent: {e}", exc_info=True)
        return None, message_history  # Return None for data, original history on error


async def run_post_onboarding_pipeline(
    pma_agent: Agent,
    jca_agent: Agent,
    onboarding_data: OnboardingData,
    message_history: List[ModelMessage],
) -> Tuple[
    Union[PedagogicalGuidelines, None],
    Union[List[str], None],  # Return list of steps from LearningPlan
    List[ModelMessage],
    Union[str, None],  # Optional error message string
]:
    """Runs PMA and JCA after onboarding is complete.

    Args:
        pma_agent: Initialized Pedagogical Master Agent.
        jca_agent: Initialized Journey Crafter Agent.
        onboarding_data: Completed onboarding data.
        message_history: Current conversation history.

    Returns:
        A tuple containing:
        - PedagogicalGuidelines object or None on error.
        - List of learning plan steps or None on error.
        - Updated message history.
        - Error message string or None if successful.
    """
    guidelines: PedagogicalGuidelines | None = None
    plan_steps: List[str] | None = None
    error_message: str | None = None
    current_history = message_history  # Start with incoming history

    # --- Run PMA --- #
    if not pma_agent:
        logger.error("PMA agent instance not provided for post-onboarding pipeline.")
        return None, None, current_history, "PMA agent not available."

    logger.info("Running Pedagogical Master Agent...")
    pma_input_prompt = (
        f"Based on the following student onboarding information:\\n"
        f"- Current Knowledge (Point A): {onboarding_data.point_a}\\n"
        f"- Learning Goal (Point B): {onboarding_data.point_b}\\n"
        f"- Preferences: {onboarding_data.preferences}\\n\\n"
        f"Generate concise pedagogical guidelines for teaching this student."
    )
    try:
        pma_result = await pma_agent.run(
            pma_input_prompt, message_history=current_history
        )
        current_history = current_history + pma_result.all_messages()
        if isinstance(pma_result.data, PedagogicalGuidelines):
            guidelines = pma_result.data
            logger.info("PMA successful.")
        else:
            logger.error(f"PMA returned unexpected data type: {type(pma_result.data)}")
            error_message = "Error: Could not generate pedagogical guidelines."
            return guidelines, plan_steps, current_history, error_message
    except Exception as e:
        logger.error(f"Error running PMA: {e}", exc_info=True)
        error_message = (
            f"An error occurred while generating pedagogical guidelines: {e}"
        )
        return guidelines, plan_steps, current_history, error_message

    # --- Run JCA (only if PMA succeeded) --- #
    if not guidelines:
        # Error occurred in PMA, already logged and error_message set.
        return guidelines, plan_steps, current_history, error_message

    if not jca_agent:
        logger.error("JCA agent instance not provided for post-onboarding pipeline.")
        return guidelines, None, current_history, "JCA agent not available."

    logger.info("Running Journey Crafter Agent...")
    jca_input_prompt = (
        f"Based on the student profile and pedagogical guidelines:\\n"
        f"- Point A: {onboarding_data.point_a}\\n"
        f"- Point B: {onboarding_data.point_b}\\n"
        f"- Preferences: {onboarding_data.preferences}\\n"
        f"- Pedagogical Guideline: {guidelines.guideline}\\n\\n"
        f"Create a concise, step-by-step learning plan (as a list of strings, max 5 steps) to get the student from Point A to Point B."
    )
    try:
        # Note: JCA uses the history updated by PMA
        jca_result = await jca_agent.run(
            jca_input_prompt, message_history=current_history
        )
        current_history = current_history + jca_result.all_messages()
        # Check if the result is LearningPlan and extract steps
        if hasattr(jca_result.data, "steps") and isinstance(
            jca_result.data.steps, list
        ):
            plan_steps = jca_result.data.steps
            logger.info("JCA successful.")
        else:
            logger.error(
                f"JCA returned unexpected data type or format: {type(jca_result.data)}"
            )
            error_message = "Error: Could not generate learning plan steps."
            return guidelines, plan_steps, current_history, error_message
    except Exception as e:
        logger.error(f"Error running JCA: {e}", exc_info=True)
        error_message = f"An error occurred while generating the learning plan: {e}"
        return guidelines, plan_steps, current_history, error_message

    # --- Return results --- #
    return (
        guidelines,
        plan_steps,
        current_history,
        error_message,
    )  # error_message is None if successful


async def run_teaching_step(
    teacher_agent: Agent,
    guidelines: PedagogicalGuidelines,
    learning_plan_steps: List[str],  # Pass only steps list
    current_step_index: int,
    is_follow_up: bool = False,
    last_user_message: str | None = None,
    answer_evaluation: AnswerEvaluationResult | None = None,
) -> Union[str, None]:
    """Runs the teacher agent for a specific step (initial or follow-up)."""
    # Logic from run_teacher_for_current_step
    # Returns the teacher's message string or None on error.
    pass


# Additional helper functions might be needed for state updates or prompt formatting.
