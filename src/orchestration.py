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
from .agents.journey_crafter_agent import LearningPlan, create_journey_crafter_agent
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
    """Runs the teacher agent for a specific step (initial or follow-up).

    Args:
        teacher_agent: The initialized Teacher agent instance.
        guidelines: The pedagogical guidelines for the session.
        learning_plan_steps: The list of steps in the learning plan.
        current_step_index: The index of the current learning step.
        is_follow_up: True if this is a follow-up after STAY/UNCLEAR, False otherwise.
        last_user_message: The student's last message (required if is_follow_up=True).
        answer_evaluation: The result from the Answer Evaluator (required if is_follow_up=True).

    Returns:
        The teacher's message string or None on error.
    """
    # --- Input Validation --- #
    if not teacher_agent:
        logger.error("Teacher agent instance not provided.")
        return None
    if not guidelines:
        logger.error("Pedagogical guidelines not provided.")
        return None
    if not learning_plan_steps or not (
        0 <= current_step_index < len(learning_plan_steps)
    ):
        logger.error(
            f"Invalid learning plan or step index for Teacher. Index: {current_step_index}, Plan length: {len(learning_plan_steps) if learning_plan_steps else 0}"
        )
        return None
    if is_follow_up and (last_user_message is None or answer_evaluation is None):
        logger.error(
            f"Teacher follow-up called for step {current_step_index} without required context (message or evaluation)."
        )
        return None  # Indicate error

    # --- Prepare Context --- #
    current_step_description = learning_plan_steps[current_step_index]
    guideline_str = guidelines.guideline

    # --- Construct Prompt --- #
    if not is_follow_up:
        # Standard prompt for introducing a step
        input_prompt = (
            f"Start teaching the student according to these instructions:\\n\\n"
            f"Pedagogical Guideline: {guideline_str}\\n"
            f"Current Learning Step: {current_step_description}\\n\\n"
            f"Introduce this step and immediately provide the first piece of instruction, "
            f"an example, or a guiding question about the topic itself to encourage engagement. "
            f"Do NOT ask generic readiness questions like 'Are you ready?'."
        )
        logger.info(
            f"Running Teacher Agent for step {current_step_index} (initial prompt)."
        )
    else:
        # Follow-up prompt incorporating Answer Evaluation
        # answer_evaluation is guaranteed to exist here due to validation above
        input_prompt = f"""**CONTEXT:**
- Current Learning Step Goal: {current_step_description}
- Pedagogical Guideline: {guideline_str}
- Student's Last Message: '{last_user_message}'
- Answer Evaluation: {answer_evaluation.evaluation} (Explanation: {answer_evaluation.explanation})

**Your Task:** Respond conversationally to the 'Student's Last Message' about the 'Current Learning Step Goal'. Your response MUST incorporate the 'Answer Evaluation' and strictly adhere to the 'Pedagogical Guideline'.

1.  **Acknowledge/Feedback:** Briefly acknowledge the student's message, incorporating the provided 'Answer Evaluation' and 'Explanation' naturally into your feedback (e.g., "That's correct because...", "Not quite, the evaluation noted that...", "That's a good question...").
2.  **Guideline-Driven Next Step:** Based on the evaluation and the 'Pedagogical Guideline', provide the *next* small piece of instruction, a clarifying question, a hint, or an example to continue the learning process for the *current* step. The *style* (e.g., Socratic, examples first) MUST match the Guideline.

**IMPORTANT:** Do NOT use step markers. Generate a single, natural conversational response. Do NOT repeat the initial introduction for this step."""

        logger.info(
            f"Running Teacher Agent for step {current_step_index} (follow-up prompt responding to: '{last_user_message[:50]}...' with evaluation: {answer_evaluation.evaluation})"
        )

    # --- Run Agent --- #
    try:
        result = await teacher_agent.run(input_prompt)  # Run without history for now
        if isinstance(result.data, str):
            logger.info(f"Teacher Agent successful for step {current_step_index}.")
            return result.data
        else:
            logger.error(
                f"Teacher Agent returned unexpected data type: {type(result.data)}"
            )
            return None  # Indicate error
    except Exception as e:
        logger.error(f"Error running Teacher Agent: {e}", exc_info=True)
        return None  # Indicate error


# --- Internal Helper Functions for handle_message --- #


async def _handle_teaching_proceed(
    current_state: SessionState,
    agents: Dict[str, Agent],
    guidelines: PedagogicalGuidelines,
    learning_plan_steps: List[str],
    current_step_index: int,
) -> Tuple[str, Dict[str, Any]]:
    """Handles the logic when the Step Evaluator returns PROCEED."""
    next_index = current_step_index + 1
    state_updates = {"current_step_index": next_index}
    reply_message = ""

    if next_index < len(learning_plan_steps):
        logger.info(f"Proceeding to step {next_index}")
        teacher_agent = agents.get("teacher_agent")
        teaching_message = await run_teaching_step(
            teacher_agent=teacher_agent,
            guidelines=guidelines,
            learning_plan_steps=learning_plan_steps,
            current_step_index=next_index,
            is_follow_up=False,
        )
        if teaching_message:
            reply_message = (
                f"Great! Let's move to the next step.\\n\\n---\\n\\n{teaching_message}"
            )
            state_updates["last_teacher_message"] = teaching_message
        else:
            logger.error(f"Failed to get teaching message for step {next_index}.")
            reply_message = (
                "Ok, moving to the next step, but I couldn't prepare the content."
            )
            state_updates["current_stage"] = "error"
    else:
        logger.info("Learning plan complete.")
        reply_message = "Congratulations! You've completed the learning plan."
        state_updates["current_stage"] = "complete"

    return reply_message, state_updates


async def _handle_teaching_stay_unclear(
    current_state: SessionState,
    agents: Dict[str, Agent],
    guidelines: PedagogicalGuidelines,
    learning_plan_steps: List[str],
    current_step_index: int,
    last_teacher_message: str | None,
    user_message: str,
) -> Tuple[str, Dict[str, Any]]:
    """Handles the logic when the Step Evaluator returns STAY or UNCLEAR."""
    state_updates: Dict[str, Any] = {}
    reply_message = ""
    answer_evaluator_agent = agents.get("answer_evaluator_agent")

    if not answer_evaluator_agent:
        logger.error("Answer Evaluator agent missing.")
        reply_message = "Error: Cannot evaluate answer context."
        state_updates["current_stage"] = "error"
        return reply_message, state_updates

    answer_evaluation = await run_answer_evaluation(
        answer_evaluator_agent=answer_evaluator_agent,
        learning_plan_steps=learning_plan_steps,
        current_step_index=current_step_index,
        last_teacher_message=last_teacher_message,
        user_message=user_message,
    )

    if answer_evaluation is None:
        logger.error(
            "Answer Evaluation failed, cannot proceed with teaching step or revision."
        )
        reply_message = "Sorry, I had trouble evaluating your answer."
        # Keep stage teaching, but don't proceed further this turn?
        return reply_message, state_updates  # Return empty updates
    else:
        # --- Potential JCA Revision Logic --- #
        should_revise_plan = answer_evaluation.evaluation in [
            "incorrect",
            "partial",
            "not_applicable",
        ]
        plan_revised_notification = ""

        if should_revise_plan:
            logger.info(
                f"Answer evaluation ({answer_evaluation.evaluation}) suggests plan revision."
            )
            jca_agent = agents.get("journey_crafter_agent")
            onboarding_data = current_state.get("onboarding_data")
            pma_guidelines = current_state.get("pedagogical_guidelines")
            current_plan = current_state.get("learning_plan", [])
            history_before_jca = current_state.get("message_history", [])

            if (
                jca_agent
                and onboarding_data
                and pma_guidelines
                and current_step_index >= 0
                and current_step_index < len(current_plan)
            ):
                current_step_desc = current_plan[current_step_index]
                remaining_steps = current_plan[current_step_index + 1 :]

                revision_prompt = (
                    f"**CRITICAL:** The student is very confused with the step: '{current_step_desc}'.\n"
                    f"Student's message showing confusion: '{user_message}'.\n"
                    f"Evaluation: {answer_evaluation.evaluation} - {answer_evaluation.explanation}\n"
                    f"**TASK:** Propose a *completely different and simpler alternative plan* for the *next 2-3 steps* to overcome this specific confusion. Forget the original remaining steps ({remaining_steps}). Focus ONLY on getting the student past this immediate hurdle before resuming the main goal."
                    f"Original Goal: {onboarding_data.point_b}. Guideline: {pma_guidelines.guideline}\n"
                    f"Respond ONLY with the simple, alternative 2-3 step list."
                )

                try:
                    logger.info("Running JCA for plan revision...")
                    jca_revision_result = await jca_agent.run(
                        revision_prompt, message_history=history_before_jca
                    )
                    # Update history state immediately
                    state_updates["message_history"] = (
                        history_before_jca + jca_revision_result.all_messages()
                    )

                    new_revised_steps = None
                    if isinstance(
                        jca_revision_result.data, LearningPlan
                    ) and isinstance(jca_revision_result.data.steps, list):
                        new_revised_steps = jca_revision_result.data.steps
                    elif isinstance(jca_revision_result.data, list):
                        new_revised_steps = jca_revision_result.data

                    if new_revised_steps is not None:
                        logger.info(
                            f"JCA successfully revised remaining steps: {new_revised_steps}"
                        )
                        plan_prefix = current_plan[: current_step_index + 1]
                        # Update learning_plan in the state_updates dict
                        state_updates["learning_plan"] = plan_prefix + new_revised_steps
                        plan_revised_notification = "\n*(Note: I've adjusted the upcoming plan based on your feedback.)*"
                    else:
                        logger.warning(
                            f"JCA ran for revision but didn't return valid list. Type: {type(jca_revision_result.data)}. Keeping original plan."
                        )
                except Exception as jca_err:
                    logger.error(
                        f"Error running JCA for plan revision: {jca_err}", exc_info=True
                    )
                    # Keep original plan on error
            else:
                logger.warning(
                    "Could not attempt plan revision due to missing JCA agent or context."
                )
        # --- End of JCA Revision Logic --- #

        # Now, get the follow-up teaching message for the CURRENT step
        teacher_agent = agents.get("teacher_agent")
        # Use the potentially updated plan if revision occurred, otherwise original
        final_plan_for_teacher = state_updates.get(
            "learning_plan", current_state.get("learning_plan")
        )

        teaching_message = await run_teaching_step(
            teacher_agent=teacher_agent,
            guidelines=guidelines,
            learning_plan_steps=final_plan_for_teacher,
            current_step_index=current_step_index,
            is_follow_up=True,
            last_user_message=user_message,
            answer_evaluation=answer_evaluation,
        )
        if teaching_message:
            reply_message = (
                teaching_message + plan_revised_notification
            )  # Append revision note
            state_updates["last_teacher_message"] = teaching_message
        else:
            logger.error("Failed to get teaching follow-up message.")
            reply_message = "Sorry, I couldn't prepare the follow-up message."
            state_updates["current_stage"] = "error"

    return reply_message, state_updates


async def _handle_teaching_error(
    evaluation_result: str | None,
) -> Tuple[str, Dict[str, Any]]:
    """Handles logic when step evaluation fails or returns unexpected value."""
    state_updates = {}
    if evaluation_result is None:
        logger.error("Step Evaluation failed.")
        reply_message = "Sorry, I had trouble evaluating your progress."
    else:  # Unexpected value
        logger.error(f"Step Evaluator returned unexpected value: {evaluation_result}")
        reply_message = "Sorry, I had trouble processing the evaluation."
        state_updates["current_stage"] = "error"

    return reply_message, state_updates


# --- Control Flow Orchestration --- #
async def handle_message(
    session_state: SessionState, user_message: str
) -> Tuple[str, SessionState]:
    """Handles a user message based on the current session state.

    This is the main entry point for processing a message after agents are initialized.
    It determines the current stage, calls the appropriate lower-level
    orchestration functions, updates the state, and returns the user reply.

    Args:
        session_state: The current state dictionary for the session.
        user_message: The message received from the user.

    Returns:
        A tuple containing:
        - The reply message string to be sent to the user.
        - The *new*, updated session state dictionary.
    """
    # 1. Copy state to modify
    current_state = session_state.copy()

    # 2. Get common info from current state
    current_stage = current_state.get("current_stage", "onboarding")
    agents = current_state.get("agents", {})
    message_history = current_state.get("message_history", [])
    reply_message = "An error occurred processing your message."  # Default error reply
    state_updates: Dict[str, Any] = {}

    logger.info(f"Handling message for stage: {current_stage}")

    # --- Onboarding Stage --- #
    if current_stage == "onboarding":
        onboarding_agent = agents.get("onboarding_agent")
        if not onboarding_agent:
            logger.error("Onboarding Agent not found in state.")
            reply_message = "Error: Onboarding agent missing."
            state_updates["current_stage"] = "error"
        else:
            oa_result, updated_history = await run_onboarding_step(
                onboarding_agent=onboarding_agent,
                user_message=user_message,
                message_history=message_history,
            )
            # Update history immediately
            state_updates["message_history"] = updated_history

            if isinstance(oa_result, OnboardingData):
                logger.info("Onboarding complete. Running post-onboarding pipeline...")
                state_updates["onboarding_data"] = oa_result
                pma_agent = agents.get("pedagogical_master_agent")
                jca_agent = agents.get("journey_crafter_agent")

                if not pma_agent or not jca_agent:
                    logger.error("PMA or JCA agent missing.")
                    reply_message = "Error: Planning agents not available."
                    state_updates["current_stage"] = "error"
                else:
                    (
                        guidelines,
                        plan_steps,
                        final_history,
                        pipeline_error,
                    ) = await run_post_onboarding_pipeline(
                        pma_agent=pma_agent,
                        jca_agent=jca_agent,
                        onboarding_data=oa_result,
                        message_history=updated_history,  # Use history from onboarding step
                    )
                    # Update history again
                    state_updates["message_history"] = final_history

                    if pipeline_error:
                        logger.error(f"Pipeline failed: {pipeline_error}")
                        reply_message = (
                            f"Sorry, I couldn't complete the planning: {pipeline_error}"
                        )
                        state_updates["current_stage"] = "error"
                    elif guidelines and plan_steps:
                        logger.info("Pipeline successful. Moving to teaching.")
                        state_updates["pedagogical_guidelines"] = guidelines
                        state_updates["learning_plan"] = plan_steps
                        state_updates["current_step_index"] = 0
                        state_updates["current_stage"] = "teaching"

                        # Trigger Teacher for Step 0
                        teacher_agent = agents.get("teacher_agent")
                        initial_teaching_message = await run_teaching_step(
                            teacher_agent=teacher_agent,
                            guidelines=guidelines,
                            learning_plan_steps=plan_steps,
                            current_step_index=0,
                            is_follow_up=False,
                        )

                        if initial_teaching_message:
                            state_updates["last_teacher_message"] = (
                                initial_teaching_message
                            )
                            reply_message = (
                                f"**Guideline:** {guidelines.guideline}\\n\\n"
                                f"Planning complete! Let's start with the first step.\\n\\n---\\n\\n{initial_teaching_message}"
                            )
                        else:
                            logger.error("Failed to get initial teaching message.")
                            reply_message = "Planning complete, but failed to prepare the first teaching step."
                            state_updates["current_stage"] = "error"
                    else:
                        logger.error("Pipeline returned unexpected state.")
                        reply_message = (
                            "Sorry, an unexpected error occurred during planning."
                        )
                        state_updates["current_stage"] = "error"

            elif isinstance(oa_result, str):
                logger.info("Onboarding agent needs more info.")
                reply_message = oa_result
                # Stage remains 'onboarding', no state updates needed beyond history
            else:  # Onboarding failed (returned None)
                logger.error("Onboarding step failed.")
                reply_message = "Sorry, something went wrong during onboarding."
                state_updates["current_stage"] = "error"

    # --- Teaching Stage --- #
    elif current_stage == "teaching":
        # Retrieve necessary state and agents
        step_evaluator_agent = agents.get("step_evaluator_agent")
        last_teacher_message = current_state.get("last_teacher_message")
        learning_plan_steps = current_state.get("learning_plan")
        current_step_index = current_state.get("current_step_index", -1)
        guidelines = current_state.get("pedagogical_guidelines")
        teacher_agent = agents.get("teacher_agent")  # Needed by helpers

        # Check for required data for teaching stage
        if (
            not step_evaluator_agent
            or not learning_plan_steps
            or current_step_index < 0
            or not guidelines
            or not teacher_agent
        ):
            logger.error("Missing required data/agents for teaching stage.")
            reply_message = "Error: Session state is incomplete for teaching."
            state_updates["current_stage"] = "error"
        else:
            # Run step evaluation
            evaluation = await run_step_evaluation(
                step_evaluator_agent=step_evaluator_agent,
                last_teacher_message=last_teacher_message,
                user_message=user_message,
            )

            # --> Log the raw evaluation result <---
            logger.info(f"Step Evaluator raw result: {evaluation}")

            # Delegate to helper based on evaluation result
            if evaluation == "PROCEED":
                reply_message, updates = await _handle_teaching_proceed(
                    current_state=current_state,
                    agents=agents,
                    guidelines=guidelines,
                    learning_plan_steps=learning_plan_steps,
                    current_step_index=current_step_index,
                )
                state_updates.update(updates)
            elif evaluation == "STAY" or evaluation == "UNCLEAR":
                reply_message, updates = await _handle_teaching_stay_unclear(
                    current_state=current_state,
                    agents=agents,
                    guidelines=guidelines,
                    learning_plan_steps=learning_plan_steps,
                    current_step_index=current_step_index,
                    last_teacher_message=last_teacher_message,
                    user_message=user_message,
                )
                state_updates.update(updates)
            else:  # Includes None (error) or unexpected values
                reply_message, updates = await _handle_teaching_error(
                    evaluation_result=evaluation  # Pass the result for logging
                )
                state_updates.update(updates)

    # --- Complete Stage --- #
    elif current_stage == "complete":
        logger.info("Handling message in complete stage.")
        reply_message = "Our current learning session is complete. Feel free to start a new chat to learn something else!"
        # No state updates needed, stage remains 'complete'

    # --- Error / Unexpected Stage --- #
    else:
        if current_stage != "error":
            logger.error(f"Reached unexpected stage in handle_message: {current_stage}")
        reply_message = (
            "Sorry, an unexpected error occurred. Please try starting a new chat."
        )
        state_updates["current_stage"] = "error"

    # 4. Apply all accumulated state updates to the state dictionary
    current_state.update(state_updates)

    # 5. Return the final reply and the *modified* current_state dictionary
    logger.info(
        f"handle_message complete. New stage: {current_state.get('current_stage')}"
    )
    return reply_message, current_state


# Additional helper functions might be needed for state updates or prompt formatting.
