# src/agents/teacher_agent.py
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

from .journey_crafter_agent import LearningPlan

# Import related data models
from .onboarding_agent import OnboardingData
from .pedagogical_master_agent import PedagogicalGuidelines


# --- Data Model for Output ---
class TeacherResponse(BaseModel):
    """Teaching response from the Teacher Agent."""

    content: str = Field(
        ...,
        description="The teaching content, explanations, examples, and questions for the student.",
    )
    current_step_index: int = Field(
        ..., description="Index of the current learning plan step being taught."
    )
    completed: bool = Field(
        False,
        description="Whether this step is considered completed and should move to the next.",
    )


# --- Agent Definition ---
def create_teacher_agent(model: OpenAIModel) -> Agent:
    """Creates the Teacher Agent instance.

    This agent is responsible for presenting a learning step to the student
    and providing follow-up guidance based on pedagogical guidelines and the
    student's response.

    Args:
        model: The PydanticAI model instance (e.g., OpenAIModel configured for OpenRouter).

    Returns:
        An initialized Agent instance for the Teacher.
    """
    # For MVP, the agent just initiates the conversation for the *first* step.
    # It receives the step description and pedagogical guideline.
    # It needs to output conversational text (string).
    return Agent(
        model=model,
        result_type=TeacherResponse,
        system_prompt=(
            "You are an expert Teacher Agent. Your task is to interact with the student, "
            "executing the learning plan step-by-step while adhering to the pedagogical guidelines. "
            "You will focus on the current step only, providing explanations, examples, and asking questions.\n\n"
            "**Input Analysis:**\n"
            "You'll receive:\n"
            "1. The student's OnboardingData (Point A, Point B, Preferences)\n"
            "2. The PedagogicalGuidelines for teaching approach\n"
            "3. The LearningPlan with steps to follow\n"
            "4. The current step index to focus on\n"
            "5. The student's message/question\n"
            "6. The conversation history\n\n"
            "**Teaching Approach:**\n"
            "1. **Focus on the current step only**. If the student asks about topics in future steps, gently redirect them.\n"
            "2. **Follow the pedagogical guidelines** carefully in your teaching style.\n"
            "3. **Provide clear explanations** with appropriate depth based on the student's knowledge level.\n"
            "4. **Include relevant examples** that connect to the student's goal when possible.\n"
            "5. **Ask checking questions** to verify understanding.\n"
            "6. **Respond to student questions** related to the current topic in-depth.\n"
            "7. **Determine when to advance** to the next step based on student demonstrations of understanding.\n\n"
            "**Output Format:**\n"
            "Respond with a TeacherResponse that includes:\n"
            "1. `content`: Your teaching explanation, examples, and questions.\n"
            "2. `current_step_index`: The step index you're currently teaching.\n"
            "3. `completed`: Whether this step is complete and should advance to the next step.\n\n"
            "**Important Rules:**\n"
            "- Your teaching must be accurate, clear, and directly relevant to the current step.\n"
            "- Don't introduce concepts that are too advanced for the student's current level.\n"
            "- Use appropriate examples based on the student's background and preferences.\n"
            "- Only mark a step as completed when the student has demonstrated sufficient understanding.\n"
            "- If the student seems confused, provide additional explanations rather than advancing.\n"
            "- If at the final step, set `completed` to true only when the overall learning goal appears achieved.\n"
        ),
    )


# Function to prepare teacher agent input from session state
def prepare_teacher_input(
    onboarding_data: OnboardingData,
    guidelines: PedagogicalGuidelines,
    learning_plan: LearningPlan,
    current_step_index: int,
    user_message: str,
) -> str:
    """
    Creates a formatted input prompt for the teacher agent based on the current session state.
    """
    # Get the current step content
    if current_step_index >= len(learning_plan.steps):
        current_step = "FINAL REVIEW: Review all previous steps and check for overall understanding."
    else:
        current_step = learning_plan.steps[current_step_index]

    # Create the combined input prompt
    return (
        f"Based on the following information, teach the student about the current step:\n\n"
        f"**Student Profile:**\n"
        f"- Current Knowledge (Point A): {onboarding_data.point_a}\n"
        f"- Learning Goal (Point B): {onboarding_data.point_b}\n"
        f"- Learning Preferences: {onboarding_data.preferences}\n\n"
        f"**Pedagogical Guideline:** {guidelines.guideline}\n\n"
        f"**Learning Plan:** {', '.join(learning_plan.steps)}\n\n"
        f"**Current Step Index:** {current_step_index}\n"
        f"**Current Step:** {current_step}\n\n"
        f"**Student Message:** {user_message}\n\n"
        f"Provide the next appropriate teaching content for this step. If the student demonstrates "
        f"sufficient understanding of this step, indicate that it's completed."
    )
