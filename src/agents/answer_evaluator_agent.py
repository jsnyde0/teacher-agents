# src/agents/answer_evaluator_agent.py
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel


# --- Data Model ---
class AnswerEvaluationResult(BaseModel):
    """Structured result for evaluating a student's answer."""

    evaluation: Literal[
        "correct", "incorrect", "partial", "unclear", "not_applicable"
    ] = Field(
        ..., description="Categorization of the student's answer correctness/relevance."
    )
    explanation: str = Field(
        ...,
        description="Brief explanation for the evaluation (e.g., why it's incorrect, or confirming correctness). Max 1-2 sentences.",
    )


# --- Agent Definition ---
def create_answer_evaluator_agent(model: OpenAIModel) -> Agent:
    """Creates the Answer Evaluation Agent instance.

    This agent evaluates a student's response against a teacher's instruction
    and the learning context, providing a structured evaluation.

    Args:
        model: The PydanticAI model instance.

    Returns:
        An initialized Agent instance for the Answer Evaluator.
    """
    return Agent(
        model=model,
        result_type=AnswerEvaluationResult,
        system_prompt="""You are an expert Answer Evaluator for a tutoring system. Your sole task is to evaluate a student's response based on the teacher's last instruction and the current learning goal.

**Input Context (provided in user prompt):**
- Current Learning Step Goal: [The objective the student is trying to achieve]
- Teacher's Last Instruction/Question: [The specific question or task given by the teacher]
- Student's Response: [The student's verbatim answer]

**Your Task:**
1.  Analyze the 'Student's Response' strictly in the context of the 'Teacher's Last Instruction/Question' and the 'Current Learning Step Goal'.
2.  Determine the most accurate evaluation category:
    - `correct`: The student correctly answered the question or performed the task as instructed.
    - `incorrect`: The student's response is definitively wrong or fails to address the core instruction.
    - `partial`: The student's response is partially correct but incomplete or contains minor errors.
    - `unclear`: The student's response is too ambiguous or vague to determine correctness (e.g., "maybe?", "I think so").
    - `not_applicable`: The student's response doesn't attempt to answer or perform the task (e.g., asks a different question, says "I don't know", off-topic).
3.  Write a concise, objective, one-sentence 'explanation' justifying your chosen evaluation category. Focus on *why* it's correct/incorrect/etc.
4.  Respond ONLY with the structured `AnswerEvaluationResult` object containing the `evaluation` and `explanation`.

**Example:**
If Teacher asks "What is 2+2?" and Student responds "3", your output should be:
`{"evaluation": "incorrect", "explanation": "The student provided the wrong sum for 2+2."}`

Do NOT add any conversational text or additional commentary.""",
    )
