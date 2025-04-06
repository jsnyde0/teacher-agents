# Teacher Agents (v0.2)

This project implements an AI-powered conversational tutoring system using Chainlit and PydanticAI. It aims to provide a personalized learning experience by guiding users through custom-generated learning plans.

## Current State (v0.2)

The system currently implements the following flow:

1.  **Onboarding:** A conversational agent (`OnboardingAgent`) gathers the user's current knowledge (Point A), learning goal (Point B), and preferences.
2.  **Pedagogy Generation:** A `PedagogicalMasterAgent` analyzes the onboarding data to create tailored teaching guidelines.
3.  **Plan Generation:** A `JourneyCrafterAgent` uses the onboarding data and guidelines to generate a concise, step-by-step `LearningPlan` (list of topics/tasks).
4.  **Teaching Loop:**
    *   A `TeacherAgent` presents the current step from the `LearningPlan`, guided by the pedagogical guidelines.
    *   A `StepEvaluatorAgent` analyzes the user's response (in context of the teacher's last message) to determine if the user is ready to proceed (`PROCEED`), needs to stay on the current step (`STAY`), or if their intent is unclear (`UNCLEAR`).
    *   The application logic orchestrates the flow, advancing through the plan based on the evaluator's feedback and triggering the `TeacherAgent` to provide follow-up or present the next step.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd teacher-agents
    ```
2.  **Create a Python environment:** (Using `uv` is recommended)
    ```bash
    uv venv
    source .venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    uv pip install -r requirements.txt
    ```
4.  **Configure API Key:**
    *   Create a `.env` file in the project root.
    *   Add your OpenRouter API key:
        ```
        OPENROUTER_API_KEY=sk-or-v1-...
        ```

## Running the Application

Use Chainlit to run the application:

```bash
uv run chainlit run app_chainlit.py -w
```

This will start the Chainlit server, and you can interact with the tutor in your web browser. The `-w` flag enables auto-reload on code changes.

## Agents

*   `src/agents/onboarding_agent.py`: Handles initial user interaction.
*   `src/agents/pedagogical_master_agent.py`: Determines teaching style.
*   `src/agents/journey_crafter_agent.py`: Creates the learning plan.
*   `src/agents/teacher_agent.py`: Delivers lesson content and feedback.
*   `src/agents/step_evaluator_agent.py`: Evaluates user readiness to proceed.

## Orchestration

*   `app_chainlit.py`: Contains the main Chainlit application logic, session management, and agent orchestration.
