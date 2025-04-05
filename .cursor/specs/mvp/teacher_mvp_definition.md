# Specification: Agentic Teacher System - MVP Definition

**Version:** 1.1 (Revised for Clarity)
**Date:** 2024-07-26
**Status:** Defined
**Related Spec:** `.cursor/specs/architecture/agentic_teacher_system.md`

## 1. MVP Goal

To deliver a functional prototype demonstrating the core multi-agent collaboration for creating and executing a personalized learning journey, with basic adaptation based *only* on explicit student feedback or blatant teacher deviation.

## 2. Overall MVP Scope

This MVP focuses on establishing the 5-agent architecture and the basic workflow:

*   **INCLUDED:** Initial onboarding, pedagogical style selection (from predefined set), initial journey planning, step-by-step teaching execution, and basic observation.
*   **INCLUDED (Simplified):** Adaptation is limited to adjusting the *future* plan based on *explicit* triggers, without requiring student confirmation.
*   **EXCLUDED:** Dynamic adaptation of teaching style during the journey, complex student understanding inference by the Monitor, student confirmation loops for plan changes, external content sources, and advanced error handling.

## 3. Agent-Specific MVP Scope

### 3.1. Onboarding Agent (MVP)

*   **Core Purpose:** Gather essential initial student data.
*   **Included Capabilities:**
    *   Ask questions to determine student's current state (Point A).
    *   Ask questions to determine student's learning goals (Point B).
    *   Ask questions to gather relevant background info.
    *   Ask questions to gather explicit pedagogical preferences (e.g., preferred explanation types).
*   **Excluded Capabilities:** Deep diagnostic assessments.
*   **Key Outputs:** Points A & B, background info, explicit preferences (passed to Pedagogical Master).

### 3.2. Pedagogical Master Agent (MVP)

*   **Core Purpose:** Select an initial teaching strategy.
*   **Included Capabilities:**
    *   Receive data from Onboarding Agent (Points A & B, preferences, background).
    *   Determine appropriate pedagogical guidelines based on the input data, leveraging LLM capabilities (guided by a system prompt).
*   **Excluded Capabilities:**
    *   Dynamically adapting the pedagogical style mid-journey.
    *   Receiving feedback flags regarding style effectiveness from the Monitor Agent.
*   **Key Outputs:** Generated initial pedagogical guidelines (passed to Journey Crafter & Teacher).

### 3.3. Journey Crafter Agent (MVP)

*   **Core Purpose:** Create and adapt the learning plan sequence.
*   **Included Capabilities:**
    *   Receive Points A & B, and initial pedagogical guidelines.
    *   Generate an initial structured learning plan (sequence of actionable steps).
    *   Receive pace/content flags from the Monitor Agent (triggered by explicit feedback/deviation).
    *   **Directly modify future steps** in the plan based on received flags (no confirmation needed).
*   **Excluded Capabilities:**
    *   Handling student confirmation requests/responses for plan changes.
*   **Key Outputs:** Initial learning plan, updated future plan steps (passed step-by-step to Teacher).

### 3.4. Teacher Agent (MVP)

*   **Core Purpose:** Interact with the student, executing one plan step at a time.
*   **Included Capabilities:**
    *   Receive the current plan step instructions from Journey Crafter.
    *   Receive initial pedagogical guidelines from Pedagogical Master.
    *   Interact with the student to execute the step (explain, ask questions, provide examples) adhering strictly to the plan step and guidelines.
    *   Generate interaction content using LLM capabilities.
*   **Excluded Capabilities:**
    *   Handling dialogue related to plan change confirmations.
    *   Independently deviating significantly from the plan step.
*   **Key Inputs:** Current plan step, pedagogical guidelines.
*   **Key Outputs:** Interaction with student (observed by Monitor).

### 3.5. Monitor Agent (MVP)

*   **Core Purpose:** Observe interactions and detect specific, explicit triggers for adaptation.
*   **Included Capabilities:**
    *   Passively observe Teacher<->Student interactions.
    *   Receive current plan step details and pedagogical guidelines for context.
    *   **Detect limited, explicit triggers:** See Section 5 below.
    *   Send flags regarding pace/content needs *only* to the Journey Crafter.
*   **Excluded Capabilities:**
    *   Inferring student understanding or pace implicitly from conversation.
    *   Detecting subtle deviations or issues with pedagogical style.
    *   Sending flags regarding style effectiveness to the Pedagogical Master.

## 4. MVP Interaction Flows

1.  **Onboarding:**
    *   `Student` interacts with `Onboarding Agent`.
    *   `Onboarding Agent` collects A, B, background, preferences.
    *   `Onboarding Agent` sends data to `Pedagogical Master`.
2.  **Initial Planning:**
    *   `Pedagogical Master` selects initial style, sends guidelines to `Journey Crafter` & `Teacher`.
    *   `Journey Crafter` receives A, B, guidelines; creates initial plan sequence.
3.  **Teaching Loop (Per Step):**
    *   `Journey Crafter` sends the *next* step's instructions to `Teacher`.
    *   `Teacher` executes the step, interacting with `Student` according to instructions & guidelines.
    *   `Monitor` observes the interaction, comparing against the step instructions.
4.  **MVP Adaptation Loop (Trigger-Based):**
    *   **Trigger:** `Monitor` detects an explicit trigger (see Section 5).
    *   **Flag:** `Monitor` sends a pace/content flag to `Journey Crafter`.
    *   **Adapt:** `Journey Crafter` receives flag, analyzes, and **directly updates future steps** in the plan.
    *   **Continue:** The loop proceeds; `Teacher` will receive the next (potentially updated) step from `Journey Crafter`.

## 5. Monitor Agent MVP Triggers

The Monitor Agent in the MVP will *only* react to:

*   **Explicit Student Commands:** Detecting specific keywords/phrases indicating pace or content needs (e.g., "go faster", "slow down", "I'm stuck", "explain this again", "skip this"). *Exact phrases TBD.*)
*   **Blatant Teacher Deviation:** Detecting if the `Teacher Agent`'s output significantly ignores or contradicts the core task defined in the current plan step instructions. (Mechanism for detection TBD - potentially keyword/instruction matching).

## 6. Key Exclusions Summary (MVP)

*   Dynamic pedagogical style adaptation.
*   Student confirmation for plan changes.
*   Implicit monitoring/inference of student state.
*   External content integration.
*   Sophisticated error handling.

## 7. Next Steps

*   Define the detailed format for the `Journey Crafter`'s plan steps.
*   Specify the exact explicit commands/phrases the `Monitor Agent` should recognize.
*   Refine the mechanism for detecting "Blatant Teacher Deviation".
*   Begin implementation planning.
