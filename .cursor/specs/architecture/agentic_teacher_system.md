# Specification: Agentic Teacher System Architecture

**Version:** 1.0
**Date:** 2024-07-26
**Status:** Draft

## 1. Overview

This document specifies the architecture for the Agentic Teacher system, a collaborative multi-agent system designed to provide personalized and adaptive learning experiences. The system comprises five core agents: Onboarding, Pedagogical Master, Journey Crafter, Teacher, and Monitor.

## 2. System Goal

- [ ] Provide a personalized learning journey tailored to individual student needs.
- [ ] Adapt the learning pace and style based on student progress and feedback.
- [ ] Guide the student effectively from their current state (Point A) to their desired learning outcome (Point B).

## 3. Agent Specifications

### 3.1. Onboarding Agent

*   **Purpose:** Gather initial information about the student and their learning goals.
*   **Inputs:**
    *   - [ ] Student interaction (responses to questions).
*   **Outputs:**
    *   - [ ] Student's current knowledge/state (Point A).
    *   - [ ] Student's desired learning outcomes (Point B).
    *   - [ ] Initial background information relevant to learning.
    *   - [ ] Explicitly stated learning preferences (for MVP).
*   **Key Interactions:**
    *   - [ ] Interacts primarily with the student during the setup phase.
    *   - [ ] Passes its collected output to the `Pedagogical Master Agent`.
*   **MVP Considerations:**
    *   Focuses on clearly capturing Points A, B, and basic pedagogical preferences through direct questions. Does not require deep diagnostic capabilities.

### 3.2. Pedagogical Master Agent

*   **Purpose:** Define the *how* (pedagogical strategy) of the learning journey.
*   **Inputs:**
    *   - [ ] Output from `Onboarding Agent` (Points A & B, preferences).
    *   - [ ] Flags from `Monitor Agent` regarding the effectiveness of the current pedagogical style.
*   **Outputs:**
    *   - [ ] A set of pedagogical guidelines/instructions for the `Teacher Agent` and potentially `Journey Crafter`.
    *   - [ ] Updates to guidelines based on `Monitor Agent` feedback.
*   **Key Interactions:**
    *   - [ ] Receives data from `Onboarding Agent`.
    *   - [ ] Sends guidelines to `Journey Crafter` and `Teacher Agent`.
    *   - [ ] Receives style effectiveness flags from `Monitor Agent`.
*   **MVP Considerations:**
    *   Selects from a *predefined* set of pedagogical styles based on `Onboarding Agent` data.
    *   Adapts guidelines based on `Monitor Agent` flags indicating style ineffectiveness. Does not require complex pedagogical theory reasoning.

### 3.3. Journey Crafter Agent

*   **Purpose:** Create and adapt the learning plan (the sequence of steps from A to B).
*   **Inputs:**
    *   - [ ] Points A & B from `Onboarding Agent`.
    *   - [ ] Pedagogical guidelines from `Pedagogical Master Agent`.
    *   - [ ] Flags from `Monitor Agent` indicating a need for pace/content adjustment.
    *   - [ ] Confirmation/rejection of proposed plan changes (via `Teacher Agent`).
*   **Outputs:**
    *   - [ ] A structured learning plan: A sequence of concrete, actionable steps for the `Teacher Agent`.
    *   - [ ] Proposed modifications to the *future* parts of the plan based on `Monitor Agent` flags.
    *   - [ ] An updated plan once modifications are confirmed.
*   **Key Interactions:**
    *   - [ ] Receives info from `Onboarding Agent` and `Pedagogical Master`.
    *   - [ ] Sends the plan (and current step identifier) to the `Teacher Agent`.
    *   - [ ] Receives pace/content flags from `Monitor Agent`.
    *   - [ ] Proposes plan changes (via `Teacher Agent`) and updates the plan based on confirmation.
*   **MVP Considerations:**
    *   Creates a logical sequence of steps.
    *   Focuses on adapting the *future* path based on flags and confirmation.

### 3.4. Teacher Agent

*   **Purpose:** Execute the learning plan step-by-step, interacting directly with the student.
*   **Inputs:**
    *   - [ ] The *specific current step* content/task from the `Journey Crafter`.
    *   - [ ] Pedagogical guidelines from the `Pedagogical Master Agent`.
    *   - [ ] Student interaction (questions, answers, feedback).
    *   - [ ] Proposed plan changes from `Journey Crafter` to relay for confirmation.
*   **Outputs:**
    *   - [ ] Explanations, examples, questions directed to the student, adhering to plan step and guidelines.
    *   - [ ] Interaction data for the `Monitor Agent` to observe.
    *   - [ ] Relayed proposed plan changes to the student.
    *   - [ ] Relayed student confirmation/rejection of changes back towards `Journey Crafter`.
*   **Key Interactions:**
    *   - [ ] Interacts heavily with the student.
    *   - [ ] Receives current step from `Journey Crafter`.
    *   - [ ] Receives guidelines from `Pedagogical Master`.
    *   - [ ] Its interaction is observed by the `Monitor Agent`.
    *   - [ ] Relays confirmation dialogue between student and `Journey Crafter`.
*   **MVP Considerations:**
    *   Focuses on faithfully executing the assigned step and adhering to pedagogical guidelines.
    *   Uses LLM for content generation within the step's bounds.
    *   Adherence ensured via system prompt and structured step input.

### 3.5. Monitor Agent

*   **Purpose:** Observe the learning process and flag deviations or issues requiring adaptation.
*   **Inputs:**
    *   - [ ] The current plan step details from `Journey Crafter`.
    *   - [ ] Pedagogical guidelines from `Pedagogical Master`.
    *   - [ ] The real-time interaction data between the `Teacher Agent` and the student.
*   **Outputs:**
    *   - [ ] Flags sent to the appropriate agent (`Journey Crafter` for pace/content issues, `Pedagogical Master` for style issues).
    *   - [ ] Flags specify the detected issue (e.g., "Student struggling," "Pace too slow," "Style ineffective," "Teacher deviated").
*   **Key Interactions:**
    *   - [ ] Observes `Teacher Agent`<->Student interaction passively.
    *   - [ ] Sends flags to `Journey Crafter` and `Pedagogical Master`.
    *   - [ ] Does not interact directly with the student.
*   **MVP Considerations:**
    *   Focuses on detecting clear triggers (explicit feedback, significant performance deviation, teacher going off-script).

## 4. Interaction Flow & Adaptation

- [ ] Initial Flow: Onboarding -> Pedagogical Master -> Journey Crafter -> Teacher (starts executing plan step 1).
- [ ] Monitoring: Monitor observes Teacher<->Student interaction constantly.
- [ ] Adaptation Loop (Pace/Content): Monitor flags -> Journey Crafter proposes future plan change -> Teacher relays proposal -> Student confirms/rejects -> Journey Crafter updates plan if confirmed.
- [ ] Adaptation Loop (Style): Monitor flags -> Pedagogical Master updates guidelines -> Teacher uses updated guidelines for subsequent steps.

## 5. Open Questions / Future Considerations

*   Detailed format of the structured learning plan steps.
*   Specific triggers and thresholds for the Monitor Agent.
*   Handling of content sources beyond LLM generation.
*   Mechanisms for deeper pedagogical reasoning (post-MVP).
*   Error handling and recovery within agent interactions.
