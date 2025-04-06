# api.py
import os
from typing import Dict, List, Any
import asyncio # Import asyncio for async operations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Agent imports
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.messages import ModelMessage # Import ModelMessage
from src.agents.journey_crafter_agent import (
    LearningPlan,
    create_journey_crafter_agent,
)
from src.agents.onboarding_agent import OnboardingData, create_onboarding_agent
from src.agents.pedagogical_master_agent import (
    PedagogicalGuidelines,
    create_pedagogical_master_agent,
)

# Load environment variables (ensure .env file is present)
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "google/gemini-flash-1.5" # Or your preferred model

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Teacher Agents API",
    description="API endpoints for interacting with the onboarding, pedagogical, and journey crafting agents.",
    version="0.1.0",
)

# --- In-Memory Session Store ---
# WARNING: This is for development only. Data is lost on server restart.
# Use Redis or a database for production.
sessions: Dict[str, Dict[str, Any]] = {}

# --- API Request and Response Models ---
class ChatMessageRequest(BaseModel):
    session_id: str
    message: str

class ChatMessageResponse(BaseModel):
    reply: str
    session_id: str
    current_stage: str | None = None # To inform the frontend about the state
    # Optionally add other data fields if needed by the frontend
    # e.g., onboarding_data: OnboardingData | None = None
    # e.g., pedagogical_guidelines: PedagogicalGuidelines | None = None
    # e.g., learning_plan: LearningPlan | None = None

# --- Helper Functions ---

def initialize_session_state(session_id: str):
    """Initializes agents and state for a new session."""
    if session_id in sessions:
        print(f"Session {session_id} already exists.")
        return

    if not OPENROUTER_API_KEY:
        # In a real API, you might handle this differently, maybe at app startup
        # or return a specific error to the client.
        print("Error: OPENROUTER_API_KEY not found. Cannot initialize agents.")
        # For now, initialize with None to avoid breaking session structure
        sessions[session_id] = {
            "onboarding_agent": None,
            "pedagogical_master_agent": None,
            "journey_crafter_agent": None,
            "message_history": [],
            "onboarding_data": None,
            "pedagogical_guidelines": None,
            "current_stage": "onboarding",
            "initialization_error": "API Key missing"
        }
        return

    # Configure the model (can be shared by agents)
    try:
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
        journey_crafter_agent = create_journey_crafter_agent(model)

        # Store agents and initial state in the session
        sessions[session_id] = {
            "onboarding_agent": onboarding_agent,
            "pedagogical_master_agent": pedagogical_master_agent,
            "journey_crafter_agent": journey_crafter_agent,
            "message_history": [], # Initialize empty history
            "onboarding_data": None,
            "pedagogical_guidelines": None,
            "current_stage": "onboarding", # Start at onboarding
            "initialization_error": None
        }
        print(f"Initialized agents and state for session: {session_id}")

    except Exception as e:
        print(f"Error initializing agents for session {session_id}: {e}")
        # Store error state to prevent further processing attempts
        sessions[session_id] = {
            "onboarding_agent": None,
            "pedagogical_master_agent": None,
            "journey_crafter_agent": None,
            "message_history": [],
            "onboarding_data": None,
            "pedagogical_guidelines": None,
            "current_stage": "error",
            "initialization_error": str(e)
        }

# Define async helper functions for running agents, mirroring app_chainlit.py
async def run_pedagogical_master_api(
    session_state: Dict[str, Any], onboarding_data: OnboardingData
) -> PedagogicalGuidelines | str:
    """Runs the PMA using the agent instance from session_state."""
    pma = session_state.get("pedagogical_master_agent")
    if not pma:
        return "Error: Pedagogical Master Agent not found in session state."

    print(f"\n--- API: Triggering PMA for session {session_state.get('session_id', 'N/A')} ---")
    pma_input_prompt = (
        f"Determine pedagogical guidelines based on the following student profile:\n"
        f"Current Knowledge (Point A): {onboarding_data.point_a}\n"
        f"Learning Goal (Point B): {onboarding_data.point_b}\n"
        f"Learning Preferences: {onboarding_data.preferences}"
    )
    try:
        # Assuming pydantic_ai's run might be async based on chainlit usage
        # If run is blocking, use: pma_result = await asyncio.to_thread(pma.run, pma_input_prompt)
        pma_result = await pma.run(pma_input_prompt) # Make sure pma.run is awaitable
        if isinstance(pma_result.data, PedagogicalGuidelines):
            print(f"API PMA Result: {pma_result.data.guideline}")
            print("----------------------------------------")
            return pma_result.data
        else:
            error_msg = f"API PMA Error: Unexpected result type {type(pma_result.data)}"
            print(error_msg)
            print("----------------------------------------")
            return error_msg
    except Exception as e:
        error_msg = f"API Error during pedagogical guideline generation: {e}"
        print(f"API PMA Error during run: {e}")
        print("----------------------------------------")
        return error_msg


async def run_journey_crafter_api(
    session_state: Dict[str, Any], onboarding_data: OnboardingData, guidelines: PedagogicalGuidelines
) -> LearningPlan | str:
    """Runs the JCA using the agent instance from session_state."""
    jca = session_state.get("journey_crafter_agent")
    if not jca:
        return "Error: Journey Crafter Agent not found in session state."

    print(f"\n--- API: Triggering JCA for session {session_state.get('session_id', 'N/A')} ---")
    jca_input_prompt = (
        f"Create a learning plan based on the following profile and guidelines:\n\n"
        f"**Student Profile:**\n"
        f"- Current Knowledge (Point A): {onboarding_data.point_a}\n"
        f"- Learning Goal (Point B): {onboarding_data.point_b}\n"
        f"- Learning Preferences: {onboarding_data.preferences}\n\n"
        f"**Pedagogical Guideline:** {guidelines.guideline}"
    )
    try:
        # Assuming jca.run might be async
        # If run is blocking, use: jca_result = await asyncio.to_thread(jca.run, jca_input_prompt)
        jca_result = await jca.run(jca_input_prompt) # Make sure jca.run is awaitable
        if isinstance(jca_result.data, LearningPlan):
            print(f"API JCA Result: {len(jca_result.data.steps)} steps generated.")
            print("--------------------------------------")
            return jca_result.data
        else:
            error_msg = f"API JCA Error: Unexpected result type {type(jca_result.data)}"
            print(error_msg)
            print("--------------------------------------")
            return error_msg
    except Exception as e:
        error_msg = f"API Error during learning plan generation: {e}"
        print(f"API JCA Error during run: {e}")
        print("--------------------------------------")
        return error_msg

async def process_message(session_id: str, user_message: str) -> ChatMessageResponse:
    """Processes the user message using the appropriate agent based on session state."""
    if session_id not in sessions:
        initialize_session_state(session_id)

    # Use .get() to safely retrieve session_state
    session_state = sessions.get(session_id)

    # Handle cases where initialization failed or session_state is None
    if not session_state or session_state.get("initialization_error"):
        error_msg = session_state.get("initialization_error", "Session not initialized properly.") if session_state else "Session not found."
        return ChatMessageResponse(reply=f"Error: Cannot process message. {error_msg}", session_id=session_id, current_stage="error")

    current_stage = session_state.get("current_stage", "onboarding")
    message_history: List[ModelMessage] = session_state.get("message_history", [])
    # Add session_id to state for logging in helper functions if needed
    session_state["session_id"] = session_id

    reply_message = "An unexpected error occurred." # Default error reply

    # If planning is already complete, inform the user
    if current_stage == "complete":
        reply_message = "Planning is complete. The next step would involve the Teacher Agent (not implemented in this API)."
        # No state change needed here, just return the message
        return ChatMessageResponse(reply=reply_message, session_id=session_id, current_stage=current_stage)

    # --- Stage: Onboarding ---
    if current_stage == "onboarding":
        onboarding_agent = session_state.get("onboarding_agent")
        if not onboarding_agent:
            reply_message = "Error: Onboarding Agent not available for this session."
            session_state["current_stage"] = "error"
            sessions[session_id] = session_state # Update state before returning
            return ChatMessageResponse(reply=reply_message, session_id=session_id, current_stage="error")
        else:
            try:
                print(f"--- API: Running Onboarding Agent for session {session_id} ---")
                # Assuming onboarding_agent.run is awaitable
                # If run is blocking, use: oa_result = await asyncio.to_thread(onboarding_agent.run, user_message, message_history=message_history)
                oa_result = await onboarding_agent.run(user_message, message_history=message_history)

                # Update history immediately after successful run
                # Make sure all_messages returns serializable objects if needed elsewhere
                message_history.extend(oa_result.all_messages())
                session_state["message_history"] = message_history

                if isinstance(oa_result.data, OnboardingData):
                    # --- Onboarding Complete: Auto-trigger PMA and JCA --- #
                    onboarding_data = oa_result.data
                    session_state["onboarding_data"] = onboarding_data
                    reply_message = ( # Initial reply confirming onboarding data
                        f"Great, thank you! Onboarding complete:\n"
                        f"- **Point A:** {onboarding_data.point_a}\n"
                        f"- **Point B:** {onboarding_data.point_b}\n"
                        f"- **Preferences:** {onboarding_data.preferences}\n\n"
                        f"Now determining teaching approach..."
                    )
                    session_state["current_stage"] = "pedagogy" # Tentative stage update

                    # Run PMA
                    pma_result = await run_pedagogical_master_api(session_state, onboarding_data)
                    if isinstance(pma_result, PedagogicalGuidelines):
                        session_state["pedagogical_guidelines"] = pma_result
                        reply_message = ( # Update reply with guideline
                            f"Great, thank you! Onboarding complete:\n"
                            f"- **Point A:** {onboarding_data.point_a}\n"
                            f"- **Point B:** {onboarding_data.point_b}\n"
                            f"- **Preferences:** {onboarding_data.preferences}\n\n"
                            f"**Suggested Guideline:** {pma_result.guideline}\n\n"
                            f"Now crafting the learning plan..."
                        )
                        session_state["current_stage"] = "journey_crafting" # Update stage

                        # Run JCA
                        jca_result = await run_journey_crafter_api(session_state, onboarding_data, pma_result)
                        if isinstance(jca_result, LearningPlan):
                             plan_steps_text = "\n".join([f"  {i + 1}. {step}" for i, step in enumerate(jca_result.steps)])
                             reply_message = ( # Final reply with plan
                                f"Great, thank you! Onboarding complete:\n"
                                f"- **Point A:** {onboarding_data.point_a}\n"
                                f"- **Point B:** {onboarding_data.point_b}\n"
                                f"- **Preferences:** {onboarding_data.preferences}\n\n"
                                f"**Suggested Guideline:** {pma_result.guideline}\n\n"
                                f"**Proposed Learning Plan:**\n{plan_steps_text}\n\n"
                                f"*(Planning complete!)*"
                            )
                             session_state["learning_plan"] = jca_result # Store the plan
                             session_state["current_stage"] = "complete" # Final stage
                        else: # JCA failed
                            reply_message = f"Onboarding and guideline generation complete, but failed to create learning plan: {jca_result}"
                            session_state["current_stage"] = "error" # Set error stage
                    else: # PMA failed
                         reply_message = f"Onboarding complete, but failed to determine pedagogical guidelines: {pma_result}"
                         session_state["current_stage"] = "error" # Set error stage

                elif isinstance(oa_result.data, str):
                    # Onboarding agent returned a string (likely asking for more info)
                    reply_message = oa_result.data
                    # Keep stage as "onboarding"
                else:
                    # Unexpected result from onboarding agent
                     reply_message = f"Error: Unexpected response type from Onboarding Agent: {type(oa_result.data)}"
                     session_state["current_stage"] = "error"

            except Exception as e:
                print(f"API Error during Onboarding stage for session {session_id}: {e}")
                reply_message = f"An error occurred during the onboarding process: {e}"
                session_state["current_stage"] = "error"

    # If the stage progressed beyond onboarding within this call, it's handled above.
    # If it's still pedagogy/journey_crafting, it means something went wrong or is unexpected.
    # This case shouldn't normally be hit if OA always transitions or returns a string.
    elif current_stage in ["pedagogy", "journey_crafting"]:
         reply_message = f"Error: Unexpected state '{current_stage}'. Processing should happen automatically after onboarding completes successfully."
         session_state["current_stage"] = "error"


    # Update the master sessions dictionary with the final state for this session_id
    sessions[session_id] = session_state

    return ChatMessageResponse(
        reply=reply_message,
        session_id=session_id,
        current_stage=session_state.get("current_stage")
    )

# --- API Endpoints ---
@app.post("/chat", response_model=ChatMessageResponse)
async def chat_endpoint(request: ChatMessageRequest):
    """
    Main endpoint to handle user messages. Requires a session_id.
    Manages conversation state based on the session_id.
    """
    if not OPENROUTER_API_KEY:
        # Check moved to initialization, but keep a check here for robustness
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not configured.")

    if not request.session_id:
        raise HTTPException(status_code=400, detail="session_id is required.")

    # Basic input validation
    if not request.message:
        raise HTTPException(status_code=400, detail="message cannot be empty.")

    # Process the message using our async helper function
    response = await process_message(request.session_id, request.message) # Use await

    # Check if process_message resulted in an error state
    if response.current_stage == "error":
        # You might want different status codes depending on the error cause
        # e.g., 500 for internal server/agent errors, 400 for bad requests (though FastAPI handles some)
        raise HTTPException(status_code=500, detail=response.reply)

    return response

@app.get("/health")
async def health_check():
    return {"status": "ok"}

# --- Running the App (for local development) ---
if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server...")
    print("API Key Loaded:", bool(OPENROUTER_API_KEY))
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 