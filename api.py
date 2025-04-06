# api.py

# --> Add logging imports <---
import logging
import os
from typing import Any, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware
from pydantic import BaseModel

# Import the new Teacher Agent types if needed by response models (TeacherResponse likely removed)
# from src.agents.teacher_agent import TeacherResponse
# --> Import orchestration module <---
from src import orchestration

# Agent imports
# --> Remove direct model/provider imports <---
# from pydantic_ai.models.openai import OpenAIModel
# from pydantic_ai.providers.openai import OpenAIProvider
# --> Remove direct agent creation imports <---

print("Starting API initialization...")

# --> Configure logging <---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables (ensure .env file is present)
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Updated model selection with larger context window
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "google/gemini-2.0-flash-lite-001"  # 200K token window

print(f"Environment loaded - API Key present: {bool(OPENROUTER_API_KEY)}")
print(f"Using model: {MODEL_NAME}")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Teacher Agents API",
    description="API endpoints for interacting with the onboarding, pedagogical, and journey crafting agents.",
    version="0.1.0",
)

print("FastAPI app initialized")

# Add CORS middleware to allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React default port
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

print("CORS middleware configured")

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
    current_stage: str | None = None  # To inform the frontend about the state
    # Optionally add other data fields if needed by the frontend
    # e.g., onboarding_data: OnboardingData | None = None
    # e.g., pedagogical_guidelines: PedagogicalGuidelines | None = None
    # e.g., learning_plan: LearningPlan | None = None # Note: LearningPlan might be just steps list now


# --- Helper Functions ---


# --> Refactor initialize_session_state (make async) <---
async def initialize_session_state(session_id: str):
    """Initializes agents and state for a new session using orchestration."""
    print(f"\n=== Initializing new session: {session_id} ===")

    if session_id in sessions:
        print(f"Session {session_id} already exists - skipping initialization")
        return

    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY not found. Cannot initialize agents.")
        # Set initial state indicating error
        sessions[session_id] = {
            "agents": {},
            "message_history": [],
            "onboarding_data": None,
            "pedagogical_guidelines": None,
            "learning_plan": None,  # Stores List[str] of steps
            "current_step_index": -1,
            "current_stage": "error",
            "last_teacher_message": None,
            "initialization_error": "API Key missing",
        }
        print(f"Created error state for session {session_id}")
        return

    # Configure the model (can be shared by agents)
    try:
        print("Initializing agents via orchestration...")
        # --> Call orchestration.initialize_agents <---
        agents = await orchestration.initialize_agents(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            model_name=MODEL_NAME,
        )

        if not agents:
            raise Exception("Agent initialization via orchestration failed.")

        print(f"Storing initial session state for {session_id}")
        # Store agents and initial state in the session
        sessions[session_id] = {
            "agents": agents,  # Store the dictionary of agents
            "message_history": [],
            "onboarding_data": None,
            "pedagogical_guidelines": None,
            "learning_plan": None,  # Stores List[str] of steps
            "current_step_index": -1,
            "current_stage": "onboarding",
            "last_teacher_message": None,  # Add missing state key
            "initialization_error": None,
        }
        print(f"Successfully initialized session {session_id}")
        print("=== Session initialization complete ===\n")

    except Exception as e:
        print(f"ERROR initializing agents for session {session_id}: {str(e)}")
        # Store error state to prevent further processing attempts
        sessions[session_id] = {
            "agents": {},
            "message_history": [],
            "onboarding_data": None,
            "pedagogical_guidelines": None,
            "learning_plan": None,
            "current_step_index": -1,
            "current_stage": "error",
            "last_teacher_message": None,
            "initialization_error": str(e),
        }
        print(f"Created error state for session {session_id}")
        print("=== Session initialization failed ===\n")


# --> Refactor process_message <---
async def process_message(session_id: str, user_message: str) -> ChatMessageResponse:
    """Processes the user message using the orchestration module."""
    print(f"\n=== Processing message for session {session_id} ===")
    print(f"User message: {user_message}")

    if session_id not in sessions:
        print(f"New session {session_id} - initializing state")
        await initialize_session_state(session_id)  # Use await for async init

    # Retrieve the current state for the session
    current_session_state = sessions.get(session_id)

    # Handle cases where initialization failed or session_state is None
    if not current_session_state or current_session_state.get("initialization_error"):
        error_msg = (
            current_session_state.get(
                "initialization_error", "Session not initialized properly."
            )
            if current_session_state
            else "Session not found."
        )
        logger.error(f"Cannot process message - {error_msg}")  # Use logger
        # Return an error response, but don't raise HTTPException here
        # Let the caller (/chat endpoint) handle the HTTP response
        return ChatMessageResponse(
            reply=f"Error: Cannot process message. {error_msg}",
            session_id=session_id,
            current_stage="error",
        )

    # --> Call orchestration.handle_message <---
    logger.info(
        f"Calling handle_message for stage: {current_session_state.get('current_stage')}"
    )  # Use logger
    reply_message, new_session_state = await orchestration.handle_message(
        session_state=current_session_state, user_message=user_message
    )

    # --> Update the session state in the global dictionary <---
    logger.info(
        f"Updating session {session_id} with new state (New Stage: {new_session_state.get('current_stage')}) "
    )  # Use logger
    sessions[session_id] = new_session_state

    print("=== Message processing complete ===\n")
    # Return the response object for the API endpoint
    return ChatMessageResponse(
        reply=reply_message,
        session_id=session_id,
        current_stage=new_session_state.get("current_stage"),
    )

    # --- Remove old stage-based logic from process_message --- #
    # (Already removed in previous edit simulation)


# --- API Endpoints ---
@app.post("/chat", response_model=ChatMessageResponse)
async def chat_endpoint(request: ChatMessageRequest):
    """
    Main endpoint to handle user messages. Requires a session_id.
    Manages conversation state based on the session_id.
    """
    print("\n=== Received chat request ===")
    print(f"Session ID: {request.session_id}")
    print(f"Message length: {len(request.message)}")

    if not OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY not configured")  # Use logger
        # Check moved to initialization, but keep a check here for robustness
        raise HTTPException(
            status_code=500, detail="OPENROUTER_API_KEY not configured."
        )

    if not request.session_id:
        logger.error("Missing session_id")  # Use logger
        raise HTTPException(status_code=400, detail="session_id is required.")

    # Basic input validation
    if not request.message:
        logger.error("Empty message received")  # Use logger
        raise HTTPException(status_code=400, detail="message cannot be empty.")

    print("Processing message...")
    # Process the message using our async helper function
    response = await process_message(request.session_id, request.message)

    # Check if process_message resulted in an error state before returning
    if response.current_stage == "error":
        logger.error(f"Error response generated: {response.reply}")  # Use logger
        # Raise HTTPException here, as this is the API boundary
        raise HTTPException(status_code=500, detail=response.reply)

    print("Successfully processed chat request")
    print(f"Response stage: {response.current_stage}")
    print("=== Chat request complete ===\n")
    return response


@app.get("/health")
async def health_check():
    print("Health check requested")
    return {"status": "ok"}


# --- Running the App (for local development) ---
if __name__ == "__main__":
    import uvicorn

    logger.info("Starting FastAPI server...")  # Use logger
    logger.info(f"API Key Loaded: {bool(OPENROUTER_API_KEY)}")
    logger.info("Starting server on http://0.0.0.0:8000")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
