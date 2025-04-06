# api.py
import asyncio
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware
from pydantic import BaseModel
from pydantic_ai.messages import ModelMessage  # Import ModelMessage

# Agent imports
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from src.agents.journey_crafter_agent import (
    LearningPlan,
    create_journey_crafter_agent,
)
from src.agents.onboarding_agent import OnboardingData, create_onboarding_agent
from src.agents.pedagogical_master_agent import (
    PedagogicalGuidelines,
    create_pedagogical_master_agent,
)

# Import the new Teacher Agent
from src.agents.teacher_agent import (
    TeacherResponse,
    create_teacher_agent,
    prepare_teacher_input,
)

print("Starting API initialization...")

# Load environment variables (ensure .env file is present)
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Updated model selection with larger context window 
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "anthropic/claude-3-opus-20240229" # 200K token window

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
    # e.g., learning_plan: LearningPlan | None = None


# --- Helper Functions ---


def initialize_session_state(session_id: str):
    """Initializes agents and state for a new session."""
    print(f"\n=== Initializing new session: {session_id} ===")
    
    if session_id in sessions:
        print(f"Session {session_id} already exists - skipping initialization")
        return

    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY not found. Cannot initialize agents.")
        # For now, initialize with None to avoid breaking session structure
        sessions[session_id] = {
            "onboarding_agent": None,
            "pedagogical_master_agent": None,
            "journey_crafter_agent": None,
            "teacher_agent": None,
            "message_history": [],
            "onboarding_data": None,
            "pedagogical_guidelines": None,
            "learning_plan": None,
            "current_step_index": 0,
            "current_stage": "onboarding",
            "initialization_error": "API Key missing",
        }
        print(f"Created error state for session {session_id}")
        return

    # Configure the model (can be shared by agents)
    try:
        print("Configuring OpenAI model...")
        model = OpenAIModel(
            MODEL_NAME,
            provider=OpenAIProvider(
                base_url=OPENROUTER_BASE_URL,
                api_key=OPENROUTER_API_KEY,
            ),
        )
        print("Model configured successfully")

        print("Creating agent instances...")
        # Create the agent instances
        onboarding_agent = create_onboarding_agent(model)
        print("- Onboarding Agent created")
        pedagogical_master_agent = create_pedagogical_master_agent(model)
        print("- Pedagogical Master Agent created")
        journey_crafter_agent = create_journey_crafter_agent(model)
        print("- Journey Crafter Agent created")
        teacher_agent = create_teacher_agent(model)
        print("- Teacher Agent created")

        print(f"Storing initial session state for {session_id}")
        # Store agents and initial state in the session
        sessions[session_id] = {
            "onboarding_agent": onboarding_agent,
            "pedagogical_master_agent": pedagogical_master_agent,
            "journey_crafter_agent": journey_crafter_agent,
            "teacher_agent": teacher_agent,
            "message_history": [],  # Initialize empty history
            "onboarding_data": None,
            "pedagogical_guidelines": None,
            "learning_plan": None,
            "current_step_index": 0,
            "current_stage": "onboarding",  # Start at onboarding
            "initialization_error": None,
        }
        print(f"Successfully initialized session {session_id}")
        print("=== Session initialization complete ===\n")

    except Exception as e:
        print(f"ERROR initializing agents for session {session_id}: {str(e)}")
        # Store error state to prevent further processing attempts
        sessions[session_id] = {
            "onboarding_agent": None,
            "pedagogical_master_agent": None,
            "journey_crafter_agent": None,
            "teacher_agent": None,
            "message_history": [],
            "onboarding_data": None,
            "pedagogical_guidelines": None,
            "learning_plan": None,
            "current_step_index": 0,
            "current_stage": "error",
            "initialization_error": str(e),
        }
        print(f"Created error state for session {session_id}")
        print("=== Session initialization failed ===\n")

# Define async helper functions for running agents, mirroring app_chainlit.py
async def run_pedagogical_master_api(
    session_state: Dict[str, Any], onboarding_data: OnboardingData
) -> PedagogicalGuidelines | str:
    """Runs the PMA using the agent instance from session_state."""
    print("\n=== Running Pedagogical Master Agent ===")
    print(f"Session ID: {session_state.get('session_id', 'N/A')}")
    
    pma = session_state.get("pedagogical_master_agent")
    if not pma:
        print("ERROR: Pedagogical Master Agent not found in session state")
        return "Error: Pedagogical Master Agent not found in session state."

    print("Preparing PMA input prompt with onboarding data:")
    print(f"- Point A: {onboarding_data.point_a}")
    print(f"- Point B: {onboarding_data.point_b}")
    print(f"- Preferences: {onboarding_data.preferences}")
    
    pma_input_prompt = (
        f"Determine pedagogical guidelines based on the following student profile:\n"
        f"Current Knowledge (Point A): {onboarding_data.point_a}\n"
        f"Learning Goal (Point B): {onboarding_data.point_b}\n"
        f"Learning Preferences: {onboarding_data.preferences}"
    )
    
    try:
        print("Executing PMA...")
        pma_result = await pma.run(pma_input_prompt)
        
        if isinstance(pma_result.data, PedagogicalGuidelines):
            print("PMA execution successful")
            print(f"Generated guideline: {pma_result.data.guideline}")
            print("=== PMA execution complete ===\n")
            return pma_result.data
        else:
            error_msg = f"PMA Error: Unexpected result type {type(pma_result.data)}"
            print(f"ERROR: {error_msg}")
            print("=== PMA execution failed ===\n")
            return error_msg
    except Exception as e:
        error_msg = f"PMA Error during execution: {str(e)}"
        print(f"ERROR: {error_msg}")
        print("=== PMA execution failed ===\n")
        return error_msg


async def run_journey_crafter_api(
    session_state: Dict[str, Any],
    onboarding_data: OnboardingData,
    guidelines: PedagogicalGuidelines,
) -> LearningPlan | str:
    """Runs the JCA using the agent instance from session_state."""
    print("\n=== Running Journey Crafter Agent ===")
    print(f"Session ID: {session_state.get('session_id', 'N/A')}")
    
    jca = session_state.get("journey_crafter_agent")
    if not jca:
        print("ERROR: Journey Crafter Agent not found in session state")
        return "Error: Journey Crafter Agent not found in session state."

    print("Preparing JCA input with:")
    print(f"- Point A: {onboarding_data.point_a}")
    print(f"- Point B: {onboarding_data.point_b}")
    print(f"- Preferences: {onboarding_data.preferences}")
    print(f"- Guidelines: {guidelines.guideline}")
    
    jca_input_prompt = (
        f"Create a learning plan based on the following profile and guidelines:\n\n"
        f"**Student Profile:**\n"
        f"- Current Knowledge (Point A): {onboarding_data.point_a}\n"
        f"- Learning Goal (Point B): {onboarding_data.point_b}\n"
        f"- Learning Preferences: {onboarding_data.preferences}\n\n"
        f"**Pedagogical Guideline:** {guidelines.guideline}"
    )
    
    try:
        print("Executing JCA...")
        jca_result = await jca.run(jca_input_prompt)
        
        if isinstance(jca_result.data, LearningPlan):
            print("JCA execution successful")
            print(f"Generated {len(jca_result.data.steps)} learning steps")
            print("Learning plan steps:")
            for i, step in enumerate(jca_result.data.steps, 1):
                print(f"  {i}. {step}")
            print("=== JCA execution complete ===\n")
            return jca_result.data
        else:
            error_msg = f"JCA Error: Unexpected result type {type(jca_result.data)}"
            print(f"ERROR: {error_msg}")
            print("=== JCA execution failed ===\n")
            return error_msg
    except Exception as e:
        error_msg = f"JCA Error during execution: {str(e)}"
        print(f"ERROR: {error_msg}")
        print("=== JCA execution failed ===\n")
        return error_msg


async def run_teacher_agent_api(
    session_state: Dict[str, Any], user_message: str
) -> TeacherResponse | str:
    """Runs the Teacher Agent using the agent instance from session_state."""
    print("\n=== Running Teacher Agent ===")
    print(f"Session ID: {session_state.get('session_id', 'N/A')}")
    print(f"User message: {user_message}")
    
    teacher = session_state.get("teacher_agent")
    if not teacher:
        print("ERROR: Teacher Agent not found in session state")
        return "Error: Teacher Agent not found in session state."

    # Get required data from session state
    onboarding_data = session_state.get("onboarding_data")
    guidelines = session_state.get("pedagogical_guidelines")
    learning_plan = session_state.get("learning_plan")
    current_step_index = session_state.get("current_step_index", 0)

    if not onboarding_data or not guidelines or not learning_plan:
        print("ERROR: Missing required data for Teacher Agent")
        print(f"- Onboarding data present: {bool(onboarding_data)}")
        print(f"- Guidelines present: {bool(guidelines)}")
        print(f"- Learning plan present: {bool(learning_plan)}")
        return "Error: Missing required data for Teacher Agent."
    
    print(f"Current step: {current_step_index + 1} of {len(learning_plan.steps)}")
    print("Preparing teacher input...")
    
    # Prepare the input for the teacher agent
    teacher_input = prepare_teacher_input(
        onboarding_data, guidelines, learning_plan, current_step_index, user_message
    )
    
    # Print input details for debugging
    print(f"Teacher input - Length: {len(teacher_input)}")
    print(f"Current step content: {learning_plan.steps[current_step_index]}")
    print(f"Input contains onboarding data: {'point_a' in teacher_input and 'point_b' in teacher_input}")
    print(f"Input contains guidelines: {guidelines.guideline[:50] in teacher_input}")
    
    try:
        print("Executing Teacher Agent...")
        # Add retry logic for empty responses
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"Attempt {attempt+1} of {max_retries}")
                teacher_result = await teacher.run(teacher_input)
                
                # Log detailed result information
                print(f"Teacher result type: {type(teacher_result)}")
                print(f"Has data attribute: {hasattr(teacher_result, 'data')}")
                if hasattr(teacher_result, 'data'):
                    print(f"Result data type: {type(teacher_result.data) if teacher_result.data else 'None'}")
                
                # Check for empty response
                if teacher_result is None:
                    print(f"ERROR: Teacher result is None on attempt {attempt+1}")
                    if attempt == max_retries - 1:
                        return "Error: Teacher Agent returned None response after multiple attempts."
                    continue
                elif hasattr(teacher_result, 'data') and teacher_result.data is None:
                    print(f"ERROR: Teacher result data is None on attempt {attempt+1}")
                    if attempt == max_retries - 1:
                        return "Error: Teacher Agent returned result with None data after multiple attempts."
                    continue
                    
                if isinstance(teacher_result.data, TeacherResponse):
                    print("Teacher Agent execution successful")
                    print(f"Response content length: {len(teacher_result.data.content)}")
                    print(f"Step {teacher_result.data.current_step_index + 1}")
                    print(f"Step completed: {teacher_result.data.completed}")
                    print("=== Teacher Agent execution complete ===\n")
                    return teacher_result.data
                else:
                    error_msg = f"Teacher Agent Error: Unexpected result type {type(teacher_result.data)}"
                    print(f"ERROR: {error_msg}")
                    print(f"Result data: {teacher_result.data}")
                    print("=== Teacher Agent execution failed ===\n")
                    return error_msg
            except Exception as retry_e:
                print(f"Attempt {attempt+1} failed with error: {str(retry_e)}")
                print(f"Error type: {type(retry_e)}")
                if hasattr(retry_e, "__dict__"):
                    print(f"Error attributes: {retry_e.__dict__}")
                if attempt == max_retries - 1:
                    raise
                print("Waiting 1 second before retry...")
                await asyncio.sleep(1)  # Short delay between retries
    except Exception as e:
        error_msg = f"Teacher Agent Error during execution: {str(e)}"
        print(f"ERROR: {error_msg}")
        print(f"Error type: {type(e)}")
        if hasattr(e, "__dict__"):
            print(f"Error attributes: {e.__dict__}")
        print("=== Teacher Agent execution failed ===\n")
        return error_msg


async def process_message(session_id: str, user_message: str) -> ChatMessageResponse:
    """Processes the user message using the appropriate agent based on session state."""
    print(f"\n=== Processing message for session {session_id} ===")
    print(f"User message: {user_message}")
    
    if session_id not in sessions:
        print(f"New session {session_id} - initializing state")
        initialize_session_state(session_id)

    # Use .get() to safely retrieve session_state
    session_state = sessions.get(session_id)

    # Handle cases where initialization failed or session_state is None
    if not session_state or session_state.get("initialization_error"):
        error_msg = session_state.get("initialization_error", "Session not initialized properly.") if session_state else "Session not found."
        print(f"ERROR: {error_msg}")
        return ChatMessageResponse(reply=f"Error: Cannot process message. {error_msg}", session_id=session_id, current_stage="error")

    current_stage = session_state.get("current_stage", "onboarding")
    print(f"Current stage: {current_stage}")
    
    message_history: List[ModelMessage] = session_state.get("message_history", [])
    print(f"Message history length: {len(message_history)}")
    
    # Add session_id to state for logging in helper functions if needed
    session_state["session_id"] = session_id

    reply_message = "An unexpected error occurred."  # Default error reply

    # --- Stage: Teaching (after planning complete) ---
    if current_stage == "teaching":
        print("\nProcessing teaching stage...")
        # Run the Teacher Agent with the user's message
        teacher_result = await run_teacher_agent_api(session_state, user_message)

        if isinstance(teacher_result, TeacherResponse):
            # Use the content from the teacher agent as the reply
            reply_message = teacher_result.content

            # Update the current step index if the step is completed
            if teacher_result.completed:
                current_step_index = session_state.get("current_step_index", 0) + 1
                print(f"Step completed - moving to step {current_step_index + 1}")
                session_state["current_step_index"] = current_step_index

                # Check if we've completed all steps
                learning_plan = session_state.get("learning_plan")
                if learning_plan and current_step_index >= len(learning_plan.steps):
                    print("All steps completed!")
                    reply_message += "\n\n**Congratulations!** You've completed all the steps in your learning plan."
                    session_state["current_stage"] = "complete_all"
                else:
                    # Add a note that we're moving to the next step
                    if learning_plan and current_step_index < len(learning_plan.steps):
                        print(f"Moving to next step: {learning_plan.steps[current_step_index]}")
                        reply_message += f"\n\n**Moving to next step:** {learning_plan.steps[current_step_index]}"
        else:
            # Handle error from teacher agent
            print(f"ERROR from Teacher Agent: {teacher_result}")
            reply_message = f"Error running Teacher Agent: {teacher_result}"
            session_state["current_stage"] = "error"

    # --- Handle stage transition from "complete" (planning) to "teaching" ---
    elif current_stage == "complete":
        print("\nProcessing complete stage...")
        # Check if we have the required data for teaching
        user_msg_lower = user_message.lower()
        if any(cmd in user_msg_lower for cmd in ["start", "begin", "continue"]):
            print("User requested to start teaching")
            # Start the teaching process
            learning_plan = session_state.get("learning_plan")
            if learning_plan and learning_plan.steps:
                print("Transitioning to teaching stage")
                session_state["current_stage"] = "teaching"
                session_state["current_step_index"] = 0

                # Run the Teacher Agent for the first step
                teacher_result = await run_teacher_agent_api(
                    session_state, "Let's begin with the first step."
                )

                if isinstance(teacher_result, TeacherResponse):
                    print("First teaching step initialized successfully")
                    # Use the content from the teacher agent as the reply
                    reply_message = (
                        f"**Starting Teaching Process**\n\n{teacher_result.content}"
                    )

                    # Update the current step index if the step is completed (unlikely for first interaction)
                    if teacher_result.completed:
                        print("First step already completed")
                        session_state["current_step_index"] = 1
                else:
                    # Handle error from teacher agent
                    print(f"ERROR starting teaching: {teacher_result}")
                    reply_message = f"Error starting teaching process: {teacher_result}"
                    session_state["current_stage"] = "error"
            else:
                print("ERROR: Learning plan not found")
                reply_message = "Error: Learning plan not found. Cannot start teaching process."
                session_state["current_stage"] = "error"
        else:
            print("Waiting for user to start teaching")
            # Prompt the user to start the teaching process
            reply_message = (
                "Planning is complete. I'm ready to teach you step by step.\n\n"
                "Type 'start' to begin the teaching process."
            )

    # --- Stage: Final completion (after all teaching steps) ---
    elif current_stage == "complete_all":
        print("\nProcessing complete_all stage...")
        reply_message = (
            "You've completed the entire learning plan! Congratulations on reaching your learning goal:\n"
            f"**{session_state.get('onboarding_data').point_b}**\n\n"
            "I hope this has been a productive learning journey for you."
        )

    # --- Stage: Onboarding ---
    elif current_stage == "onboarding":
        print("\nProcessing onboarding stage...")
        onboarding_agent = session_state.get("onboarding_agent")
        if not onboarding_agent:
            print("ERROR: Onboarding Agent not available")
            reply_message = "Error: Onboarding Agent not available for this session."
            session_state["current_stage"] = "error"
            sessions[session_id] = session_state  # Update state before returning
            return ChatMessageResponse(
                reply=reply_message, session_id=session_id, current_stage="error"
            )
        else:
            try:
                print("Running Onboarding Agent...")
                # Assuming onboarding_agent.run is awaitable
                # If run is blocking, use: oa_result = await asyncio.to_thread(onboarding_agent.run, user_message, message_history=message_history)
                oa_result = await onboarding_agent.run(
                    user_message, message_history=message_history
                )

                # Update history immediately after successful run
                print("Updating message history")
                message_history.extend(oa_result.all_messages())
                session_state["message_history"] = message_history

                if isinstance(oa_result.data, OnboardingData):
                    print("Onboarding completed successfully")
                    # --- Onboarding Complete: Auto-trigger PMA and JCA --- #
                    onboarding_data = oa_result.data
                    session_state["onboarding_data"] = onboarding_data
                    reply_message = (  # Initial reply confirming onboarding data
                        f"Great, thank you! Onboarding complete:\n"
                        f"- **Point A:** {onboarding_data.point_a}\n"
                        f"- **Point B:** {onboarding_data.point_b}\n"
                        f"- **Preferences:** {onboarding_data.preferences}\n\n"
                        f"Now determining teaching approach..."
                    )
                    session_state["current_stage"] = (
                        "pedagogy"  # Tentative stage update
                    )

                    print("Running Pedagogical Master Agent...")
                    # Run PMA
                    pma_result = await run_pedagogical_master_api(
                        session_state, onboarding_data
                    )
                    if isinstance(pma_result, PedagogicalGuidelines):
                        print("PMA completed successfully")
                        session_state["pedagogical_guidelines"] = pma_result
                        reply_message = (  # Update reply with guideline
                            f"Great, thank you! Onboarding complete:\n"
                            f"- **Point A:** {onboarding_data.point_a}\n"
                            f"- **Point B:** {onboarding_data.point_b}\n"
                            f"- **Preferences:** {onboarding_data.preferences}\n\n"
                            f"**Suggested Guideline:** {pma_result.guideline}\n\n"
                            f"Now crafting the learning plan..."
                        )
                        session_state["current_stage"] = (
                            "journey_crafting"  # Update stage
                        )

                        print("Running Journey Crafter Agent...")
                        # Run JCA
                        jca_result = await run_journey_crafter_api(
                            session_state, onboarding_data, pma_result
                        )
                        if isinstance(jca_result, LearningPlan):
                             print("JCA completed successfully")
                             plan_steps_text = "\n".join([f"  {i + 1}. {step}" for i, step in enumerate(jca_result.steps)])
                             reply_message = ( # Final reply with plan
                                f"Great, thank you! Onboarding complete:\n"
                                f"- **Point A:** {onboarding_data.point_a}\n"
                                f"- **Point B:** {onboarding_data.point_b}\n"
                                f"- **Preferences:** {onboarding_data.preferences}\n\n"
                                f"**Suggested Guideline:** {pma_result.guideline}\n\n"
                                f"**Proposed Learning Plan:**\n{plan_steps_text}\n\n"
                                f"*(Planning complete! Type 'start' to begin the teaching process.)*"
                            )
                             session_state["learning_plan"] = jca_result # Store the plan
                             session_state["current_stage"] = "complete" # Ready for teaching
                        else: # JCA failed
                            print(f"ERROR from JCA: {jca_result}")
                            reply_message = f"Onboarding and guideline generation complete, but failed to create learning plan: {jca_result}"
                            session_state["current_stage"] = "error" # Set error stage
                    else: # PMA failed
                         print(f"ERROR from PMA: {pma_result}")
                         reply_message = f"Onboarding complete, but failed to determine pedagogical guidelines: {pma_result}"
                         session_state["current_stage"] = "error" # Set error stage

                elif isinstance(oa_result.data, str):
                    print("Onboarding in progress - continuing conversation")
                    # Onboarding agent returned a string (likely asking for more info)
                    reply_message = oa_result.data
                    # Keep stage as "onboarding"
                else:
                    # Unexpected result from onboarding agent
                     print(f"ERROR: Unexpected onboarding result type: {type(oa_result.data)}")
                     reply_message = f"Error: Unexpected response type from Onboarding Agent: {type(oa_result.data)}"
                     session_state["current_stage"] = "error"

            except Exception as e:
                print(f"ERROR during onboarding: {str(e)}")
                reply_message = f"An error occurred during the onboarding process: {e}"
                session_state["current_stage"] = "error"

    # If the stage progressed beyond onboarding within this call, it's handled above.
    # If it's still pedagogy/journey_crafting, it means something went wrong or is unexpected.
    # This case shouldn't normally be hit if OA always transitions or returns a string.
    elif current_stage in ["pedagogy", "journey_crafting"]:
         print(f"ERROR: Unexpected stage '{current_stage}'")
         reply_message = f"Error: Unexpected state '{current_stage}'. Processing should happen automatically after onboarding completes successfully."
         session_state["current_stage"] = "error"

    print(f"\nFinal stage: {session_state.get('current_stage')}")
    print(f"Updating session state for {session_id}")
    # Update the master sessions dictionary with the final state for this session_id
    sessions[session_id] = session_state

    print("=== Message processing complete ===\n")
    return ChatMessageResponse(
        reply=reply_message,
        session_id=session_id,
        current_stage=session_state.get("current_stage"),
    )


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
        print("ERROR: OPENROUTER_API_KEY not configured")
        # Check moved to initialization, but keep a check here for robustness
        raise HTTPException(
            status_code=500, detail="OPENROUTER_API_KEY not configured."
        )

    if not request.session_id:
        print("ERROR: Missing session_id")
        raise HTTPException(status_code=400, detail="session_id is required.")

    # Basic input validation
    if not request.message:
        print("ERROR: Empty message")
        raise HTTPException(status_code=400, detail="message cannot be empty.")

    print("Processing message...")
    # Process the message using our async helper function
    response = await process_message(request.session_id, request.message)  # Use await

    # Check if process_message resulted in an error state
    if response.current_stage == "error":
        print(f"ERROR in response: {response.reply}")
        # You might want different status codes depending on the error cause
        # e.g., 500 for internal server/agent errors, 400 for bad requests (though FastAPI handles some)
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
    print("\n=== Starting FastAPI server ===")
    print(f"API Key Loaded: {bool(OPENROUTER_API_KEY)}")
    print("Starting server on http://0.0.0.0:8000")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 