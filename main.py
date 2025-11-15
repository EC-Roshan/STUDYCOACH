"""
EduTech Multi-Agent Learning Platform
Backend with ADK, Gemini API Integration, and FastAPI
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import google.generativeai as genai
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# GEMINI API CONFIGURATION
# ============================================================================
GEMINI_API_KEY = "AIzaSyCy3t8Mv-L0YBQtYIpEDnZh44iUCelRoSI"
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini model
model = genai.GenerativeModel('gemini-1.5-flash')

# ============================================================================
# FASTAPI SETUP
# ============================================================================
app = FastAPI(title="EduTech Multi-Agent Platform")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================
class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = "default_user"

class AgentResponse(BaseModel):
    agent_name: str
    response: str
    status: str

# ============================================================================
# AGENT SYSTEM PROMPTS
# ============================================================================

MAIN_AGENT_PROMPT = """You are the Main Routing Agent for EduTech platform. 
Your ONLY job is to analyze the user's query and return ONE WORD - the name of the best subagent to handle it.

Available agents:
- tutor_agent: For explaining concepts, teaching, learning help, how things work
- code_analyzer: For code review, debugging, programming help, syntax checking
- exam_prep: For quizzes, tests, exam preparation, practice questions
- language_agent: For grammar, vocabulary, translations, language learning
- career_agent: For career guidance, job advice, skill recommendations, courses
- analytics_agent: For progress tracking, performance metrics, statistics

Rules:
1. Return ONLY ONE WORD - the agent name
2. No explanations, no punctuation, just the agent name
3. Choose the most appropriate agent based on keywords

User Query: {query}

Best Agent (one word only):"""

AGENT_PROMPTS = {
    "greeting_agent": """You are the Greeting Agent. Welcome users warmly and introduce the EduTech platform's capabilities in 2-3 sentences.""",
    
    "tutor_agent": """You are an expert Tutor Agent. Explain the following topic clearly and concisely:

User Query: {query}

Provide a helpful educational response (keep it under 200 words):""",

    "code_analyzer": """You are a Code Analyzer Agent. Review the following code:

User Query: {query}

Provide constructive feedback with:
1. Issues found (if any)
2. Suggestions for improvement
3. Code quality score (0-100)

Keep response under 200 words:""",

    "exam_prep": """You are an Exam Preparation Agent. For the topic:

User Query: {query}

Create:
1. 3 practice questions
2. Brief study tips

Keep response under 200 words:""",

    "language_agent": """You are a Language Helper Agent. For the following:

User Query: {query}

Provide:
1. Grammar/spelling check if text provided
2. Vocabulary help or translation if requested
3. Writing tips

Keep response under 200 words:""",

    "career_agent": """You are a Career Guidance Agent. For:

User Query: {query}

Provide:
1. Relevant skills to learn
2. Course/certification suggestions
3. Career path advice

Keep response under 200 words:""",

    "analytics_agent": """You are an Analytics Agent. For:

User Query: {query}

Provide:
1. Simulated progress metrics
2. Strengths and areas for improvement
3. Recommendations

Keep response under 200 words:"""
}

# ============================================================================
# AGENT CLASSES
# ============================================================================

class LlmAgent:
    """Base Agent class"""
    
    def __init__(self, name: str, system_prompt: str):
        self.name = name
        self.system_prompt = system_prompt
        self.model = model
    
    def generate_response(self, user_query: str) -> str:
        """Generate response using Gemini API"""
        try:
            full_prompt = self.system_prompt.format(query=user_query)
            response = self.model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}"

class MainAgent(LlmAgent):
    """Main routing agent"""
    
    def route_query(self, user_query: str) -> str:
        """Determine which subagent should handle the query"""
        try:
            prompt = MAIN_AGENT_PROMPT.format(query=user_query)
            response = self.model.generate_content(prompt)
            agent_name = response.text.strip().lower()
            
            valid_agents = [
                "tutor_agent", "code_analyzer", "exam_prep", 
                "language_agent", "career_agent", "analytics_agent"
            ]
            
            return agent_name if agent_name in valid_agents else "tutor_agent"
                
        except Exception as e:
            print(f"Routing error: {str(e)}")
            return "tutor_agent"

# ============================================================================
# INITIALIZE AGENTS
# ============================================================================

main_agent = MainAgent("edu_main_agent", MAIN_AGENT_PROMPT)

subagents = {
    "greeting_agent": LlmAgent("greeting_agent", AGENT_PROMPTS["greeting_agent"]),
    "tutor_agent": LlmAgent("tutor_agent", AGENT_PROMPTS["tutor_agent"]),
    "code_analyzer": LlmAgent("code_analyzer", AGENT_PROMPTS["code_analyzer"]),
    "exam_prep": LlmAgent("exam_prep", AGENT_PROMPTS["exam_prep"]),
    "language_agent": LlmAgent("language_agent", AGENT_PROMPTS["language_agent"]),
    "career_agent": LlmAgent("career_agent", AGENT_PROMPTS["career_agent"]),
    "analytics_agent": LlmAgent("analytics_agent", AGENT_PROMPTS["analytics_agent"])
}

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "active",
        "service": "EduTech Multi-Agent Platform",
        "version": "1.0.0",
        "message": "Backend is running! Open index.html in your browser to use the UI.",
        "available_agents": list(subagents.keys())
    }

@app.post("/query")
async def process_query(request: QueryRequest):
    """Main endpoint to process user queries"""
    try:
        user_query = request.query
        
        print(f"\n{'='*60}")
        print(f"Received query: {user_query}")
        
        selected_agent_name = main_agent.route_query(user_query)
        print(f"Routed to: {selected_agent_name}")
        
        if selected_agent_name not in subagents:
            selected_agent_name = "tutor_agent"
        
        selected_agent = subagents[selected_agent_name]
        
        print(f"Generating response...")
        agent_response = selected_agent.generate_response(user_query)
        print(f"Response generated successfully")
        print(f"{'='*60}\n")
        
        return JSONResponse(content={
            "agent_name": selected_agent_name,
            "response": agent_response,
            "status": "success"
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "agent_name": "error",
                "response": f"An error occurred: {str(e)}",
                "status": "error"
            }
        )

@app.post("/agent/{agent_name}")
async def query_specific_agent(agent_name: str, request: QueryRequest):
    """Directly query a specific agent"""
    try:
        if agent_name not in subagents:
            raise HTTPException(
                status_code=404, 
                detail=f"Agent '{agent_name}' not found"
            )
        
        agent = subagents[agent_name]
        response = agent.generate_response(request.query)
        
        return JSONResponse(content={
            "agent_name": agent_name,
            "response": response,
            "status": "success"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents")
async def list_agents():
    """List all available agents"""
    return {
        "agents": [
            {"name": "greeting_agent", "description": "Welcomes users and introduces the platform"},
            {"name": "tutor_agent", "description": "Explains concepts step-by-step"},
            {"name": "code_analyzer", "description": "Reviews and provides feedback on code"},
            {"name": "exam_prep", "description": "Generates quizzes and study materials"},
            {"name": "language_agent", "description": "Helps with grammar, vocabulary, and translations"},
            {"name": "career_agent", "description": "Provides career guidance and course recommendations"},
            {"name": "analytics_agent", "description": "Tracks progress and provides performance insights"}
        ]
    }

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    import sys

    # Load server configuration from environment
    host = os.getenv('HOST', '127.0.0.1')
    port = int(os.getenv('PORT', 8000))

    # Set UTF-8 encoding for Windows console
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except:
            pass

    print("\n" + "="*70)
    print("Starting EduTech Multi-Agent Platform...")
    print("="*70)
    print(f"Backend Server: http://{host}:{port}")
    print(f"API Documentation: http://{host}:{port}/docs")
    print("Frontend: Open 'index.html' file in your browser")
    print("="*70 + "\n")
    uvicorn.run(app, host=host, port=port, log_level="info")