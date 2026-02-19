from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os


app = FastAPI(
    title="LangChain Chat API",
    version="1.0.0"
)

# Initialize LLM
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model = ChatHuggingFace(llm=llm)

# In-memory chat history (single user demo)
chat_history = [
    SystemMessage(content="You are a General Chatbot")
]


# -------- Request Model --------
class ChatRequest(BaseModel):
    message: str


# -------- Routes --------

@app.post("/chat")
def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Add user message
    chat_history.append(HumanMessage(content=request.message))

    # Invoke model
    result = model.invoke(chat_history)

    # Add AI message
    chat_history.append(AIMessage(content=result.content))

    return {
        "response": result.content
    }


@app.post("/clear")
def clear_chat():
    chat_history.clear()
    chat_history.append(SystemMessage(content="You are a General Chatbot"))
    return {"message": "Chat history reset"}
