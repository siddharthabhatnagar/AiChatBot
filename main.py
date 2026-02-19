from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typing import List
import os


app = FastAPI(
    title="LangChain Chat API",
    version="2.0.0"
)

# ---------------- LLM ----------------

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model = ChatHuggingFace(llm=llm)


# ---------------- Request Models ----------------

class Message(BaseModel):
    role: str   # "system" | "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


# ---------------- Routes ----------------

@app.post("/chat")
def chat(request: ChatRequest):

    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty")

    # Convert incoming messages to LangChain format
    chat_history = []

    for msg in request.messages:
        if msg.role == "system":
            chat_history.append(SystemMessage(content=msg.content))
        elif msg.role == "user":
            chat_history.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            chat_history.append(AIMessage(content=msg.content))
        else:
            raise HTTPException(status_code=400, detail="Invalid role type")

    # Invoke model
    result = model.invoke(chat_history)

    return {
        "response": result.content
    }
