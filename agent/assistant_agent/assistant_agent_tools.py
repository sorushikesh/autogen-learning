from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import AsyncGenerator
import logging

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

from langchain_openai import AzureChatOpenAI
from langchain_core.language_models import BaseChatModel

from util.constants import ModelDetails

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_client = AzureOpenAIChatCompletionClient(
    azure_deployment=ModelDetails.AZURE_DEPLOYMENT_NAME,
    model=ModelDetails.AZURE_MODEL_NAME,
    api_version=ModelDetails.AZURE_API_VERSION,
    azure_endpoint=ModelDetails.AZURE_ENDPOINT,
    api_key=ModelDetails.AZURE_API_KEY,
)


def get_llm() -> BaseChatModel:
    return AzureChatOpenAI(
        azure_deployment=ModelDetails.AZURE_DEPLOYMENT_NAME,
        azure_endpoint=ModelDetails.AZURE_ENDPOINT,
        api_key=ModelDetails.AZURE_API_KEY,
        temperature=0,
        api_version=ModelDetails.AZURE_API_VERSION,
    )


class GetCapitalInput(BaseModel):
    country: str


async def get_capital(input: GetCapitalInput) -> str:
    """Returns the capital of a given country."""
    llm = get_llm()
    prompt = f"What is the capital of {input.country}?"
    return llm.invoke(prompt).content


class DistanceInput(BaseModel):
    city1: str
    city2: str


async def distance_between(input: DistanceInput) -> str:
    """Returns approximate distance between two major cities."""
    llm = get_llm()
    prompt = (
        f"What is the approximate distance between {input.city1} and {input.city2}?"
    )
    return llm.invoke(prompt).content


tools = [get_capital, distance_between]

agent = AssistantAgent(
    name="geo_agent",
    model_client=model_client,
    tools=tools,
    system_message=(
        "You are a helpful assistant with deep knowledge about world geography, "
        "including countries, capitals, continents, mountains, rivers, and cultures."
    ),
    reflect_on_tool_use=True,
    model_client_stream=True,
)


class ChatRequest(BaseModel):
    message: str


async def run_agent(question: str) -> AsyncGenerator[str, None]:
    messages = [TextMessage(content=question, source="User")]
    stream = agent.on_messages_stream(messages, cancellation_token=CancellationToken())

    async for message in stream:
        if hasattr(message, "content") and message.content:
            yield f"data: {message.content}\n\n"


@app.post("/api/chat")
async def chat(request: ChatRequest):
    return StreamingResponse(run_agent(request.message), media_type="text/event-stream")
