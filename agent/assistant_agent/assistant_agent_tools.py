import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

from langchain_openai import AzureChatOpenAI
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from util.constants import ModelDetails


# Setup Azure OpenAI client for AutoGen
model_client = AzureOpenAIChatCompletionClient(
    azure_deployment=ModelDetails.AZURE_DEPLOYMENT_NAME,
    model=ModelDetails.AZURE_MODEL_NAME,
    api_version=ModelDetails.AZURE_API_VERSION,
    azure_endpoint=ModelDetails.AZURE_ENDPOINT,
    api_key=ModelDetails.AZURE_API_KEY,
)


# LangChain LLM wrapper
def get_llm() -> BaseChatModel:
    return AzureChatOpenAI(
        azure_deployment=ModelDetails.AZURE_DEPLOYMENT_NAME,
        azure_endpoint=ModelDetails.AZURE_ENDPOINT,
        api_key=ModelDetails.AZURE_API_KEY,
        temperature=0,
        api_version=ModelDetails.AZURE_API_VERSION
    )

class GetCapitalInput(BaseModel):
    country: str

# Tool to get capital of a given country
async def get_capital(input: GetCapitalInput) -> str:
    """Returns the capital of a given country."""
    llm = get_llm()
    prompt = f"What is the capital of {input.country}?"
    return llm.invoke(prompt).content

class DistanceInput(BaseModel):
    city1: str
    city2: str

# Tool to get distance between two cities
async def distance_between(input: DistanceInput) -> str:
    """Returns approximate distance between two major cities."""
    llm = get_llm()
    prompt = f"What is the approximate distance between {input.city1} and {input.city2}?"
    return llm.invoke(prompt).content


# Create the Assistant Agent
agent = AssistantAgent(
    name="geo_agent",
    model_client=model_client,
    tools=[get_capital, distance_between],
    system_message=(
        "You are a helpful assistant with deep knowledge about world geography, "
        "including countries, capitals, continents, mountains, rivers, and cultures."
    ),
    reflect_on_tool_use=True,
    model_client_stream=True,
)


# Main function to run the agent with console input/output
async def main() -> None:
    await Console(
        agent.on_messages_stream(
            [TextMessage(content="How far is spanish capital from peru's?", source="User")],
            cancellation_token=CancellationToken()))
    await model_client.close()


# Run the async main
if __name__ == "__main__":
    asyncio.run(main())
