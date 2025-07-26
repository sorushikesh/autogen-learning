import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

from util.constants import ModelDetails

model_client = AzureOpenAIChatCompletionClient(
    azure_deployment=ModelDetails.AZURE_DEPLOYMENT_NAME,
    model=ModelDetails.AZURE_MODEL_NAME,
    api_version=ModelDetails.AZURE_API_VERSION,
    azure_endpoint=ModelDetails.AZURE_ENDPOINT,
    api_key=ModelDetails.AZURE_API_KEY,
)

# Create AssistantAgent specialized in world geography
agent = AssistantAgent(
    name="geo_agent",
    model_client=model_client,
    tools=[],
    system_message="You are a helpful assistant with deep knowledge about world geography, including countries, capitals, continents, mountains, rivers, and cultures.",
    reflect_on_tool_use=False,  # No tools needed for simple Q&A
    model_client_stream=True,
)


# Main function to run the agent
async def main() -> None:
    result = await Console(agent.run_stream(task="What is the highest mountain in the world?"))
    print(result)
    print(f"AI Message : {result.messages[-1].content}")
    await model_client.close()


asyncio.run(main())
