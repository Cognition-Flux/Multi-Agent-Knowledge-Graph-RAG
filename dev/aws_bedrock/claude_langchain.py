# %%
from dotenv import load_dotenv
from langchain_aws import ChatBedrock


# Load environment variables from .env file
load_dotenv(override=True)
MODEL_ID = "us.anthropic.claude-3-5-haiku-20241022-v1:0"

llm = ChatBedrock(
    region_name="us-west-2",
    model_id=MODEL_ID,
    model_kwargs={"temperature": 0.7},
)

if __name__ == "__main__":
    import asyncio

    async def main():
        """This is a simple example of how to use the LangChain AWS Bedrock client to invoke a model."""
        response_message = await llm.ainvoke("Hello, Claude! How are you today?")
        print(response_message.content)

    asyncio.run(main())
