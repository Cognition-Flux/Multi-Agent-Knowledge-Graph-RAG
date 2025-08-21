"""Test the LangChain AWS Bedrock client to invoke a model.

uv run dev/aws_bedrock/claude_langchain.py
"""

# %%
import os

from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv
from langchain_aws import ChatBedrock, ChatBedrockConverse


# Load environment variables from .env file
load_dotenv(override=True)
MODEL_ID = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
CONVERSE_MODEL = "anthropic.claude-3-5-haiku-20241022-v1:0"
BEDROCK_REGION = os.getenv("AWS_BEDROCK_REGION", "us-west-2")


llm1 = ChatBedrock(
    region_name=BEDROCK_REGION,
    model_id=MODEL_ID,
    model_kwargs={"temperature": 0.7},
)

llm2 = ChatBedrockConverse(
    model=CONVERSE_MODEL,
    region_name=BEDROCK_REGION,
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_session_token=os.getenv("AWS_BEARER_TOKEN_BEDROCK"),
)


if __name__ == "__main__":
    import asyncio

    async def main():
        """Simple examples invoking Bedrock clients."""
        try:
            response_message = await llm1.ainvoke("Hello, Claude! How are you today?")
            print(response_message.content)
        except (ClientError, BotoCoreError, ValueError) as exc:
            print(f"No se pudo invocar el modelo {llm1}: {exc}")
        try:
            response_message = await llm2.ainvoke("Hello, Claude! How are you today?")
            print(response_message.content)
        except (ClientError, BotoCoreError, ValueError) as exc:
            print(f"No se pudo invocar el modelo {llm2}: {exc}")

    asyncio.run(main())
