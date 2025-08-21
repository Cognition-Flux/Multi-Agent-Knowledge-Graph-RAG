# %%
import os

import boto3
from dotenv import load_dotenv


load_dotenv(override=True)

# If you already set the API key as an environment variable, you can comment this line out
os.environ["AWS_BEARER_TOKEN_BEDROCK"] = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
print(os.environ["AWS_BEARER_TOKEN_BEDROCK"])
# Create an Amazon Bedrock client
client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-west-2",  # If you've configured a default region, you can omit this line
)

# Define the model and message
model_id = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
messages = [{"role": "user", "content": [{"text": "Hello"}]}]

response = client.converse(
    modelId=model_id,
    messages=messages,
)
print(response)
# %%
