# %%

import boto3
from dotenv import load_dotenv


load_dotenv(override=True)


# Create an Amazon Bedrock client
client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-west-2",  # If you've configured a default region, you can omit this line
)

# Define the model and message
MODEL_ID = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
messages = [{"role": "user", "content": [{"text": "Hello"}]}]

response = client.converse(
    modelId=MODEL_ID,
    messages=messages,
)
print(response)
# %%
