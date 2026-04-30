import pandas as pd
import os
from openai import OpenAI
from dotenv import load_dotenv

chunk_size = 10 
input_file = './input.csv'
output_file = './output.csv'
load_dotenv()

reader = pd.read_csv(input_file, chunksize=chunk_size)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key = os.getenv('OPENAI_API_KEY')
)

response = client.chat.completions.create(
    model="openai/gpt-oss-20b:free",
    messages=[
        {"role": "user", "content": ""}
    ]
)

print(response.choices[0].message.content)