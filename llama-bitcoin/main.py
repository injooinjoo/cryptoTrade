import transformers
import torch
from dotenv import load_dotenv
import os

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

load_dotenv()

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    token=os.getenv("HUGGINGFACE_TOKEN")
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

outputs = pipeline(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])
