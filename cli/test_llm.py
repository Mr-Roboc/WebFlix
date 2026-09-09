import os


from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("OPEN_ROUTER_API_KEY")

if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

def llm_query(user_query:str):
    messages = [
        {
        "role": "system",
        "content": """Fix any spelling errors in the user-provided movie search query below.Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
         Preserve punctuation and capitalization unless a change is required for a typo fix.
         If there are no spelling errors, or if you're unsure, output the original query unchanged.
          Output only the final query text, nothing else."""
    
        },

        {

        "role":"user",

        "content": f"{user_query}"
        }

        
    ]

    model_id = "meta-llama/llama-3.3-70b-instruct"

    response  = client.chat.completions.create(messages=messages,model = model_id)
    return response.choices[0].message.content
    



