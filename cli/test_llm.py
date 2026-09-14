import os


from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("OPENROUTER_API_KEY")

if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

def llm_query(user_query:str,enhance):

    if enhance=="spell":
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

    elif enhance =="rewrite":
        messages = [
            {
               "role":"system",
               "content": f"""Rewrite the user-provided movie search query below to be more specific and searchable.
            
            Consider:
            - Common movie knowledge (famous actors, popular films)
            - Genre conventions (horror = scary, animation = cartoon)
            - Keep the rewritten query concise (under 10 words)
            - It should be a Google-style search query, specific enough to yield relevant results
            - Don't use boolean logic

            Examples:
            - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
            - "movie about bear in london with marmalade" -> "Paddington London marmalade"
            - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

            If you cannot improve the query, output the original unchanged.
            Output only the rewritten query text, nothing else.
            
            """},

     {
        "role":"user",
         "content":f"{user_query}"
               
    }
        ]

    elif enhance =="expand":
        messages = [
             {
                  "role":"system",
                  "content":f"""Expand the user-provided movie search query below with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Output only the additional terms; they will be appended to the original query.

Examples:
- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"""
             },

             {
              "role":"user",
              "content":f"{user_query}"
             }
        ]

        
        
        

    model_id = "meta-llama/llama-3.3-70b-instruct"

    response  = client.chat.completions.create(messages=messages,model = model_id)
    return response.choices[0].message.content
    



