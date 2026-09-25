import os
import re


from openai import OpenAI
from dotenv import load_dotenv
import json


from sentence_transformers import CrossEncoder

load_dotenv()
api_key = os.environ.get("OPENROUTER_API_KEY")

if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    timeout=60.0,
)


model_id = "meta-llama/llama-3.3-70b-instruct"
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

        
        
        

    

    response  = client.chat.completions.create(messages=messages,model = model_id)
    return response.choices[0].message.content
    


def llm_rerank_query(query: str, doc: list[dict]):

    messages = [
        {
            "role": "system",
            "content": """You are a movie search relevance scorer.

Your task is to score how relevant a movie is to a user's search query.

Scoring:
10 = Perfect match
9 = Extremely relevant
8 = Very relevant
7 = Relevant
6 = Somewhat relevant
5 = Weakly relevant
4 = Slightly relevant
3 = Barely relevant
2 = Very poor match
1 = Almost completely irrelevant
0 = Completely irrelevant

Consider:
- The user's intent
- The movie title
- The movie description
- How directly the movie satisfies the query

Be discriminative. Do not give the same score to every movie.

Output ONLY one integer from 0 to 10."""
        },

        {
            "role": "user",
            "content": f"""
Search query:
{query}

Movie title:
{doc.get("doc_title", "")}

Movie description:
{doc.get("document", "")}

Score:"""
        }
    ]

    response = client.chat.completions.create(
        messages=messages,
        model=model_id
    )

    raw_response = response.choices[0].message.content.strip()

    print(f"LLM response: {raw_response}")

    match = re.search(r'\b(10|[0-9])\b', raw_response)

    if not match:
        raise ValueError(
            f"Could not extract score from LLM response: {raw_response}"
        )

    return float(match.group(1))




def llm_rerank_batch(query:str,documents,limit:int):
    if not documents:
        return []


    doc_map  = {}
    doc_list:list[str] = []

    for doc in documents:
        doc_id = doc["id"]


        doc_map[doc_id] = doc

        title = doc.get("doc_title", doc.get("title", ""))
        description = doc.get("document", doc.get("description", ""))
        doc_list.append(f"{doc_id}: {title} - {description[:200]}")


        doc_list_str = "\n".join(doc_list)

    # Batch ranking prompt
    prompt = f"""Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies:
{doc_list_str}

Return the movie IDs in order of relevance, best match first.

Your response must be a raw JSON array of integers.
Do not wrap the JSON in Markdown. Do not use a ```json code block.
Do not include any explanatory text.

For example:
[75, 12, 34, 2, 1]

Ranking:"""

    # ONE LLM CALL
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        max_tokens=max(64, len(documents) * 5),
    )

    # Extract response text
    ranking_text = (
        response.choices[0].message.content or ""
    ).strip()

    # Be tolerant of models that still wrap valid JSON in a code fence.
    if ranking_text.startswith("```"):
        ranking_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", ranking_text).strip()

    # Convert JSON string → Python list
    parsed_ids = json.loads(ranking_text)

    # Rebuild documents in LLM ranking order
    reranked = []

    seen_ids = set()
    for i, doc_id in enumerate(parsed_ids):
        if doc_id in doc_map:
            seen_ids.add(doc_id)
            reranked.append(
                {
                    **doc_map[doc_id],
                    "batch_rank": i + 1
                }
            )

    # Keep any valid candidates omitted by the model instead of silently
    # shrinking the result set.
    for doc in documents:
        if doc["id"] not in seen_ids:
            reranked.append({**doc, "batch_rank": len(reranked) + 1})

 
    return reranked[:limit]
     


def cross_encoder_func(query: str, document: list[dict], limit: int) -> list[dict]:
  
    cross_encode = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")

    pairs = []
    for doc in document:
        pairs.append([query, f"{doc.get('doc_title', '')} - {doc.get('document', '')}"])

    # returns a list of numbers for each pair
    scores = cross_encode.predict(pairs)

    ranked_data = sorted(
        (
            {
                **doc,
                "cross_encoder_score": float(score),
            }
            for doc, score in zip(document, scores)
        ),
        key=lambda result: result["cross_encoder_score"],
        reverse=True,
    )

    return ranked_data[:limit]

  

def evaluate_llm(query,document):


    formatted_result = []

    for doc in document:
        doc_id = doc["id"]

        title = doc["doc_title"]
        description = doc["document"][:50]

        formatted_result.append(f"{doc_id}: {title} - {description}")

    
    prompt=f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_result)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers other than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""

    response = client.chat.completions.create(

    model = model_id,
    messages = [{
        "role":"user",
        "content":prompt
    }
    ],
    temperature=0

    )


    response_text = (response.choices[0].message.content or "").strip()

    if response_text.startswith("```"):
            response_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", response_text).strip()
    

    parsed_response = json.loads(response_text)

    if not isinstance(parsed_response, list):
        raise ValueError("LLM evaluation response must be a JSON list")

    if len(parsed_response) != len(document):
        raise ValueError(
            "LLM evaluation returned "
            f"{len(parsed_response)} scores for {len(document)} documents"
        )

    score_docs = []
    for doc, score in zip(document, parsed_response):

        if score not in (0, 1, 2, 3):
            
            raise ValueError(f"Invalid LLM relevance score: {score!r}")

      
        score_docs.append({**doc, "score": score})

    return score_docs





def rag_llm(query:str,search_results:list[dict]) -> str:
    docs = [f"Title: {doc['doc_title']} , Description: {doc['document']}" for doc in search_results]

    docs_string = "\n---\n".join(docs)

    prompt = f"""You are a RAG agent for Webflix, a movie streaming service.
Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
Provide a comprehensive answer that addresses the user's query.

Query: {query}

Documents:
{docs_string}

Answer:"""

    messages = [

        {
        "role":"system",
        "content": prompt

        }
]

    response = client.chat.completions.create(model=model_id,messages=messages,temperature=0)

    return (response.choices[0].message.content or "").strip()




def llm_summarize(query,search_results:list[dict]) -> str:
    results = [f"Title: {doc['doc_title']} , Description: {doc['document']}" for doc in search_results]
    
    results_string = "\n---\n".join(results)



    prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

This should be tailored to Webflix users. Webflix is a movie streaming service.

Query: {query}

Search results:
{results_string}

Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""

    response = client.chat.completions.create(

    model = model_id,
    messages = [{
        "role":"system",
        "content":prompt
    }
    ],
    temperature=0

    )


    return (response.choices[0].message.content or "").strip()

    


def llm_citations(query,documents):
    results = [f"Title: {doc['doc_title']} , Description: {doc['document']}" for doc in documents]
        
    results_string = "\n---\n".join(results)

    

    prompt = f"""Answer the query below and give information based on the provided documents.

The answer should be tailored to users of Webflix, a movie streaming service.
If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

Query: {query}

Documents:
{results_string}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources in the format [1], [2], etc. when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the provided documents, say "I don't have enough information"
- Be direct and informative

Answer:"""


    response= client.chat.completions.create(
    model= model_id,

    messages = [{
    "role":"system",
    "content":prompt
}],
temperature= 0
    )


    return (response.choices[0].message.content or "").strip()
