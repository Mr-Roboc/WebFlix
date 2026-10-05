import mimetypes

import argparse
import os

from openai import OpenAI

from dotenv import load_dotenv

import base64

load_dotenv()


def main() -> None:


    parser = argparse.ArgumentParser(description="Multimodal Search")
    
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    multimodal = subparser.add_parser("image", help="Add Image")
    multimodal.add_argument("--image", dest="sub_image", help="Path to the image file")
    multimodal.add_argument("--query", dest="sub_query", help="Query to rewrite based on image")

    args = parser.parse_args()

    image_path = args.image or getattr(args, "sub_image", None)
    query = args.query or getattr(args, "sub_query", None)
    if not image_path or not query:
        parser.error("--image and --query are required")

    if not os.path.isfile(image_path):
        parser.error(f"image file not found: {image_path}")

    mime, _ = mimetypes.guess_type(image_path)
    mime = mime or "image/jpeg"

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    
    if not api_key:
        raise KeyError("OPENROUTER API environment varaible not set")
    
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    
    
    model_id = os.environ.get("OPENROUTER_VISION_MODEL", "google/gemini-2.5-flash")

    system_prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary"""

    data_url = f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"

    messages = [
        {
            "role":"user",
            "content":[

                {"type":"text","text":system_prompt.strip()},
                {"type":"image_url", "image_url": {"url": data_url}},
                {"type":"text", "text": query.strip()},
                   
            ],
        }
    ]

    response = client.chat.completions.create(
        messages=messages,
        model=model_id,
        max_tokens=512,
    )


    content = response.choices[0].message.content
    print(f"Rewritten query: {content.strip()}")
    
    if response.usage is not None:
        print(f"Total tokens:    {response.usage.total_tokens}")
    

   


if __name__ == "__main__":
    main()
        
