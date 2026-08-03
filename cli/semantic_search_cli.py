import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings,embed_query
import lib.search_utils

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command",required =True)


    verify_parser = subparsers.add_parser("verify",help="Verify the model")
    

    embed_parser = subparsers.add_parser("embed",help="Embed the text")
    embed_parser.add_argument("text",type=str, help ="Enter the text")


    verify_embed = subparsers.add_parser("verify_embed",help ="Verify the embedding")


    query_embed = subparsers.add_parser("embed_query",help ="Embeds the query")

    query_embed.add_argument("query",help = "Enter the query")
    
    
    # Parse incoming termainl execution

    args= parser.parse_args()

    
    match args.command:
        
#        case _:
 #           parser.print_help()

        case "verify":
            verify_model()

        case "embed":
            print(f"Encoding: {args.text} \n ")
            embed_text(args.text)

        case "verify_embed":
            verify_embeddings()

        case "embed_query":
            embed_query(args.query)

if __name__ == "__main__":
    main()
