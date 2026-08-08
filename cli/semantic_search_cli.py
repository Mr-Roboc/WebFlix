import argparse
from lib.semantic_search import verify_model, verify_embeddings,embed_query,search,chunk_text
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

    search_parser = subparsers.add_parser("search", help = "Semantic search")
    search_parser.add_argument("query",help = "String query")
    search_parser.add_argument("--limit",type= int,help="Number of results")

    chunk_parser = subparsers.add_parser("chunk" , help="Chunks the document")
    chunk_parser.add_argument("text",help="The text to be chunked")
    chunk_parser.add_argument("--chunk-size",type = int,help = "Chunk size")
    chunk_parser.add_argument("--overlap",type = int,help = "overlaps the chunk")



    
    

    
    
    
    
    # Parse incoming termainl execution

    args= parser.parse_args()

    
    match args.command:


        case "chunk":
              chunk_text(args.text, args.overlap,args.chunk_size)

        case "verify":
            verify_model()

        case "embed":
            print(f"Encoding: {args.text} \n ")
            embed_text(args.text)

        case "verify_embed":
            verify_embeddings()

        case "embed_query":
            embed_query(args.query)

        case "search":
            search(args.query,args.limit)
    
    
            
if __name__ == "__main__":
    main()
