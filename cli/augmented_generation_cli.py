import argparse


from lib.hybrid_search import rrf_search
from test_llm import rag_llm,llm_summarize
def main() ->None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation")
    subparsers = parser.add_subparsers(dest="command", help ="Available commands")


    rag_parser = subparsers.add_parser("rag",help = "Peform RAG (search + generate answer)")

    rag_parser.add_argument("query",type=str,help="Search query for RAG")

    summarizer = subparsers.add_parser("summarize",help = "Summarizes the RAG results")
    summarizer.add_argument("query", type=str, help = "Provide the search query")
    summarizer.add_argument("--limit", type=int, default=5,help = "Result limit")




    args = parser.parse_args()

    match args.command:
        case "rag":
            rrf_results = rrf_search(args.query,k=60,limit=5)
            print(rag_llm(args.query,rrf_results))


        case "summarize":
            rrf_result = rrf_search(args.query,k=60,limit=args.limit)

            print("\n LLM Summary: \n")
            print(llm_summarize(args.query,rrf_result))




        case _:
            parser.print_help()



if __name__ == "__main__":
    main()


