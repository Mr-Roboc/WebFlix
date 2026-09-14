import argparse

from lib.hybrid_search import normalize_score,weighted_search,rrf_search
from test_llm import llm_query
def main()->None:
    parser= argparse.ArgumentParser(description= "Hybrid Search")

    

    subparsers = parser.add_subparsers(dest="command", help = "available commands")


    norm = subparsers.add_parser("normalize",help = 'Normalize the score')
    norm.add_argument("scores",type = list, nargs = '*',help = "Enter the score list")

    weighted_search_parser = subparsers.add_parser("weighted-search",help="Performs weighted search")

    weighted_search_parser.add_argument("query",help="Enter the query")
    weighted_search_parser.add_argument("--alpha", type = float, help = "Parameter to control the weight between BM25 and semantic")
    weighted_search_parser.add_argument("--limit",type = int,help = "THe limit for results")

    rrf_search_parser = subparsers.add_parser("rrf-search",help="Reciprocal rank fusion")
    rrf_search_parser.add_argument("--enhance",type=str,choices=['spell','rewrite','expand'],help="Query  enhancement methdod")
    rrf_search_parser.add_argument("--rerank_method", type=str,choices=['individual'],help = 'individual rerank')
    rrf_search_parser.add_argument("query",help="input query")
    rrf_search_parser.add_argument("-k",type = int,help ="Constant parameter")
    rrf_search_parser.add_argument("--limit",type = int,help ="Result limit")

    

    args = parser.parse_args()

    match args.command:
        case '_':
            parser.print_help()

        case "normalize":
            norm_scores = normalize_score(args.scores)

            for norm_score in norm_scores:
                print(f"* {norm_score:.4f}")

        case "weighted-search":
            weighted_search(args.query,args.alpha,args.limit)
             

        case "rrf-search":
            if args.enhance=="spell":
                llm_response = llm_query(args.query,args.enhance)

                print(f"Enhanced query (SPELL): '{args.query}' -> '{llm_response}'\n")

                rrf_search(llm_response.args.k,args.limit)

            elif args.enhance=="rewrite":
                rewrite_response = llm_query(args.query,args.enhance)
                print(f"Enhanced query (REWRITE): '{args.query}' --> '{rewrite_response}'\n")
                rrf_search(rewrite_response,args.k,args.limit)

            elif args.enhance=="expand":
                expand_response = llm_query(args.query,args.enhance)
                print(f"Enhanced query (EXPAND): '{args.query}' --> '{expand_response}'\n")
                rrf_search(expand_response,args.k,args.limit)


            elif args.rerank_method=="individual":
                rrf_search(args.query,args.k,args.limit*5)
                

                




if __name__ == "__main__":
    main()

