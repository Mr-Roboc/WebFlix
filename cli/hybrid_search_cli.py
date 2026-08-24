import argparse

from lib.hybrid_search import normalize_score,weighted_search
def main()->None:
    parser= argparse.ArgumentParser(description= "Hybrid Search")

    

    subparsers = parser.add_subparsers(dest="command", help = "available commands")


    norm = subparsers.add_parser("normalize",help = 'Normalize the score')
    norm.add_argument("scores",type = list, nargs = '*',help = "Enter the score list")

    weighted_search_parser = subparsers.add_parser("weighted-search",help="Performs weighted search")

    weighted_search_parser.add_argument("query",help="Enter the query")
    weighted_search_parser.add_argument("--alpha", type = float, help = "Parameter to control the weight between BM25 and semantic")
    weighted_search_parser.add_argument("--limit",type = int,help = "THe limit for results")

    
    
    

    args = parser.parse_args()

    match args.command:
        case '_':
            parser.print_help()

        case "normalize":
            norm_scores = normalize_scores(args.scores)

            for norm_score in norm_scores:
                print(f"* {norm_score:.4f}")

        case "weighted-search":
            weighted_search(args.query,args.alpha,args.limit)
             


            




if __name__ == "__main__":
    main()

