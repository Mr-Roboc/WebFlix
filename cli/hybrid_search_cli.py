import argparse

from lib.hybrid_search import normalize_score,weighted_search,rrf_search
from test_llm import llm_query,llm_rerank_batch,llm_rerank_query,cross_encoder_func
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
    rrf_search_parser.add_argument("--rerank_method", type=str,choices=['individual','batch','cross_encoder'],help = 'individual rerank')
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

                rrf_search(llm_response,args.k,args.limit)

            elif args.enhance=="rewrite":
                rewrite_response = llm_query(args.query,args.enhance)
                print(f"Enhanced query (REWRITE): '{args.query}' --> '{rewrite_response}'\n")
                rrf_search(rewrite_response,args.k,args.limit)

            elif args.enhance=="expand":
                expand_response = llm_query(args.query,args.enhance)
                print(f"Enhanced query (EXPAND): '{args.query}' --> '{expand_response}'\n")
                rrf_search(expand_response,args.k,args.limit)


            elif args.rerank_method=="individual":
                RETRIEVAL_LIMIT = args.limit*5
                search_result = rrf_search(args.query,args.k,RETRIEVAL_LIMIT,args.rerank_method)


                reranked_results = []
                for idx, result in enumerate(search_result[:RETRIEVAL_LIMIT], 1):
                    print(f"Reranking {idx}/{RETRIEVAL_LIMIT}: {result['doc_title']}")
                    score = llm_rerank_query(args.query, result)
                    result['rerank_score'] = score
                    reranked_results.append(result)
                                    
                            
                reranked_results.sort(key=lambda x : x["rerank_score"],
                         reverse=True)
                            
                print(
                        f"\nRe-ranking top {RETRIEVAL_LIMIT} candidates "
                        f"using individual method...")
                            
                print(f"Reciprocal Rank Fusion Results for "
                        f"'{args.query}' (k={args.k}):\n")


                for idx,result in enumerate(reranked_results[:args.limit],1):
                        print(f"{idx}. {result['doc_title']}")
                        print(f"Reranking: {result['rerank_score']:.3f}/10")
                        print(f"RRF Score:{result['rrf_score']:.3f}")
                        print(f"BM25 Rank: {result['bm25_rank']}, Semantic Rank: {result['semantic_rank']}\n ")
                


            
                            


            elif args.rerank_method=="batch":
                RETRIEVAL_LIMIT = args.limit*5
                rrf_result = rrf_search(args.query,args.k,RETRIEVAL_LIMIT,args.rerank_method)
                batch_result = llm_rerank_batch(
                    args.query,
                    rrf_result[:RETRIEVAL_LIMIT],
                    RETRIEVAL_LIMIT,
                )



                print(f"\nRe-ranking top {RETRIEVAL_LIMIT} candidates "
                                        f"using batch method...")
                                            
                print(f"Reciprocal Rank Fusion Results for "
                                        f"'{args.query}' (k={args.k}):\n")
                

                for idx,result in enumerate(batch_result[:args.limit],1):
                    print(f"{idx}. {result['doc_title']}")
                    print(f"Batch rank: {result['batch_rank']}")
                    print(f"RRF Score:{result['rrf_score']:.3f}")
                    print(f"BM25 Rank: {result['bm25_rank']}, Semantic Rank: {result['semantic_rank']}\n ")


            elif args.rerank_method=="cross_encoder":
                RETRIEVAL_LIMIT = args.limit*5
                rrf_search_result = rrf_search(args.query,args.k,RETRIEVAL_LIMIT,"cross_encoder")
                cross_encoder_results = cross_encoder_func(
                    args.query,
                    rrf_search_result,
                    args.limit,
                )

                print(
                    f"\nRe-ranking top {RETRIEVAL_LIMIT} results "
                    f"using cross_encoder method..."
                )
                print(
                    f"Reciprocal Rank Fusion Results for "
                    f"'{args.query}' (k={args.k}):\n"
                )

                for idx, result in enumerate(cross_encoder_results, 1):
                    print(f"{idx}. {result['doc_title']}")
                    print(
                        f"   Cross Encoder Score: "
                        f"{result['cross_encoder_score']:.3f}"
                    )
                    print(f"   RRF Score: {result['rrf_score']:.3f}")
                    print(
                        f"   BM25 Rank: {result['bm25_rank']}, "
                        f"Semantic Rank: {result['semantic_rank']}"
                    )

                    description = result.get("document", "")
                    if len(description) > 100:
                        description = description[:100] + "..."
                    print(f"   {description}\n")

if __name__ == "__main__":
    main()
