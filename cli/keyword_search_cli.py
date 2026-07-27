#!/usr/bin/env python3

import argparse

from lib.search_utils import BM25_K1, BM25_B
from lib.keyword_search import build_command,bm25_tf_command,search_command,InvertedIndex

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build the inverted index")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    search_parser= subparsers.add_parser("tf", help = "Frequency of a term")
    search_parser.add_argument("doc_id",type = int,help="Document Id")
    search_parser.add_argument("term",type=str,help="term frequency")

    search_parser = subparsers.add_parser("idf",help = "Inverse document frequency")
    search_parser.add_argument("term",type=str,help="term frequency")

    search_parser = subparsers.add_parser("tfidf", help = "Term Frequency - Inverse Document Frequency")

    search_parser.add_argument("doc_id",type = int,help="Document Id")
    search_parser.add_argument("term",type=str)

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25tf")
    bm25_tf_parser.add_argument("doc_id",type=int,help="Document iD")
    bm25_tf_parser.add_argument("term",type=str,help = "The term")
    bm25_tf_parser.add_argument("k1",type=float,nargs='?',default=BM25_K1,help="Tunable parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")


    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    
    
    

    args = parser.parse_args()

    
    idx= InvertedIndex()
    match args.command:
        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully.")

        case "tf":
            try:
                idx.load()
            except FileNotFoundError:
                print("Error: Index not found. Please run 'build' first.")
                return
            print("ID and Term ", args.doc_id, args.term)
            print("The frequency ", idx.tf(args.doc_id, args.term))

        case "idf":
            print(f"Printing document frequency of '{args.term}':{idx.idf(args.term):.2f}")
        case "tfidf":
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {idx.tfidf(args.term,args.doc_id):.2f}")

        case "bm25idf":
            
            print(f"BM25 IDF score of '{args.term}': {idx.bm25idf(args.term):.2f}")


        case "bm25tf":
            score = bm25_tf_command(args.doc_id, args.term, args.k1,args.b)
            
            print(
                f"BM25 TF score of {args.term} in document {args.doc_id}: {score:.2f}"
                
            )

        case "bm25search":
            idx.load()
            print(f"{args.query}")
            
        case "search":
            print("Searching for:", args.query)
            results = search_command(args.query)
            for i, res in enumerate(results, 1): # this is a comment
                
                print(f"{i}. ({res['id']}) {res['title']}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

