import argparse
import json
from lib.hybrid_search import rrf_search
def main()->None:

    parser= argparse.ArgumentParser(description ="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default = 5,
        help = "Number of results to evaluate (precision@k)",
    )



    args = parser.parse_args()
    limit = args.limit

    if limit <= 0:
        parser.error("--limit must be greater than zero")

    with open("data/golden_dataset.json") as f:
        golden_data = json.load(f)

    print(f"k={limit}")
    print()

    for case in golden_data["test_cases"]:
        query = case["query"]
        relevant = case["relevant_docs"]

        results = rrf_search(
            query=query,
            k=60,
            limit=limit,
            rerank_method="individual",
        )
        retrieved = [result["doc_title"] for result in results]


        # Precision Metric
        matching_titles = set(retrieved) & set(relevant)
        precision = len(matching_titles) / limit

        # Recall metric
        total_relevant = len(set(case["relevant_docs"]))

        recall = len(matching_titles)/total_relevant

        print(
            f"- Query: {query}\n"
            f"  - Precision@{limit}: {precision:.4f}\n"
            f"  - Recall@{limit}: {recall:.4f}\n"
            f"  - Retrieved: {', '.join(retrieved)}\n"
            f"  - Relevant: {', '.join(relevant)}\n"
        )





if __name__ == "__main__":
    main()
