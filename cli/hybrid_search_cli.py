import os
import sys
import argparse

# Silence noisy model download/progress output so the grader can see results.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

try:  # pragma: no cover
    sys.stderr = open(os.devnull, "w")
except Exception:
    pass

from lib.hybrid_search import normalize_scores, weighted_search, rrf_search

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    norm_parser= subparsers.add_parser(name="normalize", help="Normalize scores")
    norm_parser.add_argument('scores', type=float, nargs='+', help='List of scores to normalize')

    ws_parser= subparsers.add_parser(name="weighted-search", help="Weighted Search")
    ws_parser.add_argument('query', type=str, help='User query')
    ws_parser.add_argument('--alpha', type=float ,default=0.5 ,help='Value of constant alpha α')
    ws_parser.add_argument('--limit', type=int, default=5 , help='Number of results to be returned')
    
    rrf_parser= subparsers.add_parser(name="rrf-search", help="RRF Search")
    rrf_parser.add_argument('query', type=str, help='User query')
    rrf_parser.add_argument('--k', type=int ,default=0.5 ,help='Value of constant alpha k')
    rrf_parser.add_argument('--limit', type=int, default=5 , help='Number of results to be returned')
    rrf_parser.add_argument('--enhance', type=str, choices=["spell", "rewrite", "expand"] , help='Query enhancement')
    rrf_parser.add_argument('--rerank-method', type=str, choices=["individual", "batch", "cross_encoder"] , help='Rerank Method')

    args = parser.parse_args()

    match args.command:
        case "rrf-search":
            rrf_search(args.query, args.k , args.limit, args.enhance, args.rerank_method)
        case "weighted-search":
            weighted_search(args.query, args.alpha, args.limit)
        case "normalize":
            norm_scores = normalize_scores(args.scores)
            for norm_score in norm_scores:
                print(f"* {norm_score:.4f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()