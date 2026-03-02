#!/usr/bin/env python3
from sympy.polys.polyconfig import query

import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, search

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_subparser = subparsers.add_parser("verify", help="Verifying the embedding model loads properly")

    embed_subparser = subparsers.add_parser("embed_text", help="Encode text with embedding model")
    embed_subparser.add_argument("text", type = str, help = "Text to be encoded" )

    verify_embedding_subparser = subparsers.add_parser("verify_embeddings", help="Verify Embeddings")

    query_embedding_subparser = subparsers.add_parser("embedquery", help="Encode query")
    query_embedding_subparser.add_argument("query", type = str, help = "User query to be encoded" )

    search_subparser = subparsers.add_parser("search", help="Search query")
    search_subparser.add_argument("query", type = str, help = "User query to be searched" )
    search_subparser.add_argument("--limit", type = int, default = 5, help = "User specidifed limit for search results"  )

    args = parser.parse_args()

    match args.command:
        case "search":
            search(args.query)
        case "embedquery":
            embed_query_text(args.query)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_text":
            embed_text(args.text)
        case "verify":
            verify_model()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()