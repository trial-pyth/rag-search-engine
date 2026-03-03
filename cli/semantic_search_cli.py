#!/usr/bin/env python3
from sympy.polys.polyconfig import query

import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, search, chunk_text

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

    chunk_subparser = subparsers.add_parser("chunk", help="Chunk a document")
    chunk_subparser.add_argument("text", type = str, help = "Document to be chunked" )
    chunk_subparser.add_argument("--overlap", type = int, help = "Number of overlapping words while chunking")
    chunk_subparser.add_argument("--chunk-size", type = int, default = 200, help = "Number of words in each fixed sized chunk")

    args = parser.parse_args()

    match args.command:
        case "chunk":
            chunk_text(args.text, args.overlap ,args.chunk_size)
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