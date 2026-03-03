#!/usr/bin/env python3
import argparse

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
    chunk_subparser.add_argument("--overlap", type = int, default = 0, help = "Number of overlapping words while chunking")
    chunk_subparser.add_argument("--chunk-size", type = int, default = 200, help = "Number of words in each fixed sized chunk")
    
    semantic_chunk_subparser = subparsers.add_parser("semantic_chunk", help="Semantic chunk a document")
    semantic_chunk_subparser.add_argument("text", type = str, help = "Document to be chunked" )
    semantic_chunk_subparser.add_argument("--overlap", type = int, default = 0, help = "Number of overlapping words while chunking")
    semantic_chunk_subparser.add_argument("--max-chunk-size", type = int, default = 4, help = "Number of words in each fixed sized chunk")

    embed_chunk_subparser = subparsers.add_parser("embed_chunks", help="Create embeddings for semantic chunks")

    search_chunk_subparser = subparsers.add_parser("search_chunked", help="Search query")
    search_chunk_subparser.add_argument("query", type = str, help = "User query to be searched" )
    search_chunk_subparser.add_argument("--limit", type = int, default = 5, help = "User specidifed limit for search results"  )



    args = parser.parse_args()

    # Delay importing heavy/optional deps until after argparse runs so `-h` works
    # even when dependencies aren't installed.
    from lib import semantic_search as ss

    match args.command:
        case "search_chunked":
            ss.search_chunked(args.query, args.limit)
        case "embed_chunks" :
            ss.embed_chunks()
        case "semantic_chunk":
            ss.chunk_text_semantic(args.text, args.overlap ,args.max_chunk_size)
        case "chunk":
            ss.chunk_text(args.text, args.overlap ,args.chunk_size)
        case "search":
            ss.search(args.query, args.limit)
        case "embedquery":
            ss.embed_query_text(args.query)
        case "verify_embeddings":
            ss.verify_embeddings()
        case "embed_text":
            ss.embed_text(args.text)
        case "verify":
            ss.verify_model()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
