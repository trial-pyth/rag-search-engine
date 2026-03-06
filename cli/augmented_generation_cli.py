import argparse
from lib.rag import rag, doc_summarization

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    
    summarization_parser = subparsers.add_parser(
        "summzarize", help="Perform RAG (search + summzarize)"
    )
    summarization_parser.add_argument("query", type=str, help="Search query for RAG")
    summarization_parser.add_argument("--limit", type=int, help="Search limit")

    args = parser.parse_args()

    match args.command:
        case "rag":
            rag(args.query)
            query = args.query
        case "summzarize":
            doc_summarization(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()