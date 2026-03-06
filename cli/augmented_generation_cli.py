import argparse
from lib.rag import rag, doc_summarization, citations_documents, answer_detailed_question

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    
    summarization_parser = subparsers.add_parser(
        "summarize", help="Perform RAG (search + summarize)"
    )
    summarization_parser.add_argument("query", type=str, help="Search query for RAG")
    summarization_parser.add_argument("--limit", type=int, default = 5,  help="Search limit")
    
    cit_parser = subparsers.add_parser(
        "citations", help="Perform RAG (search + citations)"
    )
    cit_parser.add_argument("query", type=str, help="Search query for RAG")
    cit_parser.add_argument("--limit", type=int, default = 5,  help="Search limit")
    
    qa_parser = subparsers.add_parser(
        "citations", help="Perform RAG (search + question + answer)"
    )
    qa_parser.add_argument("query", type=str, help="Search query for RAG")
    qa_parser.add_argument("--limit", type=int, default = 5,  help="Search limit")

    args = parser.parse_args()

    match args.command:
        case "rag":
            rag(args.query)
        case "summarize":
            doc_summarization(args.query, args.limit)
        case "citations":
            citations_documents(args.query, args.limit)
        case "question":
            answer_detailed_question(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()