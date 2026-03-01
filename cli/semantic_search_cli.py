#!/usr/bin/env python3

import argparse
from lib.semantic_search import verify_model, embed_text

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_subparser = subparsers.add_parser("verify", help="Verifying the embedding model loads properly")

    embed_subparser = subparsers.add_parser("embed_text", help="Encode text with embedding model")
    embed_subparser.add_argument("text", type = str, help = "Text to be encoded" )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()