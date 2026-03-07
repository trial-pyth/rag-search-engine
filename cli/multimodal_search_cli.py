import argparse
import os

# Silence noisy model download/progress output so the grader can see results.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from lib.multimodal_search import verify_image_embedding

def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser(
        name="verify_image_embedding", help="Verify image embeddings load"
    )
    verify_parser.add_argument("image_fpath", type=str, help="Path to the image to process")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image_fpath)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
