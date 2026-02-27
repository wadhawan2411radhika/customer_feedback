#!/usr/bin/env python3
"""CLI entry point for RAG feedback search system."""

import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv

from src.indexer import FeedbackIndexer
from src.llm_client import OpenAIClient
from src.rag_pipeline import RAGPipeline
from src.retriever import FeedbackRetriever

# Load environment variables from .env file
load_dotenv()


async def index_command(data_dir: Path, index_path: Path) -> None:
    """Build vector index from feedback summaries."""
    print(f"Indexing feedback summaries from {data_dir}...")
    indexer = FeedbackIndexer(data_dir=data_dir, index_path=index_path)
    count = await indexer.index_all()
    print(f"Indexed {count} feedback summaries successfully.")


async def query_command(
    query: str,
    index_path: Path,
    model: str,
    top_k: int,
) -> None:
    """Query the RAG system."""
    retriever = FeedbackRetriever(index_path=index_path, top_k=top_k)
    llm_client = OpenAIClient(model=model)
    pipeline = RAGPipeline(retriever=retriever, llm_client=llm_client)

    print(f"Querying: {query}\n")
    print("Answer: ", end="", flush=True)

    async for chunk in pipeline.query(query):
        print(chunk, end="", flush=True)
    print()

    if hasattr(pipeline.llm_client, "last_cost"):
        print(f"\n--- Cost ---\n{pipeline.llm_client.last_cost}")


async def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RAG Feedback Search System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Command to execute"
    )

    index_parser = subparsers.add_parser("index", help="Build vector index")
    index_parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory with feedback summaries (default: data)",
    )
    index_parser.add_argument(
        "--index-path",
        type=Path,
        default=Path(".chroma_db"),
        help="Path to ChromaDB index (default: .chroma_db)",
    )

    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument(
        "query",
        type=str,
        help="Query string",
    )
    query_parser.add_argument(
        "--index-path",
        type=Path,
        default=Path(".chroma_db"),
        help="ChromaDB index path (default: .chroma_db)",
    )
    query_parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)",
    )
    query_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to retrieve (default: 5)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "index":
        await index_command(args.data_dir, args.index_path)
    elif args.command == "query":
        await query_command(
            args.query, args.index_path, args.model, args.top_k
        )


if __name__ == "__main__":
    asyncio.run(main())
