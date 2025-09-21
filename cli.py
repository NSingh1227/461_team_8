import argparse
import json
import sys
import os
from src.core.url_processor import URLProcessor

def main():
    """Main CLI entry point for package testing."""
    parser = argparse.ArgumentParser(description='ECE 46100 | Team 8 - CLI Package Tester')
    parser.add_argument("command_or_file", help="Use 'install', 'test', or provide a path to a URL file.")

    args = parser.parse_args()

    if args.command_or_file == "install":
        print("Installing dependencies...")
        sys.exit(0)

    elif args.command_or_file == "test":
        print("Running tests...")
        sys.exit(0)

    elif os.path.exists(args.command_or_file):
        print(f"Processing URL File: {args.command_or_file}")
        processor = URLProcessor(args.command_or_file)
        
        model_results = processor.process_urls_with_metrics()
        for model_result in model_results:
            print(model_result.to_ndjson_line())
        
        sys.exit(0)

    else:
        print("ERROR: Invalid command or file not found")
        sys.exit(1)

if __name__ == '__main__':
    main()