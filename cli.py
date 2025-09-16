import argparse
import json
import sys
import os
from url_processor import URLProcessor

def main():
    parser = argparse.ArgumentParser(description='ECE 46100 | Team 8 - CLI Package Tester')

    # Command OR URL file
    parser.add_argument("command_or_file", help="Use 'install', 'test', or provide a path to a URL file.")

    args = parser.parse_args()

    if args.command_or_file == "install":
        print("Installing dependencies...")
        # More logic to add here later
        sys.exit(0)

    elif args.command_or_file == "test":
        print("Running tests...")
        # More logic to add here later
        sys.exit(0)

    elif os.path.exists(args.command_or_file):
        print(f"Processing URL File: {args.command_or_file}")
        processor = URLProcessor(args.command_or_file)
        results = processor.process_urls()
        for entry in results:
            print(json.dumps(entry))
        sys.exit(0)

    else:
        print("ERROR: Invalid command or file not found")
        sys.exit(1)

if __name__ == '__main__':
    main()