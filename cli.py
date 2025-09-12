import argparse, sys
from url_processor import URLProcessor

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="URL Processor CLI")
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--out", default="")
    args = p.parse_args(argv)

    processor = URLProcessor()
    results = processor.process_file(args.input)
    payload = processor.to_json(results)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(payload)
        print(f"Wrote {args.out}")
    else:
        print(payload)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
