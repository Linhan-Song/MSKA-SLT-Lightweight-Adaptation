import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Combine metrics from completed experiments.")
    parser.add_argument("metrics", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = {}
    for metrics_path in args.metrics:
        with open(metrics_path, "r", encoding="utf-8") as handle:
            result = json.load(handle)
        summary[result["experiment"]] = result
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
