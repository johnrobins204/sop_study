# ...new file...
{ adds CLI subcommand 'orchestrate' that calls orchestrator.orchestrate }

import argparse
import sys
from typing import List

from src.orchestrator import orchestrate


def run(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(prog="sop-orchestrator")
    parser.add_argument("--config", required=True, help="Path to run YAML config")
    args = parser.parse_args(argv)

    res = orchestrate(args.config)
    if res.get("success"):
        return 0
    else:
        # print minimal error info to stderr
        for e in res.get("errors", []):
            print(f"ERROR: {e}", file=sys.stderr)
        return 5


if __name__ == "__main__":
    raise SystemExit(run())