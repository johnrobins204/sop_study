import json
import sys
from pathlib import Path
from datetime import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--new-root", default="/home/johnro/sop-research/gnosis", help="New repo root path to set in backlog")
args = parser.parse_args()

backlog = Path("backlog/refactor_backlog.json")
if not backlog.exists():
    print("ERROR: backlog file not found at", backlog, file=sys.stderr)
    sys.exit(2)

# backup
bak = backlog.with_suffix(backlog.suffix + f".bak.{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}")
bak.write_bytes(backlog.read_bytes())

data = json.loads(backlog.read_text(encoding="utf-8"))
meta = data.setdefault("metadata", {})
old = meta.get("repo_root")
meta["repo_root"] = args.new_root
backlog.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

print("Updated backlog/refactor_backlog.json")
print("old repo_root:", old)
print("new repo_root:", meta["repo_root"])