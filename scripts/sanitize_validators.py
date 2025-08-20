import re
from pathlib import Path
from datetime import datetime
import shutil

ROOT = Path.cwd()
SCRIPTS = ROOT / "scripts"
DEST = SCRIPTS / "validators"
BACKUP = SCRIPTS / f"validators_backup_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
DEST.mkdir(exist_ok=True)
BACKUP.mkdir(exist_ok=True)

moved = []
for f in sorted(SCRIPTS.glob("validate_task*.py")):
    try:
        text = f.read_text(encoding="utf-8")
        # Normalize parent index to parents[2] (repo root relative to scripts/validators)
        text = re.sub(r"Path\(__file__\)\.resolve\(\)\.parents\[\s*1\s*\]", "Path(__file__).resolve().parents[2]", text)
        text = re.sub(r"Path\(__file__\)\.resolve\(\)\.parents\[\s*0\s*\]", "Path(__file__).resolve().parents[2]", text)

        # Prepend safe sys.path header if none present
        if "sys.path.insert" not in text:
            header = (
                "import sys\n"
                "from pathlib import Path\n"
                "# ensure repo root is on sys.path so 'src' imports resolve when running here\n"
                "ROOT = Path(__file__).resolve().parents[2]\n"
                "sys.path.insert(0, str(ROOT))\n\n"
            )
            text = header + text

        # backup original
        shutil.copy2(str(f), str(BACKUP / f.name))
        dest_path = DEST / f.name
        dest_path.write_text(text, encoding="utf-8")
        f.unlink()
        moved.append(str(dest_path))
    except Exception as e:
        print("failed to process", f, ":", e)

# README for validators folder
readme = DEST / "README.md"
readme.write_text(
    "Validator scripts consolidated here.\n\n"
    "Run from repo root with repo on PYTHONPATH:\n\n"
    "  PYTHONPATH=. python scripts/validators/validate_taskX.py\n\n"
    f"Backups saved to: {BACKUP}\n",
    encoding="utf-8",
)

print("Sanitization complete. Moved files:")
for p in moved:
    print(" -", p)
print("Backups saved to:", BACKUP)