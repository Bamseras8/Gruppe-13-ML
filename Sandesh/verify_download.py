"""Check that the extracted package contains the original input and code files."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'checksums.json').read_text())
failures = []
for name, expected in manifest.items():
    file = root / name
    if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != expected:
        failures.append(name)
if failures:
    raise SystemExit('Missing or changed files:\n' + '\n'.join(failures))
print(f'Download check passed: {len(manifest)} files are complete and unchanged.')
