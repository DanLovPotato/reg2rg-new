"""Read-only integrity check of the historical source, not a training smoke test."""
import hashlib
import tarfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
expected = '35c5cf2f08e33170abde5905d13962bd1c3f59f4860c111c2bb76eaa81727ce2'
for directory in ['chtc_training', 'chtc_validation_full']:
    payload = root / 'repro' / directory / 'code.tar.gz'
    assert hashlib.sha256(payload.read_bytes()).hexdigest() == expected, payload
    count = 0
    with tarfile.open(payload) as archive:
        for member in archive:
            if member.isdir():
                continue
            assert member.isfile(), member.name
            rel = Path(member.name).relative_to('code')
            assert '..' not in rel.parts
            # macOS tar stores Finder extended attributes as AppleDouble entries.
            # These are metadata, not Python/source files; retain them in the archive only.
            if any(part.startswith('._') for part in rel.parts):
                continue
            assert (root / rel).read_bytes() == archive.extractfile(member).read(), rel
            count += 1
    print(f'PASS: {directory} payload SHA-256 and {count} source files match.')
