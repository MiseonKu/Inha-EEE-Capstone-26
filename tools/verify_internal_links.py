from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

PACKAGE_PREFIXES = (
    'data/',
    'ml/',
    'optimizer/',
    'experiment_preparation/',
    'hardware/',
    'paper_reference/',
    'docs/',
    'environment/',
    'tools/',
    'README.md',
    'MANIFEST.md',
    'FILE_INDEX.txt',
)
BACKTICK_RE = re.compile(r'`([^`]+)`')
CHECKSUM_RE = re.compile(r'^[0-9a-f]{64}\s+(.+)$')
SOURCE_KEYS = {'source', 'source_root', 'source_roots', 'source_file', 'source_directory', 'source_zip', 'source_url'}


def iter_json_strings(value: Any, key: str = '') -> Iterable[str]:
    if key in SOURCE_KEYS:
        return
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item_key, item in value.items():
            yield from iter_json_strings(item, str(item_key))
    elif isinstance(value, list):
        for item in value:
            yield from iter_json_strings(item, key)


def scan_paths(root: Path) -> list[Path]:
    paths: list[Path] = [root / 'README.md', root / 'MANIFEST.md', root / 'FILE_INDEX.txt']
    paths.extend((root / 'docs').glob('*.md'))
    paths.extend((root / 'environment').glob('*.md'))
    paths.extend((root / 'data' / 'processed').rglob('*.json'))
    paths.extend((root / 'data' / 'processed_manifest').glob('*.json'))
    paths.extend((root / 'data' / 'provenance_builders').glob('*.md'))
    paths.extend((root / 'data' / 'checksums').glob('*.txt'))
    paths.extend((root / 'ml').rglob('*.json'))
    paths.extend((root / 'ml').rglob('*.md'))
    return sorted({path for path in paths if path.exists() and path.is_file()})


def candidate_paths(path: Path) -> Iterable[str]:
    text = path.read_text(encoding='utf-8', errors='replace')
    if path.suffix == '.json':
        try:
            yield from iter_json_strings(json.loads(text))
        except json.JSONDecodeError:
            return
    elif path.name.endswith('_sha256.txt') or path.name == 'FILE_INDEX.txt':
        for line in text.splitlines():
            match = CHECKSUM_RE.match(line)
            yield match.group(1) if match else line.strip()
    else:
        for match in BACKTICK_RE.finditer(text):
            yield match.group(1)


def normalize(raw: str) -> str:
    return raw.strip().strip('<>').replace(chr(92), '/')


def is_package_path(value: str) -> bool:
    if not value or '*' in value or ';' in value or value.startswith(('http://', 'https://')):
        return False
    if ' ' in value and not value.endswith(('.step', '.pdf')):
        return False
    return value.startswith(PACKAGE_PREFIXES)


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
    missing: list[str] = []
    checked = 0
    for path in scan_paths(root):
        for raw in candidate_paths(path):
            value = normalize(raw)
            if not is_package_path(value):
                continue
            checked += 1
            if not (root / value).exists():
                missing.append(f'{path.relative_to(root)} -> {value}')
    if missing:
        print('FAIL internal_links')
        for item in missing[:100]:
            print(item)
        return 1
    print(f'PASS internal_links: checked {checked} package-relative references')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
