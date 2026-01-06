# fix_numpy_aliases.py
import pathlib
import re

ROOT = pathlib.Path('.')  # repo root; run from project root
FILE_GLOB = '**/*.py'
REPLACEMENTS = [
    (r'\bnp\.int\b', 'int'),
    (r'\bnp\.float\b', 'float'),
    (r'\bnp\.bool\b', 'bool'),
    (r'\bnp\.object\b', 'object'),
    # If you prefer numpy scalar types instead, change above to np.int32 etc.
]

def fix_file(path):
    text = path.read_text(encoding='utf-8')
    new_text = text
    for pat, repl in REPLACEMENTS:
        new_text = re.sub(pat, repl, new_text)
    if new_text != text:
        backup = path.with_suffix(path.suffix + '.bak')
        path.write_text(new_text, encoding='utf-8')
        backup.write_text(text, encoding='utf-8')
        print(f"Patched {path} (backup: {backup})")

def main():
    files = list(ROOT.glob(FILE_GLOB))
    for f in files:
        # skip virtualenv and .git folders
        if 'venv' in str(f).split(pathlib.Path.sep) or '.git' in str(f):
            continue
        fix_file(f)

if __name__ == '__main__':
    main()
