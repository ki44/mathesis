import re
from pathlib import Path


def load_prompt(path: Path | str) -> str:
    """Load a .md prompt file, stripping HTML comments (markdownlint directives)."""
    text = Path(path).read_text(encoding="utf-8")
    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL).strip()
