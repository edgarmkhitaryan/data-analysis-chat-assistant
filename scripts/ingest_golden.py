"""Build (or rebuild) the Golden Bucket embedding index from curated Trios.

Reads the Trio JSON files, embeds each question with the configured embedding
model, and writes the cached index that the retriever loads at query time. Run it
with ``make ingest`` after adding or editing Trios.

Note: the learning loop promotes *learned* Trios **automatically** at runtime
(plan/010 §3) — there is no manual promotion step. This script is for the curated
authoring path (analysts adding/editing Trios in ``data/golden_trios/``).
"""

import sys
from pathlib import Path

# Make ``src/`` importable when run directly, even before an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rich.console import Console  # noqa: E402 — must follow the path bootstrap

from assistant.config import get_settings  # noqa: E402
from assistant.golden.index import build_index, load_trios  # noqa: E402

console = Console()


def main() -> int:
    settings = get_settings()
    trios = load_trios(settings.golden_trios_dir)
    if not trios:
        console.print(f"[red]No Trios found in {settings.golden_trios_dir}[/]")
        return 1

    console.print(
        f"Embedding [cyan]{len(trios)}[/] trios with [cyan]{settings.embedding_model}[/] …"
    )
    count = build_index(settings.golden_trios_dir, settings.golden_index_dir, settings)
    console.print(
        f"[green]Built Golden Bucket index[/] — {count} trios -> {settings.golden_index_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
