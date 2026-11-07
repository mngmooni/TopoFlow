"""Guard explicitly requested reproduction outputs against silent overwrites."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def prepare_outputs(*paths: Path, overwrite: bool = False) -> None:
    """Create parent directories only after rejecting colliding outputs."""
    paths = [Path(p).expanduser().resolve() for p in paths if p is not None]
    if len(set(paths)) != len(paths):
        raise ValueError("The same output path was requested more than once")
    for p in paths:
        if p.is_dir():
            raise IsADirectoryError(str(p))
        if p.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite {p}. Use --overwrite deliberately, or choose a new --output path.")
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
