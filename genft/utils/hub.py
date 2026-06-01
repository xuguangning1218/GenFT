from __future__ import annotations

from pathlib import Path


def resolve_modelscope_path(path: str, subfolder: str | None = None, cache_dir: str | None = None) -> str:
    """Resolve a local path or a modelscope:// model id to a local filesystem path."""
    if path.startswith("modelscope://"):
        try:
            from modelscope import snapshot_download
        except ImportError as exc:
            raise ImportError("Install modelscope or pass a local adapter path.") from exc

        model_id = path.removeprefix("modelscope://")
        local_path = Path(snapshot_download(model_id, cache_dir=cache_dir))
    else:
        local_path = Path(path)

    if subfolder:
        local_path = local_path / subfolder
    return str(local_path)

