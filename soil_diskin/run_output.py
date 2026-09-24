"""Preserve existing runs and record completion or interruption consistently."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterator


def require_empty_output(output: Path) -> None:
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('use a new or empty output directory; existing runs are preserved')


@contextmanager
def run_record(path: Path, metadata: dict) -> Iterator[None]:
    """Record lifecycle while callers remain responsible for their partial tables."""
    metadata.update(status='running', started_utc=datetime.now(timezone.utc).isoformat())
    path.write_text(json.dumps(metadata, indent=2)+'\n')
    try:
        yield
    except BaseException:
        metadata['status'] = 'interrupted_or_failed'
        raise
    else:
        metadata['status'] = 'complete'
    finally:
        path.write_text(json.dumps(metadata, indent=2)+'\n')
