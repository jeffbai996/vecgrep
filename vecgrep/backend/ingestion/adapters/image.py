from __future__ import annotations

from pathlib import Path
from typing import Iterator

from .base import Adapter, AdapterError, Document
from .registry import register_adapter
from ..visual_queue import VisualQueue, enabled


@register_adapter
class ImageAdapter(Adapter):
    name = 'image'

    @classmethod
    def matches(cls, source: str) -> bool:
        if not enabled() or source.startswith(('http://', 'https://')):
            return False
        path = Path(source)
        return path.is_file() and path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp', '.tif', '.tiff', '.bmp'}

    def load(self, source: str) -> Iterator[Document]:
        path = Path(source)
        try:
            queue = VisualQueue.default()
            text = queue.result(path) or ''
            state = queue.state(path)
        except (OSError, ValueError) as exc:
            raise AdapterError(f'Could not queue image {source}: {exc}') from exc
        yield Document(str(path.resolve()), text, {
            'adapter': self.name, 'path': str(path.resolve()),
            'visual_state': state, 'visual_original': str(path.resolve()),
            'visual_generated': True,
        })
