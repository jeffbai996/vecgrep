"""Service-host worker: python -m vecgrep.backend.ingestion.visual_worker."""
from __future__ import annotations

import argparse
import json
import logging
import os
import threading
from pathlib import Path
from typing import Callable

import httpx

from .visual_processor import VisualProcessor
from .visual_queue import VisualQueue

logger = logging.getLogger(__name__)


def process_one(queue: VisualQueue, *, processor: Callable[[Path], tuple[str, bool, str]]) -> bool:
    job = queue.claim()
    if job is None:
        return False
    try:
        if not queue.current(job):
            queue.retry(job, 'Source was changed or removed')
            return True
        text, complete, error = processor(Path(job['source']))
        if complete:
            queue.complete(job, text)
        else:
            queue.retry(job, error, text=text)
            logger.info('Visual enrichment pending for %s: %s', job['source'], error)
    except Exception as exc:
        # A malformed source cannot terminate the queue consumer.
        queue.retry(job, f'{type(exc).__name__}: {exc}')
        logger.warning('Visual enrichment failed for %s: %s', job['source'], exc)
    return True


def deliver(queue: VisualQueue, *, url: str, client: httpx.Client) -> int:
    delivered = 0
    for item in queue.deliveries():
        try:
            # The server registers targets during explicit index calls. A
            # delivery only revisits those sources with their original chunker.
            response = client.post(f'{url.rstrip("/")}/api/index', json={
                'source': item['source'], 'corpus': item['corpus'],
                'chunker': item['chunker'], 'force': False, 'visual_key': item['key'],
            })
            response.raise_for_status()
            body = response.json()
            if body.get('failed'):
                raise ValueError('Index request reported failed sources')
            queue.delivered(item)
            delivered += 1
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning('Enrichment delivery pending for %s: %s', item['source'], exc)
    return delivered


def main() -> None:
    parser = argparse.ArgumentParser(description='Process queued visual sources without blocking search')
    parser.add_argument('--once', action='store_true', help='Process at most one source and deliver cached results')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--vision-url', default=os.environ.get('VECGREP_VISUAL_URL'))
    parser.add_argument('--model', default=os.environ.get('VECGREP_VISUAL_MODEL', 'qwen3.5:2b'))
    parser.add_argument('--api-url', default=os.environ.get('VECGREP_VISUAL_API_URL', 'http://127.0.0.1:8765'))
    parser.add_argument('--language', default=os.environ.get('VECGREP_OCR_LANGUAGE', 'eng'))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    queue = VisualQueue.default()
    if args.status:
        print(json.dumps(queue.status(), indent=2))
        return
    headers = {}
    token = os.environ.get('VECGREP_API_TOKEN')
    if token:
        headers['Authorization'] = f'Bearer {token}'
    with httpx.Client(timeout=httpx.Timeout(300, connect=3), headers=headers) as api:
        processor = VisualProcessor(vision_url=args.vision_url, model=args.model,
                                    scratch=queue.path.parent / 'scratch', language=args.language)
        try:
            while True:
                did_work = process_one(queue, processor=processor)
                deliver(queue, url=args.api_url, client=api)
                if args.once:
                    break
                if not did_work:
                    threading.Event().wait(15)
        finally:
            processor.client.close()


if __name__ == '__main__':
    main()
