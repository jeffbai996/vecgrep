"""Order handling for a small shop."""
import asyncio
import json
from dataclasses import dataclass
from typing import Iterable

RETRY_LIMIT = 3


@dataclass
class Order:
    id: str
    total: float

    def describe(self) -> str:
        return f"{self.id}: {self.total:.2f}"


class Importer:
    """Loads orders from JSON lines."""

    source = "orders.jsonl"

    def run(self, lines: Iterable[str]) -> list[Order]:
        return [self._parse(line) for line in lines]

    def _parse(self, line: str) -> Order:
        data = json.loads(line)
        return Order(data["id"], data["total"])

    class Stats:
        def run(self, orders: list[Order]) -> float:
            return sum(o.total for o in orders)


class Exporter:
    # Writes orders back out.
    def run(self, orders: list[Order]) -> str:
        return "\n".join(json.dumps(o.__dict__) for o in orders)

    @staticmethod
    def header() -> str:
        return "id,total"

    @property
    def name(self) -> str:
        return "exporter"


async def sync_orders(importer: Importer, lines: list[str]) -> int:
    async def push(order: Order) -> None:
        await asyncio.sleep(0)

    def retryable(error: Exception) -> bool:
        return not isinstance(error, ValueError)

    orders = importer.run(lines)
    for order in orders:
        await push(order)
    return len(orders)


def main() -> None:
    print(asyncio.run(sync_orders(Importer(), [])))
