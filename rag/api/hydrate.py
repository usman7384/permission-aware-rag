from dataclasses import dataclass

from psycopg import AsyncConnection

from rag.api.retrieve import Candidate
from rag.common import pg


@dataclass(frozen=True)
class ParentHit:
    parent_id: str
    doc_id: str
    version: int
    path: str
    heading: str | None
    text: str
    score: float


async def hydrate(conn: AsyncConnection, candidates: list[Candidate]) -> list[ParentHit]:
    """Parents of the candidates, each once with its best child score, best first."""
    best: dict[str, float] = {}
    for c in candidates:
        best[c.parent_id] = max(best.get(c.parent_id, c.score), c.score)
    rows = {row[0]: row for row in await pg.get_parents(conn, list(best))} if best else {}
    hits = [
        ParentHit(*rows[parent], score=score) for parent, score in best.items() if parent in rows
    ]
    return sorted(hits, key=lambda hit: -hit.score)
