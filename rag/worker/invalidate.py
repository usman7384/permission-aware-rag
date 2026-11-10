from typing import Literal

from pydantic import BaseModel

from rag.common import opensearch_, pg, qdrant_, redis_
from rag.common.keys import CHANGES_GROUP, CHANGES_STREAM
from rag.common.log import log
from rag.common.permissions import document_readers
from rag.worker.index import Deps


class TupleChanged(BaseModel):
    subject: str
    relation: Literal["member", "viewer", "parent"]
    object: str


def changed_node(change: TupleChanged) -> str:
    """The doc or folder whose readers a `viewer` or `parent` tuple change affects."""
    return change.object if change.relation == "viewer" else change.subject


async def restamp(deps: Deps, node: str, event_id: str) -> None:
    """Recompute the readers of every live document at or beneath `node` and overwrite them."""
    async with deps.pg.connection() as conn:
        docs = await pg.select_live_documents_beneath(conn, node)
        for doc in docs:
            principals = await document_readers(conn, doc)
            await qdrant_.set_principals(deps.qdrant, deps.settings, str(doc), principals)
            await opensearch_.set_principals(deps.opensearch, deps.settings, str(doc), principals)
    log("documents_restamped", job_id=event_id, node=node, documents=len(docs))


async def handle_change(deps: Deps, message: redis_.Message) -> None:
    """Act on one change event and acknowledge it; a failed event stays pending for reclaim."""
    event_id, fields = message
    try:
        if fields["type"] == "tuple_changed":
            change = TupleChanged.model_validate(fields)
            if change.relation != "member":
                async with deps.lock:
                    await restamp(deps, changed_node(change), event_id)
    except Exception as exc:
        log("change_failed", job_id=event_id, error=repr(exc))
        return
    await redis_.ack(deps.redis, CHANGES_STREAM, CHANGES_GROUP, event_id)


async def consume_once(deps: Deps, consumer: str) -> int:
    """Handle the new change events that arrive within one blocking read; returns how many."""
    messages = await redis_.read_new(
        deps.redis, CHANGES_STREAM, CHANGES_GROUP, consumer, deps.settings.worker_block_ms
    )
    for message in messages:
        await handle_change(deps, message)
    return len(messages)


async def reclaim_once(deps: Deps, consumer: str) -> int:
    """Retry change events left unacknowledged for `RECLAIM_IDLE_MS`; returns how many."""
    messages = await redis_.claim_idle(
        deps.redis, CHANGES_STREAM, CHANGES_GROUP, consumer, deps.settings.reclaim_idle_ms
    )
    for message in messages:
        await handle_change(deps, message)
    return len(messages)
