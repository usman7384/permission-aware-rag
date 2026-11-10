from dataclasses import dataclass

Tuple = tuple[str, str, str]


@dataclass(frozen=True)
class Acl:
    folders: list[str]
    viewer_tuples: list[Tuple]
    member_tuples: list[Tuple]
    selectivity: float


def folder_group(folder: str) -> str:
    return f"group:{folder.replace('/', '-')}"


def reader_group(user: str) -> str:
    return f"group:{user.removeprefix('user:')}-readers"


def readable_docs_count(doc_count: int, folder_count: int, readable_folders: int) -> int:
    """Documents in the first `readable_folders` folders when documents are dealt round-robin."""
    base, extra = divmod(doc_count, folder_count)
    return readable_folders * base + min(readable_folders, extra)


def readable_folders_for(doc_count: int, folder_count: int, selectivity: float) -> int:
    """The number of readable folders (at least one) whose share of documents is nearest."""
    if not 0 < selectivity <= 1:
        raise ValueError(f"selectivity must be in (0, 1], got {selectivity}")
    if not 1 <= folder_count <= doc_count:
        raise ValueError(f"folder_count must be between 1 and {doc_count}, got {folder_count}")

    def error(readable: int) -> float:
        return abs(readable_docs_count(doc_count, folder_count, readable) / doc_count - selectivity)

    return min(range(1, folder_count + 1), key=error)


def synthetic_acl(
    dataset: str, doc_count: int, folder_count: int, selectivity: float, user: str
) -> Acl:
    """Folders for each document, one viewer group per folder, and memberships that let
    `user` read the share of documents nearest to `selectivity`."""
    readable = readable_folders_for(doc_count, folder_count, selectivity)
    names = [f"{dataset}/f{j:04d}" for j in range(folder_count)]
    group = reader_group(user)
    joins = [(group, "member", folder_group(f)) for f in names[:readable]]
    return Acl(
        folders=[names[i % folder_count] for i in range(doc_count)],
        viewer_tuples=[(folder_group(f), "viewer", f"folder:{f}") for f in names],
        member_tuples=[(user, "member", group), *joins],
        selectivity=readable_docs_count(doc_count, folder_count, readable) / doc_count,
    )
