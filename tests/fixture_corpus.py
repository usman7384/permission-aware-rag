from pathlib import Path

CORPUS = Path(__file__).parent / "fixtures" / "corpus"
CODE_WORDS = {
    "specs/api-limits.md": "ZEBRA-7",
    "specs/deploy-process.md": "OTTER-3",
    "specs/auth-design.md": "HERON-9",
    "specs/storage-layout.md": "LYNX-2",
    "hr/leave-policy.md": "MAPLE-4",
    "hr/expenses.md": "CEDAR-8",
    "hr/onboarding.md": "BIRCH-1",
    "hr/remote-work.md": "ASPEN-6",
    "public/company-values.md": "CORAL-5",
    "public/office-hours.md": "PEARL-0",
    "public/glossary.md": "AMBER-2",
    "public/holidays.md": "SLATE-3",
}


def _in(*folders: str) -> set[str]:
    return {path for path in CODE_WORDS if path.split("/")[0] in folders}


# Spec section 12: the documents each fixture API key can read.
READABLE = {
    "dev-ana": _in("specs", "public"),
    "dev-ben": _in("hr", "public") | {"specs/api-limits.md"},
    "dev-cal": set(),
}
