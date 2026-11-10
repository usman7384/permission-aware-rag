import shutil
from pathlib import Path

from conftest import search_paths
from test_ingest import fetch, ingest

from rag.api import pipeline
from rag.common.ids import doc_id
from rag.worker.index import Deps

FIXTURE_PDF = Path(__file__).parents[1] / "fixtures" / "pdf" / "price-list.pdf"
PDF_PATH = "public/price-list.pdf"


async def test_pdf_with_heading_and_table_is_ingested_and_table_text_retrievable(
    ingested: Deps, api_deps: pipeline.Deps
) -> None:
    shutil.copy(FIXTURE_PDF, Path(ingested.settings.corpus_dir) / PDF_PATH)
    await ingest(ingested)
    rows = await fetch(
        ingested, "SELECT heading, text FROM parents WHERE doc_id = %s", str(doc_id(PDF_PATH))
    )
    assert rows
    assert rows[0][0] == "Support Price List"
    text = "".join(parent_text for _, parent_text in rows)
    assert "Premium" in text and "2400 euros" in text
    assert PDF_PATH in await search_paths(api_deps, "dev-ana", "QUARTZ-5")
    assert PDF_PATH not in await search_paths(api_deps, "dev-cal", "QUARTZ-5")
