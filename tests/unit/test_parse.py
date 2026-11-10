from rag.worker.parse import Section, paragraphs, parse

DOC = """Intro line before any heading.

# Title

First paragraph.

## Code

```python
# not a heading

print("still code")
```

## Table

| key | value |
| --- | ----- |
| a   | 1     |

Closing text.
"""


def test_headings_split_sections() -> None:
    sections = parse("doc.md", DOC.encode())
    assert [s.heading for s in sections] == [None, "Title", "Code", "Table"]
    assert "".join(s.text for s in sections) == DOC


def test_fenced_code_block_stays_whole() -> None:
    code = parse("doc.md", DOC.encode())[2]
    assert '```python\n# not a heading\n\nprint("still code")\n```\n' in code.text
    assert [p for p in paragraphs(code.text) if "```" in p] == [
        '```python\n# not a heading\n\nprint("still code")\n```\n\n'
    ]


def test_table_stays_whole() -> None:
    table = parse("doc.md", DOC.encode())[3]
    rows = "| key | value |\n| --- | ----- |\n| a   | 1     |\n"
    assert rows in table.text
    assert rows + "\n" in paragraphs(table.text)


def test_blank_preamble_is_dropped() -> None:
    assert parse("doc.md", b"\n\n# Only\nbody\n") == [Section("Only", "# Only\nbody\n")]


def test_text_file_is_one_section() -> None:
    assert parse("notes.txt", b"# not markdown\nplain") == [Section(None, "# not markdown\nplain")]
