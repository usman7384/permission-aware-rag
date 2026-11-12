from pathlib import Path

OUT = Path(__file__).parent / "pdf" / "price-list.pdf"
HEADING = "Support Price List"
PARAGRAPH = [
    "This price list sets the yearly fee for each support plan.",
    "Every plan includes email support during office hours.",
]
TABLE = [
    ["Plan", "Yearly fee", "Response time"],
    ["Basic", "900 euros", "48 hours"],
    ["Premium", "2400 euros", "4 hours, code QUARTZ-5"],
]
LEFT, TOP, ROW_HEIGHT, COLUMN_WIDTHS = 72, 640, 24, [120, 120, 240]


def _text(font: str, size: int, x: float, y: float, text: str) -> str:
    return f"BT /{font} {size} Tf {x} {y} Td ({text}) Tj ET\n"


def _table() -> str:
    width = sum(COLUMN_WIDTHS)
    bottom = TOP - ROW_HEIGHT * len(TABLE)
    ops = ["0.5 w\n"]
    for row in range(len(TABLE) + 1):
        ops.append(
            f"{LEFT} {TOP - row * ROW_HEIGHT} m {LEFT + width} {TOP - row * ROW_HEIGHT} l S\n"
        )
    x = LEFT
    for column_width in [0, *COLUMN_WIDTHS]:
        x += column_width
        ops.append(f"{x} {TOP} m {x} {bottom} l S\n")
    for r, cells in enumerate(TABLE):
        x = LEFT
        for cell, column_width in zip(cells, COLUMN_WIDTHS, strict=True):
            font = "F2" if r == 0 else "F1"
            ops.append(_text(font, 11, x + 6, TOP - (r + 1) * ROW_HEIGHT + 8, cell))
            x += column_width
    return "".join(ops)


def _content() -> bytes:
    ops = [_text("F2", 20, LEFT, 740, HEADING)]
    ops += [_text("F1", 11, LEFT, 700 - 16 * i, line) for i, line in enumerate(PARAGRAPH)]
    return ("".join(ops) + _table()).encode("latin-1")


def build() -> bytes:
    content = _content()
    font = "<< /Type /Font /Subtype /Type1 /BaseFont /{} /Encoding /WinAnsiEncoding >>"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R"
        b" /Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"endstream",
        font.format("Helvetica").encode(),
        font.format("Helvetica-Bold").encode(),
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return bytes(out)


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_bytes(build())
