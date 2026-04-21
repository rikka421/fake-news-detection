from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas


def wrap_text(text: str, max_width: float, font_name: str, font_size: int):
    lines = []
    current = ""
    for ch in text:
        candidate = current + ch
        if pdfmetrics.stringWidth(candidate, font_name, font_size) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = ch
    if current:
        lines.append(current)
    return lines


def main():
    project_root = Path(__file__).resolve().parents[1]
    md_path = project_root / "docs" / "项目技术报告_中文版.md"
    pdf_path = project_root / "docs" / "项目技术报告_中文版.pdf"

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    font_name = "STSong-Light"

    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    page_w, page_h = A4
    left = 56
    right = 56
    top = 60
    bottom = 60
    usable_width = page_w - left - right

    y = page_h - top

    def new_page():
        nonlocal y
        c.showPage()
        c.setFont(font_name, 11)
        y = page_h - top

    c.setTitle("假新闻检测项目技术报告（中文版）")
    c.setAuthor("Project Team")
    c.setSubject("Technical Report")

    lines = md_path.read_text(encoding="utf-8").splitlines()

    for raw in lines:
        line = raw.rstrip()
        if not line:
            y -= 10
            if y < bottom:
                new_page()
            continue

        if line.startswith("# "):
            c.setFont(font_name, 16)
            text = line[2:].strip()
            for seg in wrap_text(text, usable_width, font_name, 16):
                c.drawString(left, y, seg)
                y -= 24
                if y < bottom:
                    new_page()
            c.setFont(font_name, 11)
            continue

        if line.startswith("## "):
            c.setFont(font_name, 13)
            text = line[3:].strip()
            for seg in wrap_text(text, usable_width, font_name, 13):
                c.drawString(left, y, seg)
                y -= 19
                if y < bottom:
                    new_page()
            c.setFont(font_name, 11)
            continue

        if line.startswith("- "):
            text = "• " + line[2:].strip()
        else:
            text = line

        c.setFont(font_name, 11)
        for seg in wrap_text(text, usable_width, font_name, 11):
            c.drawString(left, y, seg)
            y -= 16
            if y < bottom:
                new_page()

    c.save()
    print(f"Generated: {pdf_path}")


if __name__ == "__main__":
    main()
