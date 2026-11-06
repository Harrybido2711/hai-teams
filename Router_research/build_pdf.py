"""Render Router_research/ROUTER_RESEARCH.md to Results/ROUTER_RESEARCH.pdf, text unchanged. Needs reportlab.

Usage: /opt/miniconda3/bin/python3 Router_research/build_pdf.py
"""
import html, os, re
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import ListFlowable, ListItem, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "ROUTER_RESEARCH.md")
OUT = os.path.join(os.path.dirname(HERE), "Results", "ROUTER_RESEARCH.pdf")
FD = "/System/Library/Fonts/Supplemental/"
for name, f in [("Body", "Arial.ttf"), ("Body-B", "Arial Bold.ttf"), ("Body-I", "Arial Italic.ttf"), ("Body-BI", "Arial Bold Italic.ttf"),
                ("Code", "Courier New.ttf")]:
    pdfmetrics.registerFont(TTFont(name, FD + f))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-B", italic="Body-I", boldItalic="Body-BI")

INK, MUTED, RULE, HEAD_BG, LINK = colors.HexColor("#1F2328"), colors.HexColor("#59636E"), colors.HexColor("#D0D7DE"), colors.HexColor("#F2F4F7"), colors.HexColor("#0B5CAD")
body = ParagraphStyle("body", fontName="Body", fontSize=10, leading=14.5, textColor=INK, spaceAfter=7, alignment=TA_LEFT)
h1 = ParagraphStyle("h1", parent=body, fontName="Body-B", fontSize=20, leading=25, spaceAfter=3)
h2 = ParagraphStyle("h2", parent=body, fontName="Body-B", fontSize=13.5, leading=18, spaceBefore=13, spaceAfter=6, keepWithNext=1)
byline = ParagraphStyle("byline", parent=body, textColor=MUTED, spaceAfter=10)
cell = ParagraphStyle("cell", parent=body, fontSize=8.8, leading=12, spaceAfter=0)
head = ParagraphStyle("head", parent=cell, fontName="Body-B")
pre = ParagraphStyle("pre", fontName="Code", fontSize=7.6, leading=11, textColor=INK, backColor=HEAD_BG, borderPadding=6, spaceBefore=6, spaceAfter=12)


def inline(s):
    codes = []
    def stash(m):
        codes.append(f'<font name="Code" size="-0.3">{html.escape(m.group(1), quote=False)}</font>')
        return f"\0{len(codes) - 1}\0"
    s = re.sub(r"`([^`]+)`", stash, s)
    links = []
    def stash_link(m):
        links.append((m.group(1), m.group(2)))
        return f"\1{len(links) - 1}\1"
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", stash_link, s)
    s = html.escape(s, quote=False)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<![\*\w])\*([^*]+?)\*(?![\*\w])", r"<i>\1</i>", s)
    s = re.sub("\1(\\d+)\1", lambda m: f'<a href="{html.escape(links[int(m.group(1))][1])}" color="#0B5CAD"><u>{html.escape(links[int(m.group(1))][0], quote=False)}</u></a>', s)
    return re.sub("\0(\\d+)\0", lambda m: codes[int(m.group(1))], s)


def cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


WIDTHS = {  # share of text width, by the table's first header cell (Benchmark tables: first two)
    "#": [.04, .17, .29, .36, .14],
    "Router (macro accuracy)": [.72, .28],
    "Router": [.25, .55, .20],
    "Basis (mean over benchmarks)": [.45, .11, .11, .11, .11, .11],
    "Benchmark|Queries": [.26, .40, .34],
    "Benchmark|Best": [.25, .20, .21, .17, .17],
    "Difference, in points": [.42, .10, .18, .30],
    "Model": [.20, .20, .20, .20, .20],
}


def make_table(rows, width):
    key = rows[0][0] if rows[0][0] != "Benchmark" else "Benchmark|" + rows[0][1].split()[0]
    w = [x * width for x in WIDTHS[key]]
    data = [[Paragraph(inline(c), head if i == 0 else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=w, repeatRows=1, spaceBefore=3, spaceAfter=10)
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
                           ("LINEBELOW", (0, 0), (-1, -1), .5, RULE), ("LINEABOVE", (0, 0), (-1, 0), .5, RULE),
                           ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                           ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    return t


def footer(c, d):
    c.saveState(); c.setFont("Body", 8); c.setFillColor(MUTED)
    c.drawString(d.leftMargin, .5 * inch, "LLM Router Research"); c.drawRightString(letter[0] - d.rightMargin, .5 * inch, str(d.page)); c.restoreState()


doc = SimpleDocTemplate(OUT, pagesize=letter, leftMargin=.85 * inch, rightMargin=.85 * inch, topMargin=.8 * inch, bottomMargin=.85 * inch,
                        title="LLM Router Research", author="Harry")
width = doc.width
lines = open(SRC, encoding="utf-8").read().split("\n")
story, i = [], 0
while i < len(lines):
    ln = lines[i]
    if not ln.strip():
        i += 1
    elif ln.startswith("# "):
        story.append(Paragraph(inline(ln[2:]), h1)); i += 1
        while not lines[i].strip():
            i += 1
        story.append(Paragraph(inline(lines[i]), byline)); i += 1
    elif ln.startswith("## "):
        story.append(Paragraph(inline(ln[3:]), h2)); i += 1
    elif ln.startswith("```"):
        j = i + 1
        while not lines[j].startswith("```"):
            j += 1
        story.append(Preformatted("\n".join(lines[i + 1:j]), pre)); i = j + 1
    elif ln.startswith("|"):
        j = i
        while j < len(lines) and lines[j].startswith("|"):
            j += 1
        rows = [cells(x) for k, x in enumerate(lines[i:j]) if k != 1]
        story.append(make_table(rows, width)); i = j
    elif re.match(r"(- |\d+\. )", ln):
        ordered = ln[0].isdigit()
        j, items = i, []
        while j < len(lines) and re.match(r"(- |\d+\. )", lines[j]):
            items.append(re.sub(r"^(- |\d+\. )", "", lines[j])); j += 1
        story.append(ListFlowable([ListItem(Paragraph(inline(x), body), leftIndent=16) for x in items],
                                  bulletType="1" if ordered else "bullet", start=None if ordered else "•", leftIndent=16,
                                  bulletFormat="%s." if ordered else None,
                                  bulletFontName="Body", bulletFontSize=10))
        story.append(Spacer(1, 3)); i = j
    else:
        story.append(Paragraph(inline(ln), body)); i += 1
doc.build(story, onFirstPage=footer, onLaterPages=footer)
