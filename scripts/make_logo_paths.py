"""Outline the live Lato text in the Inkscape logo so it renders without the font.

Reads docs/src/assets/logo.svg (black text, transparent background) and writes
docs/src/assets/logo_header.svg (white text, for the dark header) and logo_light.svg (black text,
for the white light-mode header): same artwork, text converted to <path>.

Fonts (Lato, SIL OFL) are passed as arguments; fetch them from
https://github.com/google/fonts/tree/main/ofl/lato.

Usage: uv run --no-project --with fonttools python scripts/make_logo_paths.py \
    --extra-bold Lato-ExtraBold.ttf --black-italic Lato-BlackItalic.ttf
"""

import argparse
import re
from pathlib import Path

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

ASSETS = Path("docs/src/assets")
FONT_SIZE = 85.3333
LETTER_SPACING = 1.39239
TEXT_RE = re.compile(r"<text\b.*?</text>", re.DOTALL)
# (text, x, y, which font) in document order, read off the source SVG.
RUNS = [
    [("Fastkit-", 1924.0664, 426.00306), ("Mobility", 1924.0664, 536.93635)],
    [("F", 1924.0664, 426.00306)],
]


def outline(font: TTFont, text: str, x: float, y: float) -> str:
    glyphs = font.getGlyphSet()
    cmap = font.getBestCmap()
    scale = FONT_SIZE / font["head"].unitsPerEm
    pen = SVGPathPen(glyphs, ntos=lambda v: f"{v:.2f}")
    for ch in text:
        name = cmap[ord(ch)]
        glyphs[name].draw(TransformPen(pen, (scale, 0, 0, -scale, x, y)))
        x += glyphs[name].width * scale + LETTER_SPACING
    return pen.getCommands()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra-bold", required=True)
    ap.add_argument("--black-italic", required=True)
    args = ap.parse_args()
    fonts = [TTFont(args.extra_bold), TTFont(args.black_italic)]

    svg = (ASSETS / "logo.svg").read_text()
    texts = TEXT_RE.findall(svg)
    assert len(texts) == len(RUNS), "logo.svg text layout changed; update RUNS"
    for name, fill in (("logo_header.svg", "#ffffff"), ("logo_light.svg", "#000000")):
        out = svg
        for text, runs, font in zip(texts, RUNS, fonts, strict=True):
            transform = re.search(r'transform="([^"]+)"', text).group(1)
            d = " ".join(outline(font, *run) for run in runs)
            out = out.replace(text, f'<path transform="{transform}" d="{d}" style="fill:{fill};stroke:none"/>')
        (ASSETS / name).write_text(out)


if __name__ == "__main__":
    main()
