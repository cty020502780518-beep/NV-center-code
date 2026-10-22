"""Final layout repair for the flattened Figure 1 composite."""

from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "新论文图片"
NAVY = (20, 35, 52)
TEXT = (30, 38, 48)
MUTED = (75, 85, 96)


def arial(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    filename = "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / filename), size)


def centred(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, size: int, *, bold: bool = False, fill=NAVY) -> None:
    draw.text((x, y), text, anchor="ma", font=arial(size, bold), fill=fill)


def marker(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, *, outline, fill=None, size: int = 42, text_x: Optional[int] = None) -> None:
    radius = 15
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline, width=5)
    draw.text((text_x if text_x is not None else x + 30, y), text, anchor="lm", font=arial(size), fill=TEXT)


def line_key(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, *, colour, dashed: bool = False, size: int = 42, text_x: Optional[int] = None) -> None:
    if dashed:
        for start in range(x, x + 68, 23):
            draw.line((start, y, min(start + 14, x + 68), y), fill=colour, width=6)
    else:
        draw.line((x, y, x + 68, y), fill=colour, width=7)
    draw.text((text_x if text_x is not None else x + 88, y), text, anchor="lm", font=arial(size), fill=TEXT)


def dashed_line(draw: ImageDraw.ImageDraw, start, end, *, fill, width: int, dash: int, gap: int) -> None:
    x0, y0 = start
    x1, y1 = end
    length = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    distance = 0.0
    while distance < length:
        segment_end = min(distance + dash, length)
        xa = x0 + (x1 - x0) * distance / length
        ya = y0 + (y1 - y0) * distance / length
        xb = x0 + (x1 - x0) * segment_end / length
        yb = y0 + (y1 - y0) * segment_end / length
        draw.line((round(xa), round(ya), round(xb), round(yb)), fill=fill, width=width)
        distance += dash + gap


def original_figure() -> Path:
    for folder in ROOT.iterdir():
        candidate = folder / "Figure1.png"
        if folder.is_dir() and candidate.exists() and Image.open(candidate).size == (6799, 2231):
            return candidate
    raise FileNotFoundError("Untouched Figure 1 source was not found")


def main() -> None:
    source = Image.open(original_figure()).convert("RGB")
    source_plot_top = 140
    header_height = 240
    body = source.crop((0, source_plot_top, source.width, source.height))
    body_draw = ImageDraw.Draw(body)

    # Remove legacy display layers. Every replacement is placed in a known
    # data-free right-side region or the new compact header.
    body_draw.rectangle((0, 0, source.width, 50), fill="white")
    body_draw.rectangle((1000, 50, 2266, 340), fill="white")
    body_draw.rectangle((5310, 50, 6799, 365), fill="white")
    body_draw.rectangle((400, 48, 613, 300), fill="white")
    body_draw.rectangle((4870, 48, 5082, 400), fill="white")

    # Restore a uniform plotting background to the left of the 40-degree grid
    # line. Original data from 40 degrees onward remain untouched.
    dashed_line(body_draw, (400, 149), (610, 149), fill=(230, 230, 230), width=5, dash=18, gap=11)
    dashed_line(body_draw, (613, 48), (613, 300), fill=(230, 230, 230), width=5, dash=18, gap=11)
    body_draw.line((470, 135, 542, 209), fill=(219, 45, 45), width=18)
    body.paste(source.crop((540, 280, 700, 470)), (540, 140))

    dashed_line(body_draw, (4870, 164), (5082, 164), fill=(225, 225, 225), width=5, dash=8, gap=10)
    dashed_line(body_draw, (5083, 48), (5083, 400), fill=(225, 225, 225), width=5, dash=8, gap=10)
    body.paste(source.crop((5020, 220, 5230, 560)), (5020, 80))

    # Remove only the old '(b)' label; restore both the bar top and full 0.00.
    body_draw.rectangle((2580, 20, 2695, 170), fill="white")
    body_draw.rectangle((2696, 50, 2821, 160), fill=(71, 42, 121))
    body_draw.rectangle((2696, 160, 2821, 180), fill=(71, 42, 121))
    body.paste(source.crop((2400, 150, 2580, 225)), (2400, 10))

    # Large, unambiguous x labels: top line is laser power, bottom is MW power.
    axis_y = 1887 - source_plot_top
    # Clear only below the original x-axis. Preserve the y-axis tick labels,
    # then redraw the full baseline and all nine ticks above the new labels.
    body_draw.rectangle((2600, axis_y + 8, 4533, body.height), fill="white")
    body_draw.rectangle((2560, axis_y + 25, 2613, body.height), fill="white")
    body_draw.rectangle((2560, axis_y - 24, 2613, axis_y - 2), fill="white")
    states = (
        "20%\n-5 dBm", "20%\n0 dBm", "20%\n5 dBm",
        "50%\n-5 dBm", "50%\n0 dBm", "50%\n5 dBm",
        "100%\n-5 dBm", "100%\n0 dBm", "100%\n5 dBm",
    )
    centres = (2758, 2932, 3106, 3280, 3454, 3628, 3802, 3976, 4150)
    body_draw.line((2617, 50, 2617, axis_y), fill=(0, 0, 0), width=7)
    body_draw.line((2592, 50, 2617, 50), fill=(0, 0, 0), width=5)
    body_draw.line((2588, axis_y, 4291, axis_y), fill=(0, 0, 0), width=7)
    for x in centres:
        body_draw.line((x, axis_y, x, axis_y + 20), fill=(0, 0, 0), width=5)
    for x, label in zip(centres, states):
        body_draw.multiline_text((x, axis_y + 48), label, anchor="ma", font=arial(46, True),
                                 fill=NAVY, spacing=0, align="center")
    centred(body_draw, 3400, axis_y + 205, "Laser power / MW power", 34, bold=True, fill=MUTED)

    canvas = Image.new("RGB", (source.width, header_height + body.height), "white")
    canvas.paste(body, (0, header_height))
    draw = ImageDraw.Draw(canvas)
    panels = ((0, 2266), (2266, 4533), (4533, 6799))
    titles = (
        "Representative physical calibration",
        "Power-dependent slopes across nine states",
        "Physics-constrained Sim2Real augmentation",
    )
    for index, ((left, right), title) in enumerate(zip(panels, titles)):
        draw.text((left + 48, 38), f"({chr(97 + index)})", anchor="la", font=arial(56, True), fill=NAVY)
        centred(draw, (left + right) // 2 + 55, 40, title, 62, bold=True)

    # Panel A: legend and power group in the unused upper-right plot region.
    legend_y_a = header_height + 102
    legend_text_x_a = 1190
    draw.text((legend_text_x_a, legend_y_a), "Power group: laser 50%, MW 0 dBm", anchor="la", font=arial(42, True), fill=MUTED)
    marker(draw, 1135, legend_y_a + 76, "Measured seed data (n=31)", outline=(43, 127, 184), text_x=legend_text_x_a)
    line_key(draw, 1080, legend_y_a + 152, "Linear fit (slope=-0.0914 MHz/deg C)", colour=(219, 45, 45), text_x=legend_text_x_a)

    # Panel B: the explanatory lines share a common left text edge.
    b_text_x = 2905
    draw.text((b_text_x, 108), "Nine laser/MW power combinations", anchor="la", font=arial(38, True), fill=MUTED)
    line_key(draw, 2800, 178, "Mean slope: -0.0903 MHz/deg C", colour=(73, 73, 73), dashed=True, size=38, text_x=b_text_x)

    # Panel C: the key sits in its own unused upper-right plot region.
    legend_y_c = legend_y_a
    legend_text_x_c = 4533 + 1190
    draw.text((legend_text_x_c, legend_y_c), "Power group: laser 50%, MW 0 dBm", anchor="la", font=arial(42, True), fill=MUTED)
    marker(draw, 4533 + 1135, legend_y_c + 76, "Physics-informed virtual cloud", outline=(111, 181, 232), fill=(190, 222, 247), text_x=legend_text_x_c)
    marker(draw, 4533 + 1135, legend_y_c + 152, "Real seed data (n=31)", outline=(0, 0, 0), fill=(225, 48, 45), text_x=legend_text_x_c)
    line_key(draw, 4533 + 1080, legend_y_c + 228, "Calibrated physical law", colour=(0, 0, 0), dashed=True, text_x=legend_text_x_c)

    canvas.save(OUT / "Figure1.png", dpi=(600, 600), compress_level=6)
    canvas.save(OUT / "Figure1.pdf", "PDF", resolution=600.0)


if __name__ == "__main__":
    main()
