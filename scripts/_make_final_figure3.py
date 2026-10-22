from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "新论文图片"
PHOTO_PATH = ROOT / "图17.png"
ORIGINAL_FIGURE3_PATH = ROOT / "论文图片" / "Figure3.png"
STRUCTURE_SOURCE_PATH = Path(r"C:\Users\62587\Desktop\开题图片\图16.png")


def font(size: int, bold: bool = False):
    path = "C:\\Windows\\Fonts\\arialbd.ttf" if bold else "C:\\Windows\\Fonts\\arial.ttf"
    return ImageFont.truetype(path, size)


def label(draw, rect, text, target, side="right"):
    x0, y0, x1, y1 = rect
    draw.rounded_rectangle(rect, radius=12, fill=(18, 48, 78), outline=(255, 255, 255), width=2)
    box = draw.multiline_textbbox((0, 0), text, font=font(25, True), spacing=3, align="center")
    tw, th = box[2] - box[0], box[3] - box[1]
    draw.multiline_text(((x0+x1-tw)//2, (y0+y1-th)//2-2), text, fill="white", font=font(25, True), spacing=3, align="center")
    # The original photograph already contains one cyan callout per physical module.
    # Do not add a second connector line.


def polygon_block(draw, x, y, w, h, d, top, front, side, title):
    draw.polygon([(x, y), (x+d, y-d), (x+w+d, y-d), (x+w, y)], fill=top, outline=(66,80,93))
    draw.polygon([(x, y), (x+w, y), (x+w, y+h), (x, y+h)], fill=front, outline=(66,80,93))
    draw.polygon([(x+w, y), (x+w+d, y-d), (x+w+d, y+h-d), (x+w, y+h)], fill=side, outline=(66,80,93))
    bbox = draw.textbbox((0,0), title, font=font(22, True))
    draw.text((x+(w-(bbox[2]-bbox[0]))//2, y+h//2-(bbox[3]-bbox[1])//2), title, fill=(35,44,53), font=font(22, True))


def main():
    photo = Image.open(PHOTO_PATH).convert("RGB")
    photo = ImageEnhance.Contrast(photo).enhance(1.05)
    photo = ImageEnhance.Color(photo).enhance(0.96)
    photo = photo.filter(ImageFilter.UnsharpMask(radius=1.0, percent=105, threshold=4))
    d = ImageDraw.Draw(photo)
    # Overlay labels cover legacy red Chinese text; arrows point to the same real modules.
    label(d, (0, 210, 340, 310), "NV diamond +\nMW antenna", (533, 414), side="right")
    label(d, (786, 28, 1042, 116), "Magnetic-field\nstage", (675, 68), side="left")
    label(d, (780, 210, 1057, 320), "Temperature\ncontrol", (682, 314), side="left")
    label(d, (804, 520, 1042, 614), "Laser excitation\n/ collection", (684, 445), side="left")
    # The experimental photograph is dark in this corner.  White lettering
    # with a restrained dark outline remains readable without covering data.
    d.text((16, 12), "(a)", font=font(32, True), fill="white", stroke_width=2, stroke_fill=(12, 27, 43))
    d.text((70, 14), "Experimental sensing interface", font=font(25, True), fill="white", stroke_width=2, stroke_fill=(12, 27, 43))

    schematic = Image.open(STRUCTURE_SOURCE_PATH).convert("RGB")
    d = ImageDraw.Draw(schematic)
    # English labels replace only the legacy Chinese text, while retaining the original 3D hardware rendering and callout arrows.
    patches = [
        ((0, 95, 126, 176), "PT100\nRTD sensor", (8, 106)),
        ((751, 58, 938, 115), "MCH ceramic\nheater", (756, 68)),
        ((753, 128, 938, 180), "Copper spacer", (758, 141)),
        ((748, 272, 938, 323), "NV ensemble\ndiamond", (756, 281)),
        ((0, 244, 152, 400), "Microwave PCB\nwith SMA\nconnectors", (10, 263)),
        ((694, 521, 938, 578), "Laser excitation\nand collection\nmodule", (702, 531)),
    ]
    for rect, text, pos in patches:
        d.rectangle(rect, fill="white")
        d.multiline_text(pos, text, font=font(18), fill=(38, 55, 70), spacing=2)
    d.text((10, 10), "(b)", font=font(26, True), fill="black")
    d.text((55, 13), "Exploded sensing-stack schematic", font=font(20, True), fill="black")
    schematic = schematic.resize((photo.width, round(schematic.height * photo.width / schematic.width)), Image.Resampling.LANCZOS)
    schematic = ImageEnhance.Contrast(schematic).enhance(1.03)
    schematic = schematic.filter(ImageFilter.UnsharpMask(radius=0.8, percent=75, threshold=5))
    # Combine
    canvas = Image.new("RGB", (photo.width, photo.height + 26 + schematic.height), "white")
    canvas.paste(photo, (0, 0))
    canvas.paste(schematic, (0, photo.height + 26))
    canvas.save(OUT / "Figure3.png", dpi=(600,600), compress_level=6)
    canvas.save(OUT / "Figure3.pdf", "PDF", resolution=600.0)


if __name__ == "__main__":
    main()
