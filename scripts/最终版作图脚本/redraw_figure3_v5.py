"""Figure 3 rebuild, round 5 (final polish of panel (a); (b) unchanged from v4).

Root cause of the remaining "old traces": v4 erased only pixels brighter than
195 on all channels, so the semi-transparent anti-aliased fringes of the baked
legacy leaders survived as faint ghosts.  v5 erases every LOCALLY LIFTED pixel
inside the leader corridors instead (luminance above the local median), so
ghosts disappear together with their fringes, and refills them from the local
neighborhood median.

New leader style: a slim white line over a slightly wider dark halo (readable
on every part of the photo), with a slim layered arrowhead.  All four tips
touch the exact vertical centre of their box edge.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PHOTO = "/tmp/fig3_full.png"
SCHEM = "/tmp/fig3ppt/ppt/media/image1.png"
SLIDE_W = 12192000
EMU = 1057 / SLIDE_W
FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
F = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
NAVY, INK, WHITE = (18, 48, 78), (38, 55, 70), (245, 248, 246)
DARK = (24, 34, 46)
GAP, TOP = 26, 656

photo = Image.open(PHOTO).convert("RGB").crop((0, 0, 1057, 630))
sc = Image.open(SCHEM).convert("RGB")
sc_h = round(sc.height * 1057 / sc.width)
canvas = Image.new("RGB", (1057, 630 + GAP + sc_h), "white")
canvas.paste(photo, (0, 0)); canvas.paste(sc.resize((1057, sc_h), Image.LANCZOS), (0, TOP))
d = ImageDraw.Draw(canvas)

def font(sz, bold=False):
    return ImageFont.truetype(FB if bold else F, sz)

def head(tip, away_from, fill, length, spread):
    u = np.array(tip, float) - np.array(away_from, float)
    u /= np.hypot(*u)
    tip = np.array(tip, float); base = tip - u * length
    n = np.array([-u[1], u[0]])
    d.polygon([tuple(tip), tuple(base + n * length * spread),
               tuple(base - n * length * spread)], fill=fill)

# ============================================================ panel (a)
# ---- 1) erase legacy leaders INCLUDING their anti-aliased fringes
corridors = [(330, 560, 230, 432), (650, 806, 25, 100),
             (545, 806, 285, 355), (645, 810, 430, 608)]
arr = np.array(canvas).astype(int)
for x0, x1, y0, y1 in corridors:
    z = arr[y0:y1, x0:x1]
    lum = z.mean(axis=2)
    # local median luminance over a padded window (ignores lifted pixels)
    med = np.empty_like(lum)
    for y in range(z.shape[0]):
        for x in range(z.shape[1]):
            med[y, x] = np.median(lum[max(0, y - 4):y + 5, max(0, x - 4):x + 5])
    lift = lum > med + 22
    yy, xx = np.where(lift)
    for y, x in zip(yy, xx):
        ring = z[max(0, y - 6):y + 7, max(0, x - 6):x + 7].reshape(-1, 3)
        rl = ring.mean(axis=1); keep = ring[rl <= med[y, x] - 0.001]
        if len(keep) >= 3:
            z[y, x] = np.median(keep, axis=0)
canvas = Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))
d = ImageDraw.Draw(canvas)

# ---- 2) uniform boxes
BOXA = [((0, 220, 345, 320), "NV diamond +\nMW antenna"),
        ((782, 30, 1057, 130), "Magnetic-field\nstage"),
        ((782, 220, 1057, 320), "Temperature\ncontrol"),
        ((782, 516, 1057, 616), "Laser excitation\n/ collection")]
for rect, text in BOXA:
    d.rounded_rectangle(rect, radius=12, fill=NAVY, outline=(255, 255, 255), width=2)
    ft = font(25, True)
    bb = d.multiline_textbbox((0, 0), text, font=ft, spacing=3, align="center")
    d.multiline_text((rect[0] + (rect[2] - rect[0] - (bb[2] - bb[0])) / 2,
                      rect[1] + (rect[3] - rect[1] - (bb[3] - bb[1])) / 2 - 2),
                     text, fill="white", font=ft, spacing=3, align="center")

# ---- 3) halo + slim white leaders, tip on the box edge centre
LEADS = [((533, 414), (345, 270)),
         ((688, 64), (782, 80)),
         ((676, 336), (782, 270)),
         ((680, 478), (782, 566))]
for target, tip in LEADS:
    d.line([target, tip], fill=DARK, width=6)          # halo
    d.line([target, tip], fill=WHITE, width=3)         # white core
    head(tip, target, DARK, 22, 0.52)                  # dark halo head
    head(tip, target, WHITE, 16, 0.40)                 # white core head

# ============================================================ panel (b) (unchanged, v4)
d.multiline_text((10, TOP + 8), "(b) Exploded sensing-stack schematic",
                 font=font(20, True), fill=(0, 0, 0))
LABELS = [("MCH ceramic\nheater", 9591332, 532852, "right"),
          ("Copper spacer", 9591332, 1331611, "right"),
          ("NV ensemble\ndiamond", 9591332, 2598122, "right"),
          ("Laser excitation and\ncollection module", 9591332, 4933459, "right"),
          ("PT100 RTD\nsensor", 138147, 1141002, "left"),
          ("Microwave PCB\nwith SMA\nconnectors", 223666, 3105834, "left")]
COL_R, COL_L = 831, 168
ARROWS = [(7453347, 693858, "right"), (7729640, 1470111, "right"),
          (6742878, 2769514, "right"), (8315119, 5071959, "right"),
          (5315361, 1279502, "left"), (3999678, 3375550, "left")]
for x_stack, y_emu, side in ARROWS:
    y = TOP + y_emu * EMU
    if side == "right":
        d.line([(x_stack * EMU + 6, y), (COL_R - 10, y)], fill=INK, width=3)
        head((COL_R - 2, y), (COL_R - 16, y), INK, 13, 0.58)
    else:
        d.line([(COL_L + 10, y), (x_stack * EMU - 6, y)], fill=INK, width=3)
        head((COL_L + 2, y), (COL_L + 16, y), INK, 13, 0.58)
for text, bx, by, side in LABELS:
    ft = font(20)
    bb = d.multiline_textbbox((0, 0), text, font=ft, spacing=2)
    y = TOP + by * EMU + (bb[3] - bb[1]) / 2 - 12
    x = COL_R if side == "right" else COL_L - (bb[2] - bb[0])
    d.multiline_text((x, y), text, font=ft, fill=INK, spacing=2,
                     align="left" if side == "right" else "right")

# ============================================================ checks
out = np.array(canvas).astype(int)
r, g, b = out[:, :, 0], out[:, :, 1], out[:, :, 2]
navy = (r < 75) & (g < 95) & (b > 55) & (b < 165) & (b > r + 25)
wl = (r - WHITE[0] > -55) & (r > 195) & (g > 195) & (b > 185)
# 1. no locally lifted white remains inside the corridors (ghost check)
ghost = 0
for x0, x1, y0, y1 in corridors:
    z = out[y0:y1, x0:x1]; lum = z.mean(axis=2)
    for y in range(0, z.shape[0], 3):
        for x in range(0, z.shape[1], 3):
            med = np.median(lum[max(0, y - 4):y + 5, x:x + 5])
            if lum[y, x] > med + 40:
                ghost += 1
print("check-ghosts:", ghost)
assert ghost == 0
# 2. leaders: length, visibility of white core, tip on box edge
for target, tip in LEADS:
    L = int(round(np.hypot(tip[0] - target[0], tip[1] - target[1])))
    hit = sum(int(wl[int(target[1] + (tip[1] - target[1]) * t) - 1:
                     int(target[1] + (tip[1] - target[1]) * t) + 2,
                     int(target[0] + (tip[0] - target[0]) * t) - 1:
                     int(target[0] + (tip[0] - target[0]) * t) + 2].sum() > 0)
              for t in np.linspace(0.2, 0.95, 28))
    onedge = navy[tip[1] - 5:tip[1] + 6, tip[0] - 5:tip[0] + 6].any()
    print(f"leader {target}->{tip}: {L}px core-visible {hit}/28 tip-on-box {onedge}")
    assert L >= 90 and hit >= 25 and onedge
# 3. box uniformity
assert {q[0][3] - q[0][1] for q in BOXA} == {100}
assert {q[0][0] for q in BOXA[1:]} == {782}
print("boxes: 4 x 100 tall, right column at x=782 — OK")

canvas.save("/home/tianyuchu/DSH工作/NV_paper/Figure3.png", dpi=(600, 600))
canvas.save("/home/tianyuchu/DSH工作/NV_paper/Figure3.pdf", "PDF", resolution=600.0)
canvas.save("/mnt/c/Users/tianyu.chu/Desktop/论文/NV论文/Figure3.png", dpi=(600, 600))
canvas.save("/mnt/c/Users/tianyu.chu/Desktop/论文/NV论文/Figure3.pdf", "PDF", resolution=600.0)
print("Figure3 v5 saved (both dirs)")
