"""Run CountGD++ on every prompt combination and save a results table + detection images.

Positive prompt:  text | exemplars | text + exemplars
Negative prompt:  none | text | exemplars | text + exemplars
Exemplars from:   the counted image itself ("internal") or a separate image ("external")
Targets:          strawberry (negative: blueberry) and blueberry (negative: strawberry)

    python test_all_prompts.py [--repo nikigoli/CountGDPlusPlus] [--device cuda] [--out prompt_tests]

Writes <out>/results.csv, <out>/summary.png (all cases on one sheet) and one
image per case. img/336.jpg has about 16 strawberries and well over 100 blueberries.
"""
import argparse
import csv
import itertools
import os

import torch
from PIL import Image, ImageDraw

from countgdpp import CountGDPlusPlus, count

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "img")
IMAGE = os.path.join(IMG, "336.jpg")

# Visual exemplars as [x1, y1, x2, y2] pixel boxes.
EXEMPLARS = {
    "strawberry": {
        "internal": (IMAGE, [[158, 62, 206, 134], [95, 133, 145, 184]]),
        "external": (os.path.join(IMG, "synthetic_strawberry.jpg"), [[205, 95, 822, 915]]),
    },
    "blueberry": {
        "internal": (IMAGE, [[235, 38, 259, 62], [330, 165, 356, 192]]),
        "external": (os.path.join(IMG, "synthetic_blueberry.jpg"), [[143, 140, 878, 850]]),
    },
}
OTHER = {"strawberry": "blueberry", "blueberry": "strawberry"}
POS_MODES = ["text", "exemplars", "text+exemplars"]
NEG_MODES = ["none", "text", "exemplars", "text+exemplars"]
SOURCES = ["internal", "external"]


def build_cases():
    cases, seen = [], set()
    for target, pos, neg, src in itertools.product(OTHER, POS_MODES, NEG_MODES, SOURCES):
        uses_exemplars = "exemplars" in pos or "exemplars" in neg
        key = (target, pos, neg, src if uses_exemplars else "-")
        if key in seen:
            continue
        seen.add(key)
        cases.append(key)
    # Two negatives at once (text + exemplars for the other fruit, text only for "plate").
    for target in OTHER:
        cases.append((target, "text+exemplars", "two negatives", "internal"))
    return cases


def run_case(model, target, pos, neg, src):
    kwargs = {"text": target if "text" in pos else ""}
    if "exemplars" in pos:
        img, boxes = EXEMPLARS[target][src]
        kwargs.update(exemplar_boxes=boxes, exemplar_image=img)

    other = OTHER[target]
    if neg == "two negatives":
        img, boxes = EXEMPLARS[other][src]
        kwargs.update(negative_texts=[other, "plate"],
                      negative_exemplar_boxes=[boxes, None],
                      negative_exemplar_images=[img, None])
    elif neg != "none":
        kwargs["negative_texts"] = [other if "text" in neg else ""]
        if "exemplars" in neg:
            img, boxes = EXEMPLARS[other][src]
            kwargs.update(negative_exemplar_boxes=[boxes], negative_exemplar_images=[img])
    return count(model, IMAGE, **kwargs)


def draw(result, title):
    im = Image.open(IMAGE).convert("RGB")
    w, h = im.size
    d = ImageDraw.Draw(im)
    for cx, cy, bw, bh in result["boxes"]:
        d.rectangle([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h],
                    outline="cyan", width=2)
    canvas = Image.new("RGB", (w, h + 22), "white")
    canvas.paste(im, (0, 22))
    ImageDraw.Draw(canvas).text((4, 5), title, fill="black")
    return canvas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="nikigoli/CountGDPlusPlus")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="prompt_tests")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model = CountGDPlusPlus.from_pretrained(args.repo).to(args.device).eval()

    rows, tiles = [], []
    cases = build_cases()
    print(f"{'#':>3}  {'target':<10} {'positive':<15} {'negative':<15} {'exemplars':<9} {'count':>5}")
    for i, (target, pos, neg, src) in enumerate(cases, 1):
        try:
            res = run_case(model, target, pos, neg, src)
            n, err = res["count"], ""
        except Exception as e:  # keep going so one failure doesn't hide the rest
            res, n, err = None, "", f"{type(e).__name__}: {e}"
        print(f"{i:>3}  {target:<10} {pos:<15} {neg:<15} {src:<9} {n!s:>5}  {err}")
        rows.append({"case": i, "target": target, "positive": pos, "negative": neg,
                     "exemplar_source": src, "count": n, "error": err})
        if res is not None:
            tile = draw(res, f"#{i} {target} | pos: {pos} | neg: {neg} | {src} -> {n}")
            tile.save(os.path.join(args.out, f"{i:02d}_{target}_{pos}_{neg}_{src}.png".replace("+", "-").replace(" ", "-")))
            tiles.append(tile)

    with open(os.path.join(args.out, "results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    if tiles:
        cols = 4
        tw, th = tiles[0].size
        sheet = Image.new("RGB", (cols * tw, -(-len(tiles) // cols) * th), "white")
        for k, t in enumerate(tiles):
            sheet.paste(t, ((k % cols) * tw, (k // cols) * th))
        sheet.save(os.path.join(args.out, "summary.png"))

    failed = [r for r in rows if r["error"]]
    print(f"\n{len(rows) - len(failed)}/{len(rows)} cases ran. Results in {args.out}/")
    if failed:
        raise SystemExit(f"{len(failed)} case(s) failed, see the error column.")


if __name__ == "__main__":
    main()
