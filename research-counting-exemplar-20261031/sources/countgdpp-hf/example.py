"""Count strawberries (not blueberries) in img/336.jpg and save the detections.

    python example.py [--repo nikigoli/CountGDPlusPlus] [--device cuda]
"""
import argparse
import os

import torch
from PIL import Image, ImageDraw

from countgdpp import CountGDPlusPlus, count

HERE = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--repo", default="nikigoli/CountGDPlusPlus")
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

model = CountGDPlusPlus.from_pretrained(args.repo).to(args.device).eval()

image_path = os.path.join(HERE, "img", "336.jpg")
result = count(model, image_path, "strawberry", negative_texts=["blueberry"])
print("Predicted count:", result["count"])

image = Image.open(image_path).convert("RGB")
w, h = image.size
draw = ImageDraw.Draw(image)
for cx, cy, bw, bh in result["boxes"]:
    draw.rectangle([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h],
                   outline="cyan", width=3)
image.save("detections.png")
print("Saved detections.png")
