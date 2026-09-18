from __future__ import annotations

import math
import random
import struct
import zlib
from pathlib import Path

from src.semantic.schemas import Concept

PALETTE = {
    "red": (237, 59, 59), "yellow": (244, 197, 66), "blue": (67, 137, 232),
    "purple": (147, 75, 209), "green": (69, 166, 90), "brown": (148, 95, 50),
    "orange": (242, 140, 40), "black": (23, 23, 23),
}


class Canvas:
    def __init__(self, width: int, height: int):
        self.width, self.height = width, height
        self.pixels = bytearray([255]) * (width * height * 3)

    def dot(self, x, y, color, radius=1):
        for py in range(y - radius, y + radius + 1):
            for px in range(x - radius, x + radius + 1):
                if 0 <= px < self.width and 0 <= py < self.height:
                    index = (py * self.width + px) * 3
                    self.pixels[index:index + 3] = bytes(color)

    def line(self, points, color, width=5):
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            steps = max(abs(x1 - x0), abs(y1 - y0), 1)
            for step in range(steps + 1):
                ratio = step / steps
                self.dot(round(x0 + (x1 - x0) * ratio),
                         round(y0 + (y1 - y0) * ratio),
                         color, max(1, width // 2))

    def ellipse(self, box, color, width=5, fill=None):
        x0, y0, x1, y1 = box
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        rx, ry = max(1, (x1 - x0) / 2), max(1, (y1 - y0) / 2)
        if fill is not None:
            for y in range(round(y0), round(y1) + 1):
                span = rx * math.sqrt(max(0.0, 1 - ((y - cy) / ry) ** 2))
                self.line([(round(cx - span), y), (round(cx + span), y)], fill, 1)
        for degree in range(361):
            angle = math.radians(degree)
            self.dot(round(cx + rx * math.cos(angle)),
                     round(cy + ry * math.sin(angle)),
                     color, max(1, width // 2))

    def rectangle(self, box, color, width=4, fill=None):
        x0, y0, x1, y1 = box
        if fill is not None:
            for y in range(y0, y1 + 1):
                self.line([(x0, y), (x1, y)], fill, 1)
        self.line([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)], color, width)

    def save_png(self, path: Path):
        scanlines = b"".join(
            b"\x00" + bytes(self.pixels[y * self.width * 3:(y + 1) * self.width * 3])
            for y in range(self.height)
        )
        def chunk(kind, data):
            return (struct.pack(">I", len(data)) + kind + data
                    + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))
        payload = b"\x89PNG\r\n\x1a\n"
        payload += chunk(b"IHDR", struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0))
        payload += chunk(b"IDAT", zlib.compress(scanlines, 9))
        payload += chunk(b"IEND", b"")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def render_placeholder(concept: Concept, output_path: Path, seed: int,
                       width=768, height=768):
    concept.validate()
    rng, canvas = random.Random(seed), Canvas(width, height)
    colors = [PALETTE.get(name, PALETTE["blue"]) for name in concept.accent_colors]
    colors = colors or [PALETTE["blue"]]
    renderers = {
        "symbolic": _symbolic, "scene": _scene, "character": _character,
        "fantasy_hybrid": _fantasy, "object": _object,
    }
    renderers[concept.expression_type](canvas, rng, width, height, colors, PALETTE["black"])
    canvas.save_png(output_path)


def _line(canvas, rng, points, color, width=5):
    points = [(x + rng.randint(-5, 5), y + rng.randint(-5, 5)) for x, y in points]
    canvas.line(points, color, width)


def _symbolic(c, r, w, h, colors, black):
    cx, cy = int(w * .62), int(h * .45)
    c.ellipse((cx - 36, cy - 36, cx + 36, cy + 36), colors[0], 6)
    for degree in range(0, 360, 45):
        angle = math.radians(degree)
        p1 = (cx + int(48 * math.cos(angle)), cy + int(48 * math.sin(angle)))
        p2 = (cx + int(70 * math.cos(angle)), cy + int(70 * math.sin(angle)))
        _line(c, r, [p1, p2], colors[0])
    _line(c, r, [(235, 250), (285, 215), (340, 235), (372, 275), (230, 282)], black, 7)
    for x in (255, 300, 345):
        _line(c, r, [(x, 300), (x - 10, 350)], colors[-1])
    stem = int(w * .48)
    _line(c, r, [(stem, 500), (stem, 620)], colors[1 % len(colors)], 6)
    for dx, dy in ((0, -28), (-28, 0), (28, 0), (0, 28)):
        c.ellipse((stem + dx - 24, 460 + dy, stem + dx + 24, 500 + dy), black, 5)
    c.ellipse((stem - 13, 467, stem + 13, 493), black, 3, colors[0])


def _object(c, r, w, h, colors, black):
    x, y = int(w * .43), int(h * .42)
    _line(c, r, [(x, y), (x + 110, y), (x + 125, y + 210), (x - 8, y + 210), (x, y)], black, 7)
    c.rectangle((x + 14, y + 85, x + 104, y + 160), black, 4, colors[0])
    _line(c, r, [(x + 70, y), (x + 90, y - 75)], black)


def _character(c, r, w, h, colors, black):
    cx, cy = int(w * .52), int(h * .43)
    c.ellipse((cx - 55, cy - 65, cx + 55, cy + 55), black, 7)
    for points, color in [([(cx, cy + 55), (cx - 10, cy + 230)], black),
                          ([(cx, cy + 100), (cx - 95, cy + 175)], colors[0]),
                          ([(cx, cy + 100), (cx + 85, cy + 155)], colors[-1]),
                          ([(cx - 10, cy + 230), (cx - 65, cy + 310)], black),
                          ([(cx - 10, cy + 230), (cx + 55, cy + 300)], black)]:
        _line(c, r, points, color, 7)


def _scene(c, r, w, h, colors, black):
    _character(c, r, w - 180, h, colors, black)
    x, y = int(w * .58), int(h * .43)
    _line(c, r, [(x, y), (x + 130, y), (x + 130, y + 170), (x, y + 170), (x, y)], black, 6)
    _line(c, r, [(x + 65, y + 170), (x + 20, y + 285)], black)
    _line(c, r, [(x + 65, y + 170), (x + 110, y + 285)], black)
    c.ellipse((x + 42, y + 55, x + 85, y + 100), colors[0], 5)


def _fantasy(c, r, w, h, colors, black):
    cx, cy = int(w * .52), int(h * .54)
    c.ellipse((cx - 130, cy - 95, cx + 130, cy + 95), black, 8)
    _line(c, r, [(cx - 80, cy - 85), (cx - 125, cy - 165), (cx - 30, cy - 105)], colors[0], 7)
    _line(c, r, [(cx + 70, cy - 85), (cx + 125, cy - 175), (cx + 115, cy - 50)], colors[-1], 7)
    for x in (cx - 55, cx + 55):
        c.ellipse((x - 10, cy - 20, x + 10, cy), black, 2, black)
    _line(c, r, [(cx - 25, cy + 42), (cx + 25, cy + 35)], black)
