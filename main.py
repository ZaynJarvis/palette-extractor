"""Palette Extractor — drag images, extract real palettes from pixels."""

from __future__ import annotations

import base64
import binascii
import colorsys
from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

STATIC_DIR = Path(__file__).parent / "static"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGES = 16
THUMB_SIZE = 300
INITIAL_CANDIDATES = 16
MAX_OUTPUT_COLORS = 8
MIN_CLUSTER_FRACTION = 0.02
MERGE_DISTANCE = 28.0  # Euclidean RGB threshold for deduping visually-similar clusters

app = FastAPI(title="Palette Extractor", docs_url=None, redoc_url=None)


class PaletteRequest(BaseModel):
    images: list[str] = Field(..., description="Base64-encoded images (raw b64 or data URL)")


def _decode_image(b64: str) -> Image.Image:
    if "," in b64 and b64.lstrip().startswith("data:"):
        b64 = b64.split(",", 1)[1]
    try:
        raw = base64.b64decode(b64, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid base64: {exc}") from exc
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="image exceeds 10MB limit")
    try:
        img = Image.open(BytesIO(raw))
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"cannot decode image: {exc}") from exc
    return img


def _classify(rgb: tuple[int, int, int]) -> str:
    r, g, b = (c / 255.0 for c in rgb)
    _, l, s = colorsys.rgb_to_hls(r, g, b)
    if l < 0.18:
        return "dark"
    if l > 0.88:
        return "light"
    if s < 0.18:
        return "muted"
    if s > 0.55 and 0.3 < l < 0.7:
        return "vibrant"
    if l > 0.65:
        return "soft"
    if l < 0.4:
        return "deep"
    return "accent"


def _rgb_distance(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    return float(np.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))))


def extract_palette(img: Image.Image) -> list[tuple[str, str]]:
    """Pixel-accurate palette: PIL median-cut + k-means refinement, merge + threshold."""
    img = img.convert("RGB")
    img.thumbnail((THUMB_SIZE, THUMB_SIZE))

    quant = img.quantize(colors=INITIAL_CANDIDATES, method=Image.Quantize.MEDIANCUT, kmeans=2)
    palette_bytes = quant.getpalette() or []
    indices = np.array(quant, dtype=np.int32).flatten()
    counts = np.bincount(indices, minlength=INITIAL_CANDIDATES)
    total = int(counts.sum())
    if total == 0:
        return []

    clusters: list[tuple[int, tuple[int, int, int]]] = []
    for i in range(INITIAL_CANDIDATES):
        if counts[i] / total < MIN_CLUSTER_FRACTION:
            continue
        rgb = (palette_bytes[i * 3], palette_bytes[i * 3 + 1], palette_bytes[i * 3 + 2])
        clusters.append((int(counts[i]), rgb))

    clusters.sort(reverse=True, key=lambda x: x[0])

    merged: list[tuple[int, tuple[int, int, int]]] = []
    for cnt, rgb in clusters:
        if any(_rgb_distance(rgb, m_rgb) < MERGE_DISTANCE for _, m_rgb in merged):
            continue
        merged.append((cnt, rgb))
        if len(merged) >= MAX_OUTPUT_COLORS:
            break

    out: list[tuple[str, str]] = []
    used: dict[str, int] = {}
    for idx, (_, rgb) in enumerate(merged):
        label = "dominant" if idx == 0 else _classify(rgb)
        used[label] = used.get(label, 0) + 1
        final = label if used[label] == 1 else f"{label}{used[label]}"
        out.append((final, "#{:02x}{:02x}{:02x}".format(*rgb)))
    return out


def _format_palette_text(palettes: list[list[tuple[str, str]]]) -> str:
    if len(palettes) == 1:
        return "\n".join(f"{t}: {h}" for t, h in palettes[0]) + "\n"
    parts = []
    for i, p in enumerate(palettes, 1):
        body = "\n".join(f"{t}: {h}" for t, h in p) or "(no colors)"
        parts.append(f"# image {i}\n{body}")
    return "\n---\n".join(parts) + "\n"


@app.post("/api", response_class=PlainTextResponse)
async def api(req: PaletteRequest) -> PlainTextResponse:
    if not req.images:
        raise HTTPException(status_code=400, detail="images array is empty")
    if len(req.images) > MAX_IMAGES:
        raise HTTPException(status_code=400, detail=f"max {MAX_IMAGES} images per request")
    palettes = [extract_palette(_decode_image(b64)) for b64 in req.images]
    return PlainTextResponse(_format_palette_text(palettes))


@app.get("/")
async def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@app.get("/health", response_class=PlainTextResponse)
async def health() -> str:
    return "ok"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
