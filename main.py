"""Palette Extractor — drag images, extract real palettes from pixels."""

from __future__ import annotations

import base64
import binascii
import colorsys
from io import BytesIO
from pathlib import Path
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
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


# ---------------------------------------------------------------------------
# Profile Extractor — color-difference grid detection for avatar cropping
# ---------------------------------------------------------------------------

PROFILE_OUTPUT_SIZE = 24
MIN_CELL_SIZE = 20
MIN_AVATAR_DIM = 30
BORDER_TOLERANCE = 25
SEPARATOR_THRESHOLD = 0.5
MAJOR_SEP_THRESHOLD = 0.85
MAJOR_SEP_MIN_WIDTH = 8


def _detect_border_color(arr: np.ndarray, sample_width: int = 10) -> np.ndarray:
    """Detect dominant border color by sampling image edges."""
    h, w = arr.shape[:2]
    sw = min(sample_width, h // 4, w // 4)
    edges = np.concatenate([
        arr[:sw, :, :3].reshape(-1, 3),
        arr[-sw:, :, :3].reshape(-1, 3),
        arr[:, :sw, :3].reshape(-1, 3),
        arr[:, -sw:, :3].reshape(-1, 3),
    ])
    return np.median(edges, axis=0)


def _border_fraction(arr: np.ndarray, bg: np.ndarray, tol: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-row and per-col fraction of pixels matching border color."""
    diff = np.abs(arr[:, :, :3].astype(float) - bg).max(axis=2)
    mask = diff < tol
    return mask.mean(axis=1), mask.mean(axis=0)


def _group_consecutive(indices: np.ndarray, min_gap: int = 3) -> list[tuple[int, int]]:
    if len(indices) == 0:
        return []
    groups: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx - prev > min_gap:
            groups.append((start, prev))
            start = idx
        prev = idx
    groups.append((start, prev))
    return groups


def _find_separator_bands(
    frac: np.ndarray, threshold: float, min_width: int = 2,
) -> list[tuple[int, int]]:
    indices = np.where(frac > threshold)[0]
    groups = _group_consecutive(indices)
    return [(s, e) for s, e in groups if e - s + 1 >= min_width]


def _bands_to_cells(
    bands: list[tuple[int, int]], total: int, min_size: int = MIN_CELL_SIZE,
) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    prev = 0
    for s, e in bands:
        if s - prev >= min_size:
            cells.append((prev, s))
        prev = e + 1
    if total - prev >= min_size:
        cells.append((prev, total))
    return cells


def _trim_padding(arr: np.ndarray, bg: np.ndarray, tol: int) -> np.ndarray:
    diff = np.abs(arr[:, :, :3].astype(float) - bg).max(axis=2)
    content = diff >= tol
    rows = np.where(content.mean(axis=1) > 0.08)[0]
    cols = np.where(content.mean(axis=0) > 0.08)[0]
    if len(rows) == 0 or len(cols) == 0:
        return arr
    return arr[int(rows[0]):int(rows[-1]) + 1, int(cols[0]):int(cols[-1]) + 1]


def _image_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def extract_profiles(img: Image.Image, *, force: bool = False) -> list[str]:
    """Detect avatar grid and extract each avatar as 24x24 base64 PNG.

    When force=True, skip grid detection and always attempt extraction
    using any separator bands found (even weak ones).
    Returns empty list if no cells are found.
    """
    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]
    if h < 40 or w < 40:
        return []

    bg = _detect_border_color(arr)
    row_frac, col_frac = _border_fraction(arr, bg, BORDER_TOLERANCE)

    # Find major panel separators (wide bands)
    major_rows = _find_separator_bands(row_frac, MAJOR_SEP_THRESHOLD, MAJOR_SEP_MIN_WIDTH)
    major_cols = _find_separator_bands(col_frac, MAJOR_SEP_THRESHOLD, MAJOR_SEP_MIN_WIDTH)

    # Without force: need at least 2 separator bands on each axis
    if not force and len(major_rows) < 2 and len(major_cols) < 2:
        return []

    panel_rows = _bands_to_cells(major_rows, h, min_size=40)
    panel_cols = _bands_to_cells(major_cols, w, min_size=40)

    # Force mode: if no panels found, treat entire image as one panel
    if not panel_rows:
        panel_rows = [(0, h)]
    if not panel_cols:
        panel_cols = [(0, w)]

    avatars: list[str] = []
    for pr0, pr1 in panel_rows:
        for pc0, pc1 in panel_cols:
            panel = arr[pr0:pr1, pc0:pc1]
            ph, pw = panel.shape[:2]

            p_row_frac, p_col_frac = _border_fraction(panel, bg, BORDER_TOLERANCE)
            sub_row_bands = _find_separator_bands(p_row_frac, SEPARATOR_THRESHOLD, 2)
            sub_col_bands = _find_separator_bands(p_col_frac, SEPARATOR_THRESHOLD, 2)
            sub_rows = _bands_to_cells(sub_row_bands, ph, MIN_CELL_SIZE)
            sub_cols = _bands_to_cells(sub_col_bands, pw, MIN_CELL_SIZE)

            # Force mode: if no sub-cells, treat entire panel as one cell
            if force and not sub_rows:
                sub_rows = [(0, ph)]
            if force and not sub_cols:
                sub_cols = [(0, pw)]

            for sr0, sr1 in sub_rows:
                for sc0, sc1 in sub_cols:
                    cell = panel[sr0:sr1, sc0:sc1]
                    ch, cw = cell.shape[:2]
                    aspect = cw / max(ch, 1)
                    if ch < 50 or cw < 50 or aspect > 3.0 or aspect < 0.25:
                        if not force:
                            continue
                    trimmed = _trim_padding(cell, bg, BORDER_TOLERANCE)
                    th, tw = trimmed.shape[:2]
                    if th < MIN_AVATAR_DIM or tw < MIN_AVATAR_DIM:
                        if not force:
                            continue
                        trimmed = cell  # force: use untrimmed
                    pil = Image.fromarray(trimmed).resize(
                        (PROFILE_OUTPUT_SIZE, PROFILE_OUTPUT_SIZE), Image.NEAREST,
                    )
                    avatars.append(_image_to_b64(pil))

    return avatars


class ProfileRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded composite image")
    force: bool = Field(False, description="Force extraction, skip grid detection")


@app.post("/api/profiles")
async def api_profiles(req: ProfileRequest) -> JSONResponse:
    img = _decode_image(req.image)
    results = extract_profiles(img, force=req.force)
    return JSONResponse({"avatars": results, "count": len(results)})


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


@app.get("/profiles")
async def profiles_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "profiles.html", media_type="text/html")


@app.get("/health", response_class=PlainTextResponse)
async def health() -> str:
    return "ok"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
