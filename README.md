# palette.exe

Drop images → real pixel-counted color palettes. Brutalism-themed web UI + a plain-text API. No storage.

## Why

Most palette tools eyeball colors or use tiny sample grids. This one actually quantizes the pixels: PIL median-cut for 16 initial candidates, 2 k-means refinement passes, then cluster-size thresholding and perceptual deduplication. Output is 3–8 colors depending on the image.

## Run

```bash
uv sync
uv run python main.py         # http://localhost:8000
```

Or:

```bash
uv run uvicorn main:app --reload
```

## Web

Drop one or many images on the page. Each swatch is clickable:
- animates up-left, revealing a black square shadow (pure brutalism)
- hex is copied to clipboard

## API

`POST /api` with JSON. Accepts raw base64 or data URLs, up to 16 images, 10 MB each.

```bash
curl -s http://localhost:8000/api \
  -H 'Content-Type: application/json' \
  -d "{\"images\":[\"$(base64 -i photo.jpg)\"]}"
```

Response (`text/plain`, single image):

```
dominant: #2f4858
vibrant: #ff6b35
muted: #a7a8a6
dark: #181d20
```

Multiple images are separated by `---`:

```
# image 1
dominant: #2f4858
...
---
# image 2
dominant: #e5e7eb
...
```

## Labels

- `dominant` — largest cluster
- `vibrant` / `muted` / `dark` / `light` / `deep` / `soft` / `accent` — classified by HSL
- duplicate labels get numeric suffixes (`accent2`, `accent3`, ...)

## Algorithm

1. Resize to max 300 px (preserves palette, ~40× speedup).
2. `Image.quantize(16, MEDIANCUT, kmeans=2)` — true pixel quantization.
3. Drop clusters < 2% of pixels.
4. Sort by frequency. Merge clusters within RGB distance 28 of a larger one.
5. Cap at 8 colors.

## Not here

- No persistence. Images live only in the request lifecycle.
- No auth. Run behind whatever you like.
- No CORS allow-all — add middleware if you embed it cross-origin.
