# IrisVision — Human Iris Localization & Validation

Real-time, in-browser localization of the human iris with an independent, classical-vision validation stage.
Everything runs on the client (WebAssembly + WebGL) — no frame ever leaves the device.

**Live demo:** https://subzero-005.github.io/iris-vision/

## Features

- **Live webcam, image upload or bundled sample photo** (drag & drop supported; `?sample` auto-runs it)
- **Iris localization** from MediaPipe Face Landmarker's 478 landmarks (5 per iris)
- **Sub-pixel circle fit** — centred algebraic (Kåsa) least-squares fit with RMS residual
- **Independent validation** of every fit:
  - OpenCV.js `HoughCircles` on a histogram-equalized eye crop (center error and radius ratio)
  - Radial intensity contrast between iris and sclera, sampled inside the eyelid polygon
  - Eye openness (EAR) so closed eyes are reported as `OCCLUDED`, not guessed
- **Eye-state analytics** — blink detection with hysteresis, blink rate, gaze direction with recentring
- **Distance estimation** from the ~11.7 mm human iris diameter and a pinhole camera model
- **Live charts** for EAR and validation score, plus a gaze heat map
- **Export** an annotated PNG or a JSON measurement report

## Pipeline

```
frame ─▶ Face Landmarker ─▶ iris limbus points ─▶ least-squares circle
                                                       │
            ┌──────────────────────────────────────────┤
            ▼                    ▼                     ▼
     Hough circle check    radial contrast       eye aspect ratio
            └──────────┬─────────┴─────────────────────┘
                       ▼
         score = 0.45·geometry + 0.35·contrast + 0.20·openness
                       ▼
            PASS (≥ 0.60) · FAIL · OCCLUDED (EAR < 0.17)
```

| Signal | Definition |
| --- | --- |
| Geometry | `(1 − Δc / 0.6r) · (1 − |1 − r_hough / r| / 0.5)` |
| Contrast | `(I_ring − I_iris) / I_ring`, iris ≤ 0.7r, ring 1.25r–1.9r |
| Openness | `clamp((EAR − 0.15) / 0.12)` |
| Distance | `Z = f · 11.7 mm / d_px`, `f = (W/2) / tan(HFOV/2)`, HFOV ≈ 60° |

If OpenCV.js fails to load, validation continues on contrast and openness alone.

## Run locally

Camera access requires a secure context, so serve the folder over `localhost`:

```bash
python3 -m http.server 8000
# open http://localhost:8000
```

## Deploy

The site is static. On GitHub: **Settings → Pages → Deploy from branch → `main` / root**.

## Tech

MediaPipe Tasks Vision · OpenCV.js 4.9 · Canvas 2D · vanilla ES modules — no build step.

## Limitations

IrisVision is a research and portfolio demo, not a certified biometric system. Distance assumes a ~60° webcam
field of view, and gaze is a geometric estimate without per-user calibration beyond recentring.

## Credits

Sample photo: astronaut Kalpana Chawla, official NASA portrait (public domain, via Wikimedia Commons).

## License

MIT
