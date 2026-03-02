# 🔬 Ultrasound Preprocessing Pipeline

> Anonymization and preprocessing pipeline for ultrasound datasets — supports TELEMED and GE probes, multi-resolution fan masking, and automatic probe classification.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Scripts](#scripts)
  - [1. extract_frame.py](#1-extract_framepy)
  - [2. crop_image.py](#2-crop_imagepy)
  - [3. create_fan_mask.py](#3-create_fan_maskpy)
  - [4. ultrasound_pipeline.py](#4-ultrasound_pipelinepy)
  - [5. diagnose_folder.py](#5-diagnose_folderpy)
- [Fan Mask JSON Format](#fan-mask-json-format)
- [Probe Classification](#probe-classification)
- [Output Structure](#output-structure)
- [Workflow](#workflow)
- [Troubleshooting](#troubleshooting)

---

## Overview

This pipeline prepares raw ultrasound videos and images for model training by:

- **Separating** data by probe type (TELEMED vs GE) using black pixel ratio classification
- **Anonymizing** frames by applying a fan-shaped mask (removes patient overlays, depth scales, text)
- **Standardizing** output to a consistent video format (`.mp4`)
- **Supporting multiple resolutions** — each resolution can have its own fan mask
- **Logging** all processing steps and running quality control checks

---

## Project Structure

```
ultrasound-pipeline/
│
├── ultrasound_pipeline.py    # Main processing pipeline
├── create_fan_mask.py        # Interactive tool to draw fan polygon
├── extract_frame.py          # Extract a frame from a video
├── crop_image.py             # Crop image before drawing mask
├── diagnose_folder.py        # Diagnose a patient folder
├── requirements.txt          # Python dependencies
│
├── masks/                    # Fan mask JSON files (one per resolution)
│   ├── fan_telemed_1365x865.json
│   ├── fan_telemed_1172x608.json
│   ├── fan_telemed_800x600.json
│   └── fan_ge_912x683.json
│
├── dataset_raw/              # ← Raw input data (not tracked by git)
│   ├── P0001/
│   ├── P0002/
│   └── ...
│
└── dataset_raw_processed/    # ← Output (not tracked by git)
    ├── TELEMED/
    │   ├── video1_fan.mp4
    │   └── video1_meta.json
    └── GE/
        ├── video2_fan.mp4
        └── video2_meta.json
```

---

## Installation

### Requirements

- Python 3.8+
- Windows / Linux / macOS

### Install dependencies

```bash
pip install "numpy<2" opencv-python matplotlib
```

Or using the requirements file:

```bash
pip install -r requirements.txt
```

### requirements.txt

```
opencv-python
numpy<2
matplotlib
```

> ⚠️ NumPy 2.x is not compatible with the current version of matplotlib. Always use `numpy<2`.

---

## Quick Start

### 1. Extract a frame from your video

```bash
python extract_frame.py --video my_video.avi --output sample.png
```

### 2. Draw the fan mask interactively

```bash
python create_fan_mask.py --image sample.png --probe TELEMED
```

### 3. Run the full pipeline

```bash
python ultrasound_pipeline.py \
  --input dataset_raw/ \
  --masks_dir masks/ \
  --threshold 0.55 \
  --workers 4
```

---

## Scripts

### 1. `extract_frame.py`

Extracts a single frame from a video file and saves it as PNG.

```bash
python extract_frame.py --video <path_to_video> [--frame N] [--output name.png]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--video` | required | Path to the input video |
| `--frame` | `0` | Frame index to extract |
| `--output` | auto | Output PNG filename (accents stripped automatically) |

**Example:**
```bash
python extract_frame.py --video dataset_raw/P0007/video.avi --frame 10 --output p0007_sample.png
```

---

### 2. `crop_image.py`

Interactive tool to crop an image before drawing the mask. Useful when the image contains annotations, depth scales, or large black borders that interfere with mask drawing.

```bash
python crop_image.py --image <path_to_image>
```

**Controls:**

| Key | Action |
|-----|--------|
| Click + drag | Select crop region |
| `C` | Confirm and save cropped image |
| `R` | Reset selection |
| `Q` / `ESC` | Quit without saving |

---

### 3. `create_fan_mask.py`

Interactive tool to draw a polygon around the ultrasound fan region. Saves the result as a JSON mask file.

```bash
python create_fan_mask.py --image <path_to_image> --probe [TELEMED|GE]
```

**Controls:**

| Key / Action | Description |
|---|---|
| Left click | Add a point |
| Right click | Remove last point |
| `S` | Save mask + preview |
| `Z` | Undo last point |
| `R` | Reset all points |
| `Q` / `ESC` | Quit without saving |

**Tips:**
- Start at the top-center of the fan
- Follow the contour clockwise
- Use 8–20 points total
- Add more points on curved sections (bottom arc)
- The shortcuts panel opens in a **separate window** — it does not overlap the image

**Output:** `fan_telemed.json` or `fan_ge.json`

Rename the output file to include the resolution for multi-resolution support:
```bash
# Windows
rename fan_telemed.json fan_telemed_1172x608.json
move fan_telemed_1172x608.json masks\

# Linux/Mac
mv fan_telemed.json masks/fan_telemed_1172x608.json
```

---

### 4. `ultrasound_pipeline.py`

Main pipeline. Processes all videos and image sequences in a folder tree.

```bash
python ultrasound_pipeline.py \
  --input <dataset_folder> \
  --masks_dir <masks_folder> \
  --threshold 0.55 \
  --workers 4 \
  [--histogram] \
  [--n_samples 20] \
  [--resize WIDTH HEIGHT]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | required | Root folder containing patient subfolders |
| `--masks_dir` | `.` | Folder containing all `fan_*.json` files |
| `--threshold` | `0.35` | Black ratio threshold: `< threshold` → TELEMED, `>= threshold` → GE |
| `--workers` | `4` | Number of parallel processing threads |
| `--histogram` | off | Compute and save black ratio histogram before processing |
| `--n_samples` | `20` | Number of samples used for histogram |
| `--resize` | off | Optional output resize, e.g. `--resize 256 256` (default: original resolution) |

**Examples:**

```bash
# Basic run — keep original resolution
python ultrasound_pipeline.py --input dataset_raw/ --masks_dir masks/ --threshold 0.55

# With histogram to validate threshold
python ultrasound_pipeline.py --input dataset_raw/ --masks_dir masks/ --threshold 0.55 --histogram

# With resize to 512x512
python ultrasound_pipeline.py --input dataset_raw/ --masks_dir masks/ --threshold 0.55 --resize 512 512
```

**What it does for each file:**

```
FOR each video/image sequence:
    → Read first frame
    → Compute black_ratio
    → Classify probe (TELEMED or GE)
    → Select the correct fan mask (exact resolution or nearest)
    → Apply mask to every frame
    → Write output video to dataset_raw_processed/TELEMED/ or /GE/
    → Save metadata JSON
    → Run quality control check
```

---

### 5. `diagnose_folder.py`

Diagnoses a specific patient folder to identify why it failed to process.

```bash
python diagnose_folder.py --folder <path_to_folder> [--threshold 0.55]
```

Reports:
- All files found (name, size, extension)
- Whether OpenCV can open each video
- Resolution, FPS, frame count, codec
- Black ratio and detected probe type
- Saves a diagnostic frame PNG for visual inspection

---

## Fan Mask JSON Format

Each mask file must follow this format:

```json
{
  "probe": "TELEMED",
  "image_size": [1172, 608],
  "fan_polygon": [
    [550, 10],
    [300, 200],
    [150, 450],
    [200, 580],
    [580, 600],
    [970, 580],
    [1020, 450],
    [870, 200]
  ],
  "n_points": 8
}
```

| Field | Description |
|-------|-------------|
| `probe` | `TELEMED` or `GE` |
| `image_size` | `[width, height]` of the reference image used to draw the mask |
| `fan_polygon` | List of `[x, y]` points defining the fan region |

### Naming convention

```
fan_telemed_1172x608.json   ← TELEMED probe, resolution 1172x608
fan_telemed_800x600.json    ← TELEMED probe, resolution 800x600
fan_ge_912x683.json         ← GE probe, resolution 912x683
```

If no exact match is found for a video resolution, the pipeline automatically selects the nearest available mask and logs a warning.

---

## Probe Classification

Probes are classified by computing the **black pixel ratio** of the first frame:

```
black_ratio = number of pixels with intensity < 5 / total pixels
```

- `black_ratio < threshold` → **TELEMED** (less black border)
- `black_ratio >= threshold` → **GE** (larger black sector geometry)

### Choosing the right threshold

Run the pipeline with `--histogram` to visualize the distribution:

```bash
python ultrasound_pipeline.py --input dataset_raw/ --masks_dir masks/ --threshold 0.55 --histogram
```

This saves `black_ratio_histogram.png`. Look for the **gap** between the two clusters — place the threshold in that gap.

**Example from this dataset:**

| Cluster | Range | Probe |
|---------|-------|-------|
| Low | ~0.40–0.45 | TELEMED |
| High | ~0.58–0.82 | GE |
| **Threshold** | **0.55** | — |

---

## Output Structure

```
dataset_raw_processed/
├── TELEMED/
│   ├── video1_fan.mp4          # Anonymized video
│   ├── video1_meta.json        # Metadata
│   ├── video2_fan.mp4
│   └── video2_meta.json
└── GE/
    ├── video3_fan.mp4
    └── video3_meta.json
```

### Metadata file example (`video1_meta.json`)

```json
{
  "probe": "TELEMED",
  "black_ratio": 0.4231,
  "fps": 25.0,
  "frames": 120,
  "output_size": [1172, 608]
}
```

---

## Workflow

```
Raw data (videos / images)
        │
        ▼
extract_frame.py          ← Extract sample frame from video
        │
        ▼
crop_image.py             ← (optional) Remove borders/annotations
        │
        ▼
create_fan_mask.py        ← Draw polygon → fan_probe_WxH.json
        │
        ▼
ultrasound_pipeline.py    ← Full processing
   ├── inspect_dataset()
   ├── detect_probe()        (black ratio)
   ├── get_mask_for_frame()  (resolution-aware)
   ├── apply_fan_mask()
   ├── resize_frame()        (optional)
   ├── write video (.mp4)
   ├── save_metadata()
   └── quality_check()
        │
        ▼
dataset_raw_processed/
   ├── TELEMED/
   └── GE/
```

---

## Troubleshooting

### `numpy.core.multiarray failed to import`
NumPy 2.x is incompatible with your version of matplotlib.
```bash
pip install "numpy<2" --upgrade
pip install matplotlib --upgrade
```

### `cv2 cannot open video`
Codec may be missing. Convert with ffmpeg:
```bash
ffmpeg -i input.avi -c:v libx264 output.mp4
```

### `can't open/read file` with accents in filename
OpenCV does not support accented characters in file paths on Windows.
```cmd
rename "vidéo.png" "video.png"
```
`extract_frame.py` now strips accents automatically from output filenames.

### Fan mask too small / too large
The reference resolution in your JSON does not match the video resolution.
- Check: `python diagnose_folder.py --folder P000X`
- Fix: create a new mask using a frame from that specific video
- Or: add a dedicated `fan_probe_WxH.json` for that resolution in `masks/`

### Probe misclassified (TELEMED detected as GE or vice versa)
Run histogram to find the right threshold:
```bash
python ultrasound_pipeline.py --input dataset_raw/ --masks_dir masks/ --histogram --threshold 0.55
```
Then adjust `--threshold` accordingly.

---

## Notes

- Raw data, processed videos, and PNG/JPG files are excluded from git tracking via `.gitignore`
- Only code and mask JSON files are versioned
- All processing steps are logged to `pipeline.log`
- Test on 3–5 cases before running on the full dataset

---

*Pipeline developed for ultrasound spine dataset preprocessing — Dakar 2026*

