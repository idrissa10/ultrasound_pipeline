# 🔬 Ultrasound Preprocessing Pipeline

> Anonymization and preprocessing pipeline for ultrasound datasets — supports TELEMED and GE probes, multi-resolution fan masking, and automatic probe classification.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [🚀 Where to Start — Step-by-Step Guide](#-where-to-start--step-by-step-guide)
  - [Step 0 — Get a Reference PNG Image](#step-0--get-a-reference-png-image)
  - [Step 1 — Crop the Image (Optional)](#step-1--crop-the-image-optional)
  - [Step 2 — Draw the Fan Mask](#step-2--draw-the-fan-mask)
  - [Step 3 — Organize the Masks](#step-3--organize-the-masks)
  - [Step 4 — Run the Full Pipeline](#step-4--run-the-full-pipeline)
- [Scripts — Full Reference](#scripts--full-reference)
- [Fan Mask JSON Format](#fan-mask-json-format)
- [Probe Classification](#probe-classification)
- [Output Structure](#output-structure)
- [Troubleshooting](#troubleshooting)

---

## Overview

This pipeline prepares raw ultrasound videos and images for model training by:

- **Separating** data by probe type (TELEMED vs GE) using black pixel ratio classification
- **Anonymizing** frames by applying a fan-shaped mask (removes patient overlays, depth scales, text)
- **Standardizing** output to a consistent video format (`.mp4`)
- **Supporting multiple resolutions** — each resolution can have its own fan mask
- **Logging** all processing steps and running automatic quality control checks

---

## Project Structure

```
ultrasound-pipeline/
├── ultrasound_pipeline.py    # Main processing pipeline
├── create_fan_mask.py        # Interactive tool to draw the fan polygon
├── extract_frame.py          # Extract a frame from a video
├── crop_image.py             # Crop image before drawing the mask
├── diagnose_folder.py        # Diagnose a specific patient folder
├── requirements.txt
│
├── masks/                    # Fan mask JSON files (one per resolution)
│   ├── fan_telemed_1172x704.json
│   ├── fan_telemed_1172x608.json
│   ├── fan_telemed_800x600.json
│   └── fan_ge_912x683.json
│
├── dataset_raw/              # Raw input data
│   ├── P0001/
│   ├── P0002/
│   └── ...
│
└── dataset_raw_processed/    # Output generated automatically
    ├── TELEMED/
    └── GE/
```

---

## Installation

**Requirements:** Python 3.8+, Windows / Linux / macOS

```bash
pip install "numpy<2" opencv-python matplotlib
```

> ⚠️ NumPy 2.x is not compatible with matplotlib. Always use `numpy<2`.

**`requirements.txt`**
```
opencv-python
numpy<2
matplotlib
```

---

## 🚀 Where to Start — Step-by-Step Guide

> Before running the pipeline, you need to create the **fan mask JSON files** for each probe type and each resolution. These files define the useful ultrasound fan region to keep.
>
> ✅ **The mask is created only once per probe and per resolution.** Once the JSON files are ready, you can process your entire dataset.

---

### Step 0 — Get a Reference PNG Image

The mask is drawn on a **still image** (PNG or JPG).
Two cases depending on what you have:

---

#### ✅ Case A — You already have a JPG or PNG image

Skip directly to Step 2.

---

#### 📹 Case B — You only have a video (most common case)

Use `extract_frame.py` to extract a frame from your video:

```bash
python extract_frame.py --video my_telemed_video.avi --output telemed_sample.png
```

By default, the **first frame** is extracted. If it is black or blurry, try a later frame:

```bash
python extract_frame.py --video my_telemed_video.avi --frame 30 --output telemed_sample.png
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--video` | required | Path to the video file (.avi, .mp4, ...) |
| `--frame` | `0` | Index of the frame to extract |
| `--output` | automatic | Output PNG filename |

> ⚠️ **Accented characters in filenames**: OpenCV does not support accented characters on Windows.
> `extract_frame.py` automatically strips accents from generated filenames.
> If you provide a name manually with `--output`, avoid accented characters (é, è, à...).

**Result:** a PNG image extracted from your video, ready for mask drawing.

---

### Step 1 — Crop the Image (Optional)

If your image contains **wide black borders, hospital text, or depth scales** that interfere with mask drawing, crop it first:

```bash
python crop_image.py --image telemed_sample.png
```

A window opens. Click and drag to select only the fan area, then press `C` to confirm.

→ Generates `telemed_sample_cropped.png`

**When to use this step?**
- ✅ The image has annotations or text outside the fan
- ✅ The GE probe displays a depth scale on the side
- ❌ Not needed if the fan already occupies most of the image

---

### Step 2 — Draw the Fan Mask

Launch the interactive tool with your reference image:

```bash
# For a TELEMED probe
python create_fan_mask.py --image telemed_sample.png --probe TELEMED

# For a GE probe
python create_fan_mask.py --image ge_sample.jpg --probe GE
```

**Two windows open:**
- `Fan Mask Tool` — your image in full screen, click freely anywhere
- `Shortcuts` — separate help panel, can be moved aside

**How to draw the polygon:**

```
        ●──────────────●       <- start here (top of the fan)
       /                 \
      ●                   ●   <- follow the contour
     /                     \
    ●───────────────────────●  <- follow the bottom arc (more points here)
```

- Start at the **top of the fan** and follow the contour **clockwise**
- **8 to 20 points** are enough for good precision
- Add more points along the curved bottom arc to follow it accurately

**Keyboard shortcuts:**

| Key | Action |
|-----|--------|
| Left click | Add a point |
| Right click | Remove the last point |
| `Z` | Undo last point |
| `R` | Reset all points |
| `S` | Save + preview |
| `Q` / `ESC` | Quit without saving |

After pressing `S`, a side-by-side preview is shown (original vs masked). Confirm if the result looks correct.

→ Generates `fan_telemed.json` or `fan_ge.json`

---

### Step 3 — Organize the Masks

#### If you have only one resolution per probe

```cmd
mkdir masks
move fan_telemed.json masks\fan_telemed.json
move fan_ge.json masks\fan_ge.json
```

#### If you have multiple resolutions (common case)

Rename each JSON file with the corresponding resolution, then move it to `masks/`:

```cmd
mkdir masks

rem TELEMED 1172x704
python extract_frame.py --video path\to\video_1172x704.avi --output ref_1172x704.png
python create_fan_mask.py --image ref_1172x704.png --probe TELEMED
move fan_telemed.json masks\fan_telemed_1172x704.json

rem TELEMED 1172x608
python extract_frame.py --video path\to\video_1172x608.avi --output ref_1172x608.png
python create_fan_mask.py --image ref_1172x608.png --probe TELEMED
move fan_telemed.json masks\fan_telemed_1172x608.json

rem TELEMED 800x600
python extract_frame.py --video path\to\video_800x600.avi --output ref_800x600.png
python create_fan_mask.py --image ref_800x600.png --probe TELEMED
move fan_telemed.json masks\fan_telemed_800x600.json

rem GE (move existing file)
move fan_ge.json masks\fan_ge_912x683.json
```

> 💡 **How to find the resolutions in your dataset?**
> ```cmd
> python -c "
> import cv2, os
> folder = r'path\to\dataset'
> resolutions = {}
> for root, dirs, files in os.walk(folder):
>     for f in files:
>         if f.endswith(('.mp4', '.avi')):
>             cap = cv2.VideoCapture(os.path.join(root, f))
>             if cap.isOpened():
>                 w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
>                 h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
>                 resolutions.setdefault(f'{w}x{h}', []).append(f)
>             cap.release()
> for res, files in resolutions.items():
>     print(f'{res} -> {len(files)} video(s)')
> "
> ```

---

### Step 4 — Run the Full Pipeline

Once all mask JSON files are ready in the `masks/` folder:

```bash
python ultrasound_pipeline.py \
  --input dataset_raw/ \
  --masks_dir masks/ \
  --threshold 0.55 \
  --workers 4
```

**On Windows:**
```cmd
python ultrasound_pipeline.py --input dataset_raw --masks_dir masks --threshold 0.55 --workers 4
```

**With histogram** (recommended the first time to validate the threshold):
```cmd
python ultrasound_pipeline.py --input dataset_raw --masks_dir masks --threshold 0.55 --histogram
```

**With resize** (if the model requires a fixed input size):
```cmd
python ultrasound_pipeline.py --input dataset_raw --masks_dir masks --threshold 0.55 --resize 512 512
```

> 💡 Test on 3 to 5 cases before running on the full dataset.

Results are automatically generated in `dataset_raw_processed/`.

---

## Scripts — Full Reference

### `extract_frame.py`
Extracts a single frame from a video and saves it as PNG. Accented characters are automatically stripped from the output filename.

```bash
python extract_frame.py --video <video> [--frame N] [--output name.png]
```

---

### `crop_image.py`
Interactive tool to crop an image before drawing the fan mask.

```bash
python crop_image.py --image <image>
```

| Key | Action |
|-----|--------|
| Click + drag | Select crop region |
| `C` | Confirm and save |
| `R` | Reset selection |
| `Q` / `ESC` | Quit without saving |

---

### `create_fan_mask.py`
Interactive tool to draw the fan polygon and generate the JSON mask file.

```bash
python create_fan_mask.py --image <image> --probe [TELEMED|GE]
```

---

### `ultrasound_pipeline.py`
Main pipeline. Processes all patient folders in parallel.

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | required | Root folder of the raw dataset |
| `--masks_dir` | `.` | Folder containing all `fan_*.json` files |
| `--threshold` | `0.55` | Black ratio threshold: `< threshold` = TELEMED, `>= threshold` = GE |
| `--workers` | `4` | Number of parallel threads |
| `--histogram` | off | Generate black ratio histogram |
| `--n_samples` | `20` | Number of samples for histogram |
| `--resize` | off | Optional resize e.g. `--resize 256 256` (default: original resolution kept) |

---

### `diagnose_folder.py`
Diagnoses a specific patient folder to identify processing issues.

```bash
python diagnose_folder.py --folder path/to/P0007 [--threshold 0.55]
```

Reports: files found, codec, resolution, FPS, frame count, black ratio, detected probe. Saves a diagnostic PNG frame for visual inspection.

---

## Fan Mask JSON Format

```json
{
  "probe": "TELEMED",
  "image_size": [1172, 608],
  "fan_polygon": [
    [550, 10], [300, 200], [150, 450],
    [200, 580], [580, 600], [970, 580],
    [1020, 450], [870, 200]
  ],
  "n_points": 8
}
```

| Field | Description |
|-------|-------------|
| `probe` | `TELEMED` or `GE` |
| `image_size` | `[width, height]` of the reference image used to draw the mask |
| `fan_polygon` | List of `[x, y]` points defining the fan region |

**Naming convention:**
```
fan_telemed_1172x608.json   <- TELEMED probe, resolution 1172x608
fan_telemed_800x600.json    <- TELEMED probe, resolution 800x600
fan_ge_912x683.json         <- GE probe, resolution 912x683
```

If no exact resolution match is found, the pipeline automatically selects the nearest available mask and logs a warning in `pipeline.log`.

---

## Probe Classification

```
black_ratio = number of pixels with intensity < 5 / total pixels
```

- `black_ratio < threshold` → **TELEMED** (less black border)
- `black_ratio >= threshold` → **GE** (larger black sector geometry)

### Choosing the right threshold

Run with `--histogram` to visualize the distribution and find the gap between the two clusters:

```bash
python ultrasound_pipeline.py --input dataset_raw/ --masks_dir masks/ --threshold 0.55 --histogram
```

**Example from this dataset:**

| Cluster | Range | Probe |
|---------|-------|-------|
| Low | ~0.40–0.45 | TELEMED |
| High | ~0.58–0.82 | GE |
| **Recommended threshold** | **0.55** | — |

---

## Output Structure

```
dataset_raw_processed/
├── TELEMED/
│   ├── video1_fan.mp4          # Anonymized video
│   ├── video1_meta.json        # Metadata
│   └── ...
└── GE/
    ├── video2_fan.mp4
    ├── video2_meta.json
    └── ...
```

**Metadata file example:**
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

## Troubleshooting

### `numpy.core.multiarray failed to import`
NumPy 2.x is incompatible with your version of matplotlib.
```bash
pip install "numpy<2" --upgrade && pip install matplotlib --upgrade
```

### `cv2 cannot open video` — missing codec
Convert the video using ffmpeg:
```bash
ffmpeg -i input.avi -c:v libx264 output.mp4
```

### `can't open/read file` — accented characters in filename
OpenCV does not support accented characters in file paths on Windows.
```cmd
rename "vidéo.png" "video.png"
```
`extract_frame.py` strips accents automatically from generated filenames.

### Fan mask too small or too large
The reference resolution in your JSON does not match the video resolution.
```bash
python diagnose_folder.py --folder path/to/P000X
```
Create a new mask using a frame extracted from that specific video, rename it with the correct resolution, and place it in `masks/`.

### Probe misclassified (TELEMED detected as GE or vice versa)
```bash
python ultrasound_pipeline.py --input dataset_raw/ --masks_dir masks/ --histogram --threshold 0.55
```
Adjust `--threshold` based on the gap visible in the histogram.

---

*Pipeline developed for ultrasound spine dataset preprocessing — Dakar 2026*
