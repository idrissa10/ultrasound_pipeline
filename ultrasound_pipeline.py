import os
import cv2
import json
import argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging

# =========================================================
# CONFIG
# =========================================================
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
VIDEO_EXT  = (".mp4", ".avi", ".mov", ".mkv", ".wmv")
BLACK_THRESHOLD = 5
DEFAULT_FPS     = 25

# =========================================================
# LOGGING
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# =========================================================
# FAN MASK LOADING
# =========================================================
def load_fan(json_path: str):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Fan mask JSON not found: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)
    fan_polygon = np.array(data["fan_polygon"], np.int32)
    ref_size    = tuple(data["image_size"])  # (width, height)
    return fan_polygon, ref_size


def load_fan_masks(telemed_json: str, ge_json: str):
    telemed_poly, telemed_ref = load_fan(telemed_json)
    ge_poly,      ge_ref      = load_fan(ge_json)
    return {
        "TELEMED": (telemed_poly, telemed_ref),
        "GE":      (ge_poly,      ge_ref),
    }


# =========================================================
# BLACK RATIO
# =========================================================
def compute_black_ratio(frame: np.ndarray) -> float:
    gray         = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    black_pixels = np.sum(gray < BLACK_THRESHOLD)
    return black_pixels / gray.size


# =========================================================
# PROBE DETECTION
# =========================================================
def detect_probe(frame: np.ndarray, threshold: float):
    ratio = compute_black_ratio(frame)
    probe = "TELEMED" if ratio < threshold else "GE"
    return probe, ratio


# =========================================================
# HISTOGRAM
# =========================================================
def compute_histogram(input_dir: str, threshold: float, n_samples: int = 20,
                      save_path: str = "black_ratio_histogram.png"):
    ratios    = []
    collected = 0

    for root, _, files in os.walk(input_dir):
        if collected >= n_samples:
            break
        for f in files:
            if collected >= n_samples:
                break
            ext  = os.path.splitext(f)[1].lower()
            path = os.path.join(root, f)
            frame = None
            if ext in IMAGE_EXT:
                frame = cv2.imread(path)
            elif ext in VIDEO_EXT:
                cap = cv2.VideoCapture(path)
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    frame = None
            if frame is not None:
                ratios.append(compute_black_ratio(frame))
                collected += 1

    if not ratios:
        log.warning("No samples found for histogram.")
        return

    plt.figure(figsize=(8, 5))
    plt.hist(ratios, bins=15, color="steelblue", edgecolor="white")
    plt.axvline(threshold, color="red", linestyle="--",
                label=f"Threshold = {threshold:.2f}")
    plt.title(f"Black Ratio Histogram ({len(ratios)} samples)")
    plt.xlabel("black_ratio")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    log.info(f"Histogram saved → {save_path}")
    log.info(f"Black ratios — min:{min(ratios):.3f}  max:{max(ratios):.3f}  "
             f"mean:{np.mean(ratios):.3f}  std:{np.std(ratios):.3f}")
    return ratios


# =========================================================
# APPLY FAN MASK
# =========================================================
def apply_fan_mask(img: np.ndarray, fan_polygon: np.ndarray,
                   ref_size: tuple) -> np.ndarray:
    h, w         = img.shape[:2]
    ref_w, ref_h = ref_size

    scaled = fan_polygon.astype(np.float32).copy()
    scaled[:, 0] *= w / ref_w
    scaled[:, 1] *= h / ref_h
    scaled = scaled.astype(np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [scaled], 255)

    return cv2.bitwise_and(img, img, mask=mask)


# =========================================================
# RESIZE  — optionnel, uniquement si target_size est fourni
# =========================================================
def resize_frame(frame: np.ndarray, target_size: tuple = None) -> np.ndarray:
    if target_size is None:
        return frame  # ← résolution originale conservée
    return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)


# =========================================================
# PROCESS A SINGLE FRAME
# =========================================================
def process_frame(frame, fan_polygon, ref_size, target_size=None):
    masked = apply_fan_mask(frame, fan_polygon, ref_size)
    return resize_frame(masked, target_size)


# =========================================================
# IMAGE SEQUENCE → VIDEO
# =========================================================
def images_to_video(image_paths: list, output_video: str,
                    fan_masks: dict, threshold: float, target_size=None):
    image_paths = sorted(image_paths, key=lambda p: os.path.basename(p))

    first = cv2.imread(image_paths[0])
    if first is None:
        log.error(f"Cannot read first image: {image_paths[0]}")
        return None, None, 0

    probe, ratio          = detect_probe(first, threshold)
    fan_polygon, ref_size = fan_masks[probe]

    # Résolution de sortie
    if target_size:
        out_w, out_h = target_size
    else:
        out_h, out_w = first.shape[:2]  # résolution originale

    os.makedirs(os.path.dirname(output_video) or ".", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video, fourcc, DEFAULT_FPS, (out_w, out_h))
    count  = 0

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            log.warning(f"Skipped unreadable image: {path}")
            continue
        writer.write(process_frame(img, fan_polygon, ref_size, target_size))
        count += 1

    writer.release()
    parent_folder = os.path.basename(os.path.dirname(output_video))
    log.info(f"[OK] dossier={parent_folder} | fichier={os.path.basename(output_video)} | probe={probe} | size={out_w}x{out_h} | frames={count} (converti depuis images)")
    return probe, ratio, count


# =========================================================
# VIDEO PROCESSING
# =========================================================
def process_video(path: str, output_dir: str,
                  fan_masks: dict, threshold: float, target_size=None):
    if "_fan" in os.path.basename(path).lower():
        log.info(f"[SKIP] already processed: {path}")
        return

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        log.error(f"Cannot open video: {path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    ret, first_frame = cap.read()
    if not ret:
        log.error(f"Cannot read first frame: {path}")
        cap.release()
        return

    probe, ratio          = detect_probe(first_frame, threshold)
    fan_polygon, ref_size = fan_masks[probe]

    # Résolution de sortie
    if target_size:
        out_w, out_h = target_size
    else:
        out_h, out_w = first_frame.shape[:2]  # résolution originale

    probe_dir = os.path.join(output_dir, probe)
    os.makedirs(probe_dir, exist_ok=True)

    base      = os.path.splitext(os.path.basename(path))[0]
    out_video = os.path.join(probe_dir, base + "_fan.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video, fourcc, fps, (out_w, out_h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(process_frame(frame, fan_polygon, ref_size, target_size))
        count += 1

    cap.release()
    writer.release()

    save_metadata(probe_dir, base, probe, ratio, fps, count, out_w, out_h)
    parent_folder = os.path.basename(os.path.dirname(path))
    log.info(f"[OK] dossier={parent_folder} | fichier={os.path.basename(path)} | probe={probe} | size={out_w}x{out_h} | frames={count}")


# =========================================================
# METADATA
# =========================================================
def save_metadata(folder, name, probe, ratio, fps, frames, out_w, out_h):
    meta = {
        "probe":       probe,
        "black_ratio": float(ratio),
        "fps":         float(fps),
        "frames":      int(frames),
        "output_size": [out_w, out_h],
    }
    with open(os.path.join(folder, name + "_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


# =========================================================
# DATASET INSPECTION
# =========================================================
def inspect_dataset(input_dir: str) -> dict:
    stats      = defaultdict(int)
    case_count = defaultdict(int)
    img_count  = 0
    vid_count  = 0

    for root, _, files in os.walk(input_dir):
        folder_name = os.path.relpath(root, input_dir)
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            stats[ext]              += 1
            case_count[folder_name] += 1
            if ext in IMAGE_EXT:
                img_count += 1
            elif ext in VIDEO_EXT:
                vid_count += 1

    log.info("\n===== DATASET INSPECTION REPORT =====")
    log.info(f"Total image files : {img_count}")
    log.info(f"Total video files : {vid_count}")
    log.info("File formats found:")
    for k, v in stats.items():
        log.info(f"  {k}: {v} file(s)")
    log.info("Files per folder:")
    for folder, cnt in case_count.items():
        log.info(f"  {folder}: {cnt} file(s)")
    log.info("======================================\n")
    return dict(stats)


# =========================================================
# QUALITY CONTROL
# =========================================================
def quality_check(video_path: str, target_size=None) -> bool:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.warning(f"[QC FAIL] Cannot open {video_path}")
        return False

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if target_size:
        ok = (w == target_size[0] and h == target_size[1] and n > 0)
    else:
        ok = (w > 0 and h > 0 and n > 0)

    status = "OK" if ok else "FAIL"
    log.info(f"[QC {status}] {os.path.basename(video_path)}  "
             f"size={w}x{h}  frames={n}")
    return ok


# =========================================================
# FULL PIPELINE
# =========================================================
def process_folder(input_dir: str, fan_masks: dict, threshold: float,
                   max_workers: int = 4, target_size=None):
    if not os.path.exists(input_dir):
        log.error(f"Input directory not found: {input_dir}")
        return

    output_dir = input_dir.rstrip("\\/") + "_processed"
    os.makedirs(output_dir, exist_ok=True)

    if target_size:
        log.info(f"[INFO] Resize activé → {target_size[0]}x{target_size[1]}")
    else:
        log.info("[INFO] Résolution originale conservée (pas de resize)")

    inspect_dataset(input_dir)

    tasks = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for root, _, files in os.walk(input_dir):
            images = []
            videos = []

            for f in files:
                path = os.path.join(root, f)
                ext  = os.path.splitext(f)[1].lower()
                if ext in IMAGE_EXT:
                    images.append(path)
                elif ext in VIDEO_EXT:
                    videos.append(path)

            # ── Log du contenu du dossier ──────────────────
            if videos or images:
                folder_name = os.path.relpath(root, input_dir)
                total_files = len(videos) + (1 if images else 0)
                log.info(f"")
                log.info(f"{'─'*55}")
                log.info(f"[DOSSIER] {folder_name}  →  "
                         f"{len(videos)} vidéo(s)  {len(images)} image(s)")
                for v in videos:
                    cap = cv2.VideoCapture(v)
                    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
                    cap.release()
                    log.info(f"   • {os.path.basename(v)}  →  {n} frames")
                if images:
                    log.info(f"   • {len(images)} image(s) → seront converties en 1 vidéo")
                log.info(f"{'─'*55}")

            for v in videos:
                tasks.append(
                    executor.submit(
                        process_video, v, output_dir,
                        fan_masks, threshold, target_size
                    )
                )

            if images:
                base      = os.path.basename(root)
                out_video = os.path.join(output_dir, base + "_fan.mp4")
                tasks.append(
                    executor.submit(
                        images_to_video,
                        images, out_video, fan_masks, threshold, target_size
                    )
                )

        for t in tasks:
            try:
                t.result()
            except Exception as e:
                log.error(f"Task failed: {e}")

    log.info("\n===== QUALITY CONTROL =====")
    for probe in ("TELEMED", "GE"):
        probe_dir = os.path.join(output_dir, probe)
        if not os.path.exists(probe_dir):
            continue
        for f in os.listdir(probe_dir):
            if f.endswith(".mp4"):
                quality_check(os.path.join(probe_dir, f), target_size)

    log.info(f"[DONE] Output → {output_dir}")


# =========================================================
# CLI
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Ultrasound anonymisation & preprocessing pipeline"
    )
    parser.add_argument("--input",       required=True,
                        help="Root dataset folder")
    parser.add_argument("--fan_telemed", default="fan_telemed.json")
    parser.add_argument("--fan_ge",      default="fan_ge.json")
    parser.add_argument("--threshold",   type=float, default=0.35,
                        help="Black ratio threshold (TELEMED < t <= GE)")
    parser.add_argument("--workers",     type=int, default=4)
    parser.add_argument("--histogram",   action="store_true",
                        help="Compute & save black ratio histogram")
    parser.add_argument("--n_samples",   type=int, default=20)
    parser.add_argument("--resize",      type=int, nargs=2,
                        metavar=("WIDTH", "HEIGHT"), default=None,
                        help="Resize optionnel ex: --resize 256 256  "
                             "(par défaut: résolution originale conservée)")

    args = parser.parse_args()

    fan_masks   = load_fan_masks(args.fan_telemed, args.fan_ge)
    target_size = tuple(args.resize) if args.resize else None

    if args.histogram:
        compute_histogram(args.input, args.threshold, args.n_samples)

    process_folder(args.input, fan_masks, args.threshold,
                   args.workers, target_size)


if __name__ == "__main__":
    main()
import os
import cv2
import json
import argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging

# =========================================================
# CONFIG
# =========================================================
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
VIDEO_EXT  = (".mp4", ".avi", ".mov", ".mkv", ".wmv")
BLACK_THRESHOLD = 5
DEFAULT_FPS     = 25

# =========================================================
# LOGGING
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# =========================================================
# FAN MASK LOADING
# =========================================================
def load_fan(json_path: str):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Fan mask JSON not found: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)
    fan_polygon = np.array(data["fan_polygon"], np.int32)
    ref_size    = tuple(data["image_size"])  # (width, height)
    return fan_polygon, ref_size


def load_fan_masks(telemed_json: str, ge_json: str):
    telemed_poly, telemed_ref = load_fan(telemed_json)
    ge_poly,      ge_ref      = load_fan(ge_json)
    return {
        "TELEMED": (telemed_poly, telemed_ref),
        "GE":      (ge_poly,      ge_ref),
    }


# =========================================================
# BLACK RATIO
# =========================================================
def compute_black_ratio(frame: np.ndarray) -> float:
    gray         = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    black_pixels = np.sum(gray < BLACK_THRESHOLD)
    return black_pixels / gray.size


# =========================================================
# PROBE DETECTION
# =========================================================
def detect_probe(frame: np.ndarray, threshold: float):
    ratio = compute_black_ratio(frame)
    probe = "TELEMED" if ratio < threshold else "GE"
    return probe, ratio


# =========================================================
# HISTOGRAM
# =========================================================
def compute_histogram(input_dir: str, threshold: float, n_samples: int = 20,
                      save_path: str = "black_ratio_histogram.png"):
    ratios    = []
    collected = 0

    for root, _, files in os.walk(input_dir):
        if collected >= n_samples:
            break
        for f in files:
            if collected >= n_samples:
                break
            ext  = os.path.splitext(f)[1].lower()
            path = os.path.join(root, f)
            frame = None
            if ext in IMAGE_EXT:
                frame = cv2.imread(path)
            elif ext in VIDEO_EXT:
                cap = cv2.VideoCapture(path)
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    frame = None
            if frame is not None:
                ratios.append(compute_black_ratio(frame))
                collected += 1

    if not ratios:
        log.warning("No samples found for histogram.")
        return

    plt.figure(figsize=(8, 5))
    plt.hist(ratios, bins=15, color="steelblue", edgecolor="white")
    plt.axvline(threshold, color="red", linestyle="--",
                label=f"Threshold = {threshold:.2f}")
    plt.title(f"Black Ratio Histogram ({len(ratios)} samples)")
    plt.xlabel("black_ratio")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    log.info(f"Histogram saved → {save_path}")
    log.info(f"Black ratios — min:{min(ratios):.3f}  max:{max(ratios):.3f}  "
             f"mean:{np.mean(ratios):.3f}  std:{np.std(ratios):.3f}")
    return ratios


# =========================================================
# APPLY FAN MASK
# =========================================================
def apply_fan_mask(img: np.ndarray, fan_polygon: np.ndarray,
                   ref_size: tuple) -> np.ndarray:
    h, w         = img.shape[:2]
    ref_w, ref_h = ref_size

    scaled = fan_polygon.astype(np.float32).copy()
    scaled[:, 0] *= w / ref_w
    scaled[:, 1] *= h / ref_h
    scaled = scaled.astype(np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [scaled], 255)

    return cv2.bitwise_and(img, img, mask=mask)


# =========================================================
# RESIZE  — optionnel, uniquement si target_size est fourni
# =========================================================
def resize_frame(frame: np.ndarray, target_size: tuple = None) -> np.ndarray:
    if target_size is None:
        return frame  # ← résolution originale conservée
    return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)


# =========================================================
# PROCESS A SINGLE FRAME
# =========================================================
def process_frame(frame, fan_polygon, ref_size, target_size=None):
    masked = apply_fan_mask(frame, fan_polygon, ref_size)
    return resize_frame(masked, target_size)


# =========================================================
# IMAGE SEQUENCE → VIDEO
# =========================================================
def images_to_video(image_paths: list, output_video: str,
                    fan_masks: dict, threshold: float, target_size=None):
    image_paths = sorted(image_paths, key=lambda p: os.path.basename(p))

    first = cv2.imread(image_paths[0])
    if first is None:
        log.error(f"Cannot read first image: {image_paths[0]}")
        return None, None, 0

    probe, ratio          = detect_probe(first, threshold)
    fan_polygon, ref_size = fan_masks[probe]

    # Résolution de sortie
    if target_size:
        out_w, out_h = target_size
    else:
        out_h, out_w = first.shape[:2]  # résolution originale

    os.makedirs(os.path.dirname(output_video) or ".", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video, fourcc, DEFAULT_FPS, (out_w, out_h))
    count  = 0

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            log.warning(f"Skipped unreadable image: {path}")
            continue
        writer.write(process_frame(img, fan_polygon, ref_size, target_size))
        count += 1

    writer.release()
    parent_folder = os.path.basename(os.path.dirname(output_video))
    log.info(f"[OK] dossier={parent_folder} | fichier={os.path.basename(output_video)} | probe={probe} | size={out_w}x{out_h} | frames={count} (converti depuis images)")
    return probe, ratio, count


# =========================================================
# VIDEO PROCESSING
# =========================================================
def process_video(path: str, output_dir: str,
                  fan_masks: dict, threshold: float, target_size=None):
    if "_fan" in os.path.basename(path).lower():
        log.info(f"[SKIP] already processed: {path}")
        return

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        log.error(f"Cannot open video: {path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    ret, first_frame = cap.read()
    if not ret:
        log.error(f"Cannot read first frame: {path}")
        cap.release()
        return

    probe, ratio          = detect_probe(first_frame, threshold)
    fan_polygon, ref_size = fan_masks[probe]

    # Résolution de sortie
    if target_size:
        out_w, out_h = target_size
    else:
        out_h, out_w = first_frame.shape[:2]  # résolution originale

    probe_dir = os.path.join(output_dir, probe)
    os.makedirs(probe_dir, exist_ok=True)

    base      = os.path.splitext(os.path.basename(path))[0]
    out_video = os.path.join(probe_dir, base + "_fan.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video, fourcc, fps, (out_w, out_h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(process_frame(frame, fan_polygon, ref_size, target_size))
        count += 1

    cap.release()
    writer.release()

    save_metadata(probe_dir, base, probe, ratio, fps, count, out_w, out_h)
    parent_folder = os.path.basename(os.path.dirname(path))
    log.info(f"[OK] dossier={parent_folder} | fichier={os.path.basename(path)} | probe={probe} | size={out_w}x{out_h} | frames={count}")


# =========================================================
# METADATA
# =========================================================
def save_metadata(folder, name, probe, ratio, fps, frames, out_w, out_h):
    meta = {
        "probe":       probe,
        "black_ratio": float(ratio),
        "fps":         float(fps),
        "frames":      int(frames),
        "output_size": [out_w, out_h],
    }
    with open(os.path.join(folder, name + "_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


# =========================================================
# DATASET INSPECTION
# =========================================================
def inspect_dataset(input_dir: str) -> dict:
    stats      = defaultdict(int)
    case_count = defaultdict(int)
    img_count  = 0
    vid_count  = 0

    for root, _, files in os.walk(input_dir):
        folder_name = os.path.relpath(root, input_dir)
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            stats[ext]              += 1
            case_count[folder_name] += 1
            if ext in IMAGE_EXT:
                img_count += 1
            elif ext in VIDEO_EXT:
                vid_count += 1

    log.info("\n===== DATASET INSPECTION REPORT =====")
    log.info(f"Total image files : {img_count}")
    log.info(f"Total video files : {vid_count}")
    log.info("File formats found:")
    for k, v in stats.items():
        log.info(f"  {k}: {v} file(s)")
    log.info("Files per folder:")
    for folder, cnt in case_count.items():
        log.info(f"  {folder}: {cnt} file(s)")
    log.info("======================================\n")
    return dict(stats)


# =========================================================
# QUALITY CONTROL
# =========================================================
def quality_check(video_path: str, target_size=None) -> bool:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.warning(f"[QC FAIL] Cannot open {video_path}")
        return False

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if target_size:
        ok = (w == target_size[0] and h == target_size[1] and n > 0)
    else:
        ok = (w > 0 and h > 0 and n > 0)

    status = "OK" if ok else "FAIL"
    log.info(f"[QC {status}] {os.path.basename(video_path)}  "
             f"size={w}x{h}  frames={n}")
    return ok


# =========================================================
# FULL PIPELINE
# =========================================================
def process_folder(input_dir: str, fan_masks: dict, threshold: float,
                   max_workers: int = 4, target_size=None):
    if not os.path.exists(input_dir):
        log.error(f"Input directory not found: {input_dir}")
        return

    output_dir = input_dir.rstrip("\\/") + "_processed"
    os.makedirs(output_dir, exist_ok=True)

    if target_size:
        log.info(f"[INFO] Resize activé → {target_size[0]}x{target_size[1]}")
    else:
        log.info("[INFO] Résolution originale conservée (pas de resize)")

    inspect_dataset(input_dir)

    tasks = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for root, _, files in os.walk(input_dir):
            images = []
            videos = []

            for f in files:
                path = os.path.join(root, f)
                ext  = os.path.splitext(f)[1].lower()
                if ext in IMAGE_EXT:
                    images.append(path)
                elif ext in VIDEO_EXT:
                    videos.append(path)

            # ── Log du contenu du dossier ──────────────────
            if videos or images:
                folder_name = os.path.relpath(root, input_dir)
                total_files = len(videos) + (1 if images else 0)
                log.info(f"")
                log.info(f"{'─'*55}")
                log.info(f"[DOSSIER] {folder_name}  →  "
                         f"{len(videos)} vidéo(s)  {len(images)} image(s)")
                for v in videos:
                    cap = cv2.VideoCapture(v)
                    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
                    cap.release()
                    log.info(f"   • {os.path.basename(v)}  →  {n} frames")
                if images:
                    log.info(f"   • {len(images)} image(s) → seront converties en 1 vidéo")
                log.info(f"{'─'*55}")

            for v in videos:
                tasks.append(
                    executor.submit(
                        process_video, v, output_dir,
                        fan_masks, threshold, target_size
                    )
                )

            if images:
                base      = os.path.basename(root)
                out_video = os.path.join(output_dir, base + "_fan.mp4")
                tasks.append(
                    executor.submit(
                        images_to_video,
                        images, out_video, fan_masks, threshold, target_size
                    )
                )

        for t in tasks:
            try:
                t.result()
            except Exception as e:
                log.error(f"Task failed: {e}")

    log.info("\n===== QUALITY CONTROL =====")
    for probe in ("TELEMED", "GE"):
        probe_dir = os.path.join(output_dir, probe)
        if not os.path.exists(probe_dir):
            continue
        for f in os.listdir(probe_dir):
            if f.endswith(".mp4"):
                quality_check(os.path.join(probe_dir, f), target_size)

    log.info(f"[DONE] Output → {output_dir}")


# =========================================================
# CLI
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Ultrasound anonymisation & preprocessing pipeline"
    )
    parser.add_argument("--input",       required=True,
                        help="Root dataset folder")
    parser.add_argument("--fan_telemed", default="fan_telemed.json")
    parser.add_argument("--fan_ge",      default="fan_ge.json")
    parser.add_argument("--threshold",   type=float, default=0.35,
                        help="Black ratio threshold (TELEMED < t <= GE)")
    parser.add_argument("--workers",     type=int, default=4)
    parser.add_argument("--histogram",   action="store_true",
                        help="Compute & save black ratio histogram")
    parser.add_argument("--n_samples",   type=int, default=20)
    parser.add_argument("--resize",      type=int, nargs=2,
                        metavar=("WIDTH", "HEIGHT"), default=None,
                        help="Resize optionnel ex: --resize 256 256  "
                             "(par défaut: résolution originale conservée)")

    args = parser.parse_args()

    fan_masks   = load_fan_masks(args.fan_telemed, args.fan_ge)
    target_size = tuple(args.resize) if args.resize else None

    if args.histogram:
        compute_histogram(args.input, args.threshold, args.n_samples)

    process_folder(args.input, fan_masks, args.threshold,
                   args.workers, target_size)


if __name__ == "__main__":
    main()
