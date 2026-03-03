"""
=============================================================
  DIAGNOSTIC — Analyze a specific patient folder
=============================================================

USAGE:
  python diagnose_folder.py --folder path/to/P0007
  python diagnose_folder.py --folder path/to/P0007 --threshold 0.55

=============================================================
"""

import os
import cv2
import json
import argparse
import numpy as np

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
VIDEO_EXT  = (".mp4", ".avi", ".mov", ".mkv", ".wmv")
BLACK_THRESHOLD = 5


def diagnose(folder: str, threshold: float = 0.55):
    print(f"\n{'='*60}")
    print(f"  DIAGNOSTIC: {folder}")
    print(f"{'='*60}\n")

    if not os.path.exists(folder):
        print(f"[ERROR] Folder not found: {folder}")
        return

    # ── 1. List all files ─────────────────────────────────
    all_files = []
    for root, dirs, files in os.walk(folder):
        print(f"[FOLDER] {root}")
        print(f"  Subfolders : {dirs}")
        for f in files:
            path = os.path.join(root, f)
            size = os.path.getsize(path)
            ext  = os.path.splitext(f)[1].lower()
            print(f"  [FILE] {f}  ({size} bytes)  ext={ext}")
            all_files.append((path, ext, size))

    print(f"\n-> Total files found : {len(all_files)}")

    videos = [(p, e, s) for p, e, s in all_files if e in VIDEO_EXT]
    images = [(p, e, s) for p, e, s in all_files if e in IMAGE_EXT]

    print(f"-> Videos : {len(videos)}")
    print(f"-> Images : {len(images)}")

    if not videos and not images:
        print("\n[PROBLEM] No image or video file recognized in this folder!")
        exts_found = set(e for _, e, _ in all_files)
        print(f"  Extensions found: {exts_found}")
        print(f"  Supported video : {VIDEO_EXT}")
        print(f"  Supported image : {IMAGE_EXT}")
        return

    # ── 2. Test each video ────────────────────────────────
    print(f"\n{'─'*60}")
    print("  VIDEO TESTS")
    print(f"{'─'*60}")

    for path, ext, size in videos:
        print(f"\n[VIDEO] {os.path.basename(path)}")
        print(f"  File size : {size} bytes")

        if size == 0:
            print("  [ERROR] Empty file (0 bytes)!")
            continue

        # Test open
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print("  [ERROR] cv2.VideoCapture cannot open this file!")
            print("  -> Possible cause: missing codec or corrupted file")
            print("  -> Fix: ffmpeg -i input.avi -c:v libx264 output.mp4")
            cap.release()
            continue

        fps      = cap.get(cv2.CAP_PROP_FPS)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fcc      = int(cap.get(cv2.CAP_PROP_FOURCC))
        fcc_str  = "".join([chr((fcc >> 8 * i) & 0xFF) for i in range(4)])

        print(f"  Resolution : {width} x {height}")
        print(f"  FPS        : {fps}")
        print(f"  Frames     : {n_frames}")
        print(f"  Codec      : {fcc_str}")

        # Test read first frame
        ret, frame = cap.read()
        if not ret or frame is None:
            print("  [ERROR] Cannot read the first frame!")
            cap.release()
            continue

        print(f"  First frame : OK  shape={frame.shape}")

        # Black ratio
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ratio = np.sum(gray < BLACK_THRESHOLD) / gray.size
        probe = "TELEMED" if ratio < threshold else "GE"

        print(f"  Black ratio : {ratio:.4f}  (threshold={threshold})")
        print(f"  -> Detected probe : {probe}")

        # Save diagnostic frame
        out_img = os.path.splitext(path)[0] + "_diag_frame0.png"
        cv2.imwrite(out_img, frame)
        print(f"  -> Diagnostic frame saved: {out_img}")

        cap.release()

    # ── 3. Test each image ────────────────────────────────
    if images:
        print(f"\n{'─'*60}")
        print("  IMAGE TESTS")
        print(f"{'─'*60}")

        for path, ext, size in images:
            print(f"\n[IMAGE] {os.path.basename(path)}")
            print(f"  File size : {size} bytes")

            if size == 0:
                print("  [ERROR] Empty file (0 bytes)!")
                continue

            img = cv2.imread(path)
            if img is None:
                print("  [ERROR] Cannot read image!")
                print("  -> Check for accented characters in the filename")
                continue

            h, w = img.shape[:2]
            gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ratio = np.sum(gray < BLACK_THRESHOLD) / gray.size
            probe = "TELEMED" if ratio < threshold else "GE"

            print(f"  Resolution  : {w} x {h}")
            print(f"  Black ratio : {ratio:.4f}  (threshold={threshold})")
            print(f"  -> Detected probe : {probe}")

    # ── 4. Summary ────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")

    all_ok = True
    for path, ext, size in videos:
        if size == 0:
            print(f"  [EMPTY]  {os.path.basename(path)}")
            all_ok = False
            continue
        cap    = cv2.VideoCapture(path)
        opened = cap.isOpened()
        cap.release()
        status = "OK" if opened else "FAIL"
        if not opened:
            all_ok = False
        print(f"  [{status}]  {os.path.basename(path)}")

    for path, ext, size in images:
        img    = cv2.imread(path)
        status = "OK" if img is not None else "FAIL"
        if img is None:
            all_ok = False
        print(f"  [{status}]  {os.path.basename(path)}")

    print()
    if all_ok:
        print("  All files opened successfully.")
        print(f"  If the mask is wrong, check --threshold (current={threshold})")
        print("  or create a new mask for this specific resolution.")
    else:
        print("  Some files failed. Possible fixes:")
        print("  1. Convert with ffmpeg: ffmpeg -i input.avi -c:v libx264 output.mp4")
        print("  2. Remove accented characters from filenames")
        print("  3. Check that the file is not corrupted")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose a patient folder to identify processing issues"
    )
    parser.add_argument("--folder",    required=True,
                        help="Path to the patient folder (e.g. P0007)")
    parser.add_argument("--threshold", type=float, default=0.55,
                        help="Black ratio threshold for probe detection (default: 0.55)")
    args = parser.parse_args()
    diagnose(args.folder, args.threshold)


if __name__ == "__main__":
    main()
