"""
=============================================================
  EXTRACT FRAME — Extraire une frame d'une vidéo
=============================================================

UTILISATION :
  # Extraire la première frame :
  python extract_frame.py --video ma_video.avi

  # Extraire une frame spécifique (ex: frame n°10) :
  python extract_frame.py --video ma_video.avi --frame 10

  # Choisir le nom du fichier de sortie :
  python extract_frame.py --video ma_video.avi --output telemed_sample.png
=============================================================
"""

import cv2
import argparse
import os
import sys


def extract_frame(video_path: str, frame_index: int = 0, output_path: str = None):

    if not os.path.exists(video_path):
        print(f"[ERREUR] Vidéo introuvable : {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERREUR] Impossible d'ouvrir la vidéo : {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n{'='*50}")
    print(f"  Vidéo      : {os.path.basename(video_path)}")
    print(f"  Résolution : {width} × {height}")
    print(f"  FPS        : {fps}")
    print(f"  Frames     : {total_frames}")
    print(f"{'='*50}")

    # Vérifier que l'index demandé est valide
    if frame_index >= total_frames:
        print(f"[AVERTISSEMENT] Frame {frame_index} inexistante "
              f"(total={total_frames}). Utilisation de la frame 0.")
        frame_index = 0

    # Aller à la frame souhaitée
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("[ERREUR] Impossible de lire la frame.")
        sys.exit(1)

    # Nom de sortie par défaut
    if output_path is None:
        base        = os.path.splitext(os.path.basename(video_path))[0]
        output_path = f"{base}_frame{frame_index}.png"

    cv2.imwrite(output_path, frame)
    print(f"\n✅  Frame extraite → {output_path}")
    print(f"   Index  : {frame_index}")
    print(f"   Taille : {width} × {height} px")
    print(f"\n👉  Lance maintenant :")
    print(f"   python create_fan_mask.py --image {output_path} --probe TELEMED\n")


def main():
    parser = argparse.ArgumentParser(
        description="Extraire une frame d'une vidéo pour créer le masque fan"
    )
    parser.add_argument("--video",  required=True,
                        help="Chemin vers la vidéo (.avi, .mp4, ...)")
    parser.add_argument("--frame",  type=int, default=0,
                        help="Index de la frame à extraire (défaut: 0 = première)")
    parser.add_argument("--output", default=None,
                        help="Nom du fichier PNG de sortie (optionnel)")
    args = parser.parse_args()

    extract_frame(args.video, args.frame, args.output)


if __name__ == "__main__":
    main()
