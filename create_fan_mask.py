"""
=============================================================
  OUTIL INTERACTIF — Création du masque fan (polygone)
=============================================================

UTILISATION :
  # Pour TELEMED :
  python create_fan_mask.py --image ma_image_telemed.png --probe TELEMED

  # Pour GE :
  python create_fan_mask.py --image ma_image_ge.png --probe GE

CONTRÔLES :
  - Clic gauche       → ajouter un point
  - Clic droit        → supprimer le dernier point
  - Touche  S         → sauvegarder et quitter
  - Touche  R         → recommencer (effacer tous les points)
  - Touche  Z         → annuler le dernier point
  - Touche  Q / ESC   → quitter sans sauvegarder

CONSEILS :
  - Commence par le coin supérieur gauche du fan
  - Suis le contour dans le sens horaire
  - 8 à 20 points suffisent pour une bonne précision
  - Pour un arc de cercle, ajoute plus de points sur la courbe
=============================================================
"""

import cv2
import json
import argparse
import numpy as np
import os
import sys

# ── couleurs ──────────────────────────────────────────────
COLOR_POINT   = (0, 255, 0)       # vert
COLOR_LINE    = (0, 200, 255)     # jaune-cyan
COLOR_POLYGON = (0, 255, 0)       # vert rempli semi-transparent
COLOR_LAST    = (0, 0, 255)       # rouge = dernier point
COLOR_TEXT    = (255, 255, 255)   # blanc
BG_TEXT       = (0, 0, 0)         # noir
RADIUS        = 5
THICKNESS     = 2

# ── état global ───────────────────────────────────────────
points     = []
img_orig   = None
img_draw   = None
WIN_NAME   = "Fan Mask Tool"


def show_help_panel(n_points):
    """Affiche les raccourcis dans une fenêtre séparée qui ne couvre pas l'image."""
    panel = np.zeros((200, 480, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)

    lines = [
        ("=== RACCOURCIS ===",         (0, 220, 255), 0.6),
        ("Clic gauche  : ajouter un point",   (255, 255, 255), 0.5),
        ("Clic droit   : supprimer dernier",  (255, 255, 255), 0.5),
        ("Z            : annuler dernier",    (255, 255, 255), 0.5),
        ("R            : tout effacer",       (255, 255, 255), 0.5),
        ("S            : sauvegarder",        (0, 255, 100),   0.5),
        ("Q / ESC      : quitter",            (100, 100, 255), 0.5),
        (f"Points poses : {n_points}",        (0, 220, 255),   0.55),
    ]

    y = 25
    for text, color, scale in lines:
        cv2.putText(panel, text, (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        y += 23

    cv2.imshow("Raccourcis", panel)


def redraw():
    """Redessine l'image avec les points et lignes actuels — sans bandeau."""
    global img_draw
    img_draw = img_orig.copy()

    n = len(points)

    # Remplissage semi-transparent si ≥ 3 points
    if n >= 3:
        overlay = img_draw.copy()
        pts     = np.array(points, np.int32)
        cv2.fillPoly(overlay, [pts], (0, 180, 0))
        cv2.addWeighted(overlay, 0.25, img_draw, 0.75, 0, img_draw)

    # Lignes entre les points
    for i in range(1, n):
        cv2.line(img_draw, points[i - 1], points[i], COLOR_LINE, THICKNESS)

    # Fermeture du polygone
    if n >= 3:
        cv2.line(img_draw, points[-1], points[0], COLOR_LINE, 1)

    # Points numérotés
    for i, p in enumerate(points):
        color = COLOR_LAST if i == n - 1 else COLOR_POINT
        cv2.circle(img_draw, p, RADIUS, color, -1)
        cv2.putText(img_draw, str(i + 1), (p[0] + 6, p[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)

    cv2.imshow(WIN_NAME, img_draw)
    show_help_panel(n)  # panneau séparé, ne touche pas l'image


def mouse_callback(event, x, y, flags, param):
    global points

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        redraw()

    elif event == cv2.EVENT_RBUTTONDOWN:
        if points:
            points.pop()
            redraw()


def save_json(probe: str, image_path: str):
    global points

    if len(points) < 3:
        print("[ERREUR] Il faut au moins 3 points pour définir un polygone.")
        return False

    # Résolution réelle de l'image
    h, w = img_orig.shape[:2]

    output = f"fan_{probe.lower()}.json"
    data   = {
        "probe":       probe,
        "image_size":  [w, h],
        "fan_polygon": [list(p) for p in points],
        "source_image": os.path.basename(image_path),
        "n_points":    len(points),
    }

    with open(output, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n✅  Masque sauvegardé → {output}")
    print(f"   Résolution : {w} × {h}")
    print(f"   Points     : {len(points)}")
    print(f"   Polygone   : {points}")
    return True


def preview_mask(probe: str):
    """Affiche un aperçu du masque final appliqué sur l'image."""
    if len(points) < 3:
        return

    preview = img_orig.copy()
    mask    = np.zeros(preview.shape[:2], dtype=np.uint8)
    pts     = np.array(points, np.int32)
    cv2.fillPoly(mask, [pts], 255)
    result  = cv2.bitwise_and(preview, preview, mask=mask)

    h, w    = preview.shape[:2]
    side    = np.zeros_like(preview)
    combined = np.hstack([preview, result])

    cv2.putText(combined, "Original", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(combined, f"Masque {probe}", (w + 10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Aperçu du masque — Appuie sur une touche pour fermer", combined)
    cv2.waitKey(0)
    cv2.destroyWindow("Aperçu du masque — Appuie sur une touche pour fermer")


def run(image_path: str, probe: str):
    global img_orig, img_draw, points

    if not os.path.exists(image_path):
        print(f"[ERREUR] Image introuvable : {image_path}")
        sys.exit(1)

    img_orig = cv2.imread(image_path)
    if img_orig is None:
        print(f"[ERREUR] Impossible de lire l'image : {image_path}")
        sys.exit(1)

    h, w = img_orig.shape[:2]
    print(f"\n{'='*55}")
    print(f"  Probe    : {probe}")
    print(f"  Image    : {image_path}  ({w} × {h})")
    print(f"{'='*55}")
    print("  → Clique sur les bords du fan ultrason")
    print("  → S pour sauvegarder  |  Q pour quitter")
    print(f"{'='*55}\n")

    # Redimensionner si l'image est très grande pour l'affichage
    display_scale = 1.0
    max_display   = 1100
    if w > max_display or h > max_display:
        display_scale = max_display / max(w, h)
        img_orig = cv2.resize(img_orig,
                              (int(w * display_scale), int(h * display_scale)),
                              interpolation=cv2.INTER_AREA)
        print(f"[INFO] Image redimensionnée pour l'affichage "
              f"({int(w*display_scale)} × {int(h*display_scale)})")

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN_NAME, mouse_callback)
    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key == ord("s") or key == ord("S"):
            preview_mask(probe)
            if save_json(probe, image_path):
                break
            else:
                print("Ajoute au moins 3 points avant de sauvegarder.")

        elif key == ord("r") or key == ord("R"):
            points = []
            print("[RESET] Tous les points effacés.")
            redraw()

        elif key == ord("z") or key == ord("Z"):
            if points:
                removed = points.pop()
                print(f"[ANNULER] Point {removed} supprimé.")
                redraw()

        elif key == 27 or key == ord("q") or key == ord("Q"):
            print("[QUITTER] Aucune sauvegarde.")
            break

    cv2.destroyAllWindows()


# =========================================================
# CLI
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Outil interactif de création de masque fan ultrason"
    )
    parser.add_argument("--image", required=True,
                        help="Chemin vers l'image exemple (TELEMED ou GE)")
    parser.add_argument("--probe", required=True, choices=["TELEMED", "GE"],
                        help="Type de sonde : TELEMED ou GE")
    args = parser.parse_args()

    run(args.image, args.probe.upper())


if __name__ == "__main__":
    main()