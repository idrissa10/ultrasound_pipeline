"""
=============================================================
  CROP TOOL — Recadrer l'image avant de dessiner le masque
=============================================================

UTILISATION :
  python crop_image.py --image mon_image_ge.jpg

CONTRÔLES :
  - Clic gauche + glisser  → dessiner le rectangle de recadrage
  - Touche C               → confirmer et sauvegarder
  - Touche R               → recommencer
  - Touche Q / ESC         → quitter
=============================================================
"""

import cv2
import numpy as np
import argparse
import os
import sys

WIN_NAME   = "Crop Tool — Clique + glisse pour recadrer"
img_orig   = None
img_draw   = None
rect_start = None
rect_end   = None
drawing    = False


def redraw():
    global img_draw
    img_draw = img_orig.copy()

    if rect_start and rect_end:
        x1, y1 = rect_start
        x2, y2 = rect_end
        # Rectangle de sélection
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Zone sélectionnée en surbrillance
        overlay = img_draw.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 180, 0), -1)
        cv2.addWeighted(overlay, 0.2, img_draw, 0.8, 0, img_draw)
        # Dimensions
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        cv2.putText(img_draw, f"{w} x {h} px", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Bandeau bas
    h_img, w_img = img_draw.shape[:2]
    cv2.rectangle(img_draw, (0, h_img - 50), (w_img, h_img), (30, 30, 30), -1)
    cv2.putText(img_draw,
                "Clic + glisse: selectionner  |  C: Confirmer  |  R: Reset  |  Q: Quitter",
                (10, h_img - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if rect_start and rect_end:
        cv2.putText(img_draw, "Appuie sur C pour confirmer le recadrage",
                    (10, h_img - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow(WIN_NAME, img_draw)


def mouse_callback(event, x, y, flags, param):
    global drawing, rect_start, rect_end

    h_img = img_orig.shape[0]
    if y > h_img - 50:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing    = True
        rect_start = (x, y)
        rect_end   = None

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        rect_end = (x, y)
        redraw()

    elif event == cv2.EVENT_LBUTTONUP:
        drawing  = False
        rect_end = (x, y)
        redraw()


def run(image_path: str):
    global img_orig, img_draw, rect_start, rect_end, drawing

    if not os.path.exists(image_path):
        print(f"[ERREUR] Image introuvable : {image_path}")
        sys.exit(1)

    img_orig = cv2.imread(image_path)
    if img_orig is None:
        print(f"[ERREUR] Impossible de lire : {image_path}")
        sys.exit(1)

    h, w = img_orig.shape[:2]
    print(f"\n{'='*55}")
    print(f"  Image    : {os.path.basename(image_path)}")
    print(f"  Taille   : {w} × {h} px")
    print(f"{'='*55}")
    print("  → Clique et glisse pour sélectionner la zone du fan")
    print("  → C pour confirmer  |  Q pour quitter")
    print(f"{'='*55}\n")

    # Redimensionner pour l'affichage si trop grande
    display_scale = 1.0
    max_display   = 1100
    if w > max_display or h > max_display:
        display_scale = max_display / max(w, h)
        img_orig = cv2.resize(img_orig,
                              (int(w * display_scale), int(h * display_scale)),
                              interpolation=cv2.INTER_AREA)
        print(f"[INFO] Affichage redimensionné ({int(w*display_scale)} × {int(h*display_scale)})")

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN_NAME, mouse_callback)
    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key == ord("c") or key == ord("C"):
            if rect_start and rect_end:
                x1 = min(rect_start[0], rect_end[0])
                y1 = min(rect_start[1], rect_end[1])
                x2 = max(rect_start[0], rect_end[0])
                y2 = max(rect_start[1], rect_end[1])

                if (x2 - x1) < 10 or (y2 - y1) < 10:
                    print("[AVERTISSEMENT] Sélection trop petite, recommence.")
                    continue

                cropped = img_orig[y1:y2, x1:x2]

                base    = os.path.splitext(os.path.basename(image_path))[0]
                out     = f"{base}_cropped.png"
                cv2.imwrite(out, cropped)

                ch, cw = cropped.shape[:2]
                print(f"\n✅  Image recadrée sauvegardée → {out}")
                print(f"   Taille recadrée : {cw} × {ch} px")
                print(f"\n👉  Lance maintenant :")
                print(f"   python create_fan_mask.py --image {out} --probe GE\n")

                # Aperçu
                cv2.imshow("Résultat recadré — appuie sur une touche", cropped)
                cv2.waitKey(0)
                break
            else:
                print("[INFO] Dessine d'abord un rectangle.")

        elif key == ord("r") or key == ord("R"):
            rect_start = None
            rect_end   = None
            drawing    = False
            print("[RESET] Sélection effacée.")
            redraw()

        elif key == 27 or key == ord("q") or key == ord("Q"):
            print("[QUITTER] Aucune sauvegarde.")
            break

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Recadrer une image ultrason avant de dessiner le masque fan"
    )
    parser.add_argument("--image", required=True,
                        help="Chemin vers l'image à recadrer")
    args = parser.parse_args()
    run(args.image)


if __name__ == "__main__":
    main()
