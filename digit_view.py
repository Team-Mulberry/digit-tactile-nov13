"""Live DIGIT viewer: raw | contact diff vs baseline | rolling raster.
Keys: b = re-capture baseline, q/ESC = quit.  Usage: python digit_view.py [cam_index]"""
import sys, time
import cv2
import numpy as np

idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    sys.exit(f"Could not open camera {idx}")

W, H, RW = 320, 240, 400  # panel size, raster width (time samples)

def grab():
    ok, f = cap.read()
    if not ok:
        return None
    return cv2.resize(f, (W, H)).astype(np.float32)

for _ in range(15):  # let auto-exposure settle
    grab()
base = np.mean([grab() for _ in range(10)], axis=0)
raster = np.zeros((H, RW), np.float32)
t0, n, fps = time.time(), 0, 0.0

while True:
    f = grab()
    if f is None:
        print("frame read failed"); break
    diff = np.abs(f - base).mean(axis=2)                     # per-pixel change
    raster = np.roll(raster, -1, axis=1)
    raster[:, -1] = diff.mean(axis=1)                         # row profile -> one column
    sig = diff.mean()

    d_vis = cv2.applyColorMap(np.clip(diff * 4, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    r_vis = cv2.applyColorMap(np.clip(raster * 8, 0, 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    panel = np.hstack([f.astype(np.uint8), d_vis, r_vis])

    n += 1
    if time.time() - t0 > 1:
        fps, n, t0 = n / (time.time() - t0), 0, time.time()
    cv2.putText(panel, f"cam {idx}  {fps:4.1f} fps  contact={sig:5.2f}", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(panel, "raw | diff vs baseline | raster (rows x time)", (8, H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.imshow("DIGIT", panel)

    k = cv2.waitKey(1) & 0xFF
    if k in (ord("q"), 27):
        break
    if k == ord("b"):
        base = np.mean([grab() for _ in range(10)], axis=0)

cap.release()
cv2.destroyAllWindows()
