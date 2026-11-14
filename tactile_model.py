"""Shared bits: diff -> ResNet-18 (frozen) embedding, augmentation. Used by train.py and collect.py."""
import cv2
import numpy as np
import torch
import torchvision

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
_enc = None
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def encoder():
    global _enc
    if _enc is None:
        m = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        m.fc = torch.nn.Identity()
        _enc = m.eval().to(DEV)
    return _enc


def diff_map(frame, base):
    """frame/base uint8|float [H,W,3] -> float32 [H,W] abs change."""
    return np.abs(frame.astype(np.float32) - base.astype(np.float32)).mean(axis=2)


def contact(d):
    return float(d.mean())


def _contact_map(frame, base):
    """Drift-compensated, lightly blurred |change| map (global per-channel brightness shift removed)."""
    x = frame.astype(np.float32) - base.astype(np.float32)
    x -= np.median(x.reshape(-1, 3), axis=0)
    return cv2.GaussianBlur(np.abs(x).mean(axis=2), (0, 0), 2)


def contact_score(frame, base):
    """Contact detector: 99th percentile of the contact map -> sensitive to small local presses."""
    return float(np.percentile(_contact_map(frame, base), 99))


AREA_LEVEL = 8.0   # contact-map value counted as "in contact" (rest p99 ~ 4)
INTENSITY_NAMES = ["peak (p99)", "mean change", "contact area", "log total change"]


def intensity(frame, base):
    """How strongly / widely the gel deformed — the info per-frame normalisation throws away.
    Matters for hard vs soft. Flip/rotate/shift invariant, so augmentation reuses it."""
    m = _contact_map(frame, base)
    return np.array([np.percentile(m, 99), m.mean(), (m > AREA_LEVEL).mean(), np.log1p(m.sum())], np.float32)


def features(norm_diff, inten):
    """Final feature vector: ResNet-18 latent of the normalised diff (512) + intensity stats (4)."""
    return np.hstack([embed([norm_diff]), inten[None]])


def norm(d):
    """Per-frame normalise so overall brightness shifts don't dominate."""
    d = d - d.min()
    return d / (d.max() + 1e-6)


def edges(d):
    """Edges view: Sobel gradient magnitude of a (normalised) diff map, normalised again."""
    d = cv2.GaussianBlur(np.ascontiguousarray(d, dtype=np.float32), (0, 0), 3)   # smooth first: raw Sobel = noise
    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=5); gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=5)
    return norm(np.hypot(gx, gy))


def augment(d, rng):
    """Random flip, rotation (+-25deg) and shift (+-10%) of a normalised diff map."""
    if rng.random() < 0.5:
        d = d[:, ::-1]
    if rng.random() < 0.5:
        d = d[::-1, :]
    h, w = d.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-25, 25), 1.0)
    M[:, 2] += rng.uniform(-0.1, 0.1, 2) * (w, h)
    return cv2.warpAffine(np.ascontiguousarray(d), M, (w, h), borderMode=cv2.BORDER_CONSTANT)


@torch.no_grad()
def embed(diffs, bs=64):
    """list of normalised diff maps [H,W] -> np [N,512]."""
    out = []
    for i in range(0, len(diffs), bs):
        x = torch.from_numpy(np.stack([cv2.resize(d, (224, 224)) for d in diffs[i:i + bs]])).float()
        x = x[:, None].repeat(1, 3, 1, 1)
        x = ((x - MEAN) / STD).to(DEV)
        out.append(encoder()(x).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 512), np.float32)


# ---------------- force-vector field ----------------
# Per grid cell: (shear_x, shear_y, normal).
#   shear  = dense optical flow (px) of the gel between baseline and frame  -> tangential load
#   normal = drift-compensated local change magnitude (_contact_map)        -> indentation / pressure PROXY
# Uncalibrated (relative units). Proper depth needs a ball-press photometric calibration; proper newtons a load cell.
FIELD_GRID = (12, 16)   # rows, cols


def force_field(frame, base, full=False):
    """-> field [12,16,3] = (fx, fy, fz); with full=True also returns full-res (flow [H,W,2], normal [H,W])."""
    f = np.asarray(frame); b = np.asarray(base)
    g0 = cv2.cvtColor(b.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(f.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 21, 3, 7, 1.5, 0)
    nz = _contact_map(f, b)
    h, w = FIELD_GRID
    field = np.dstack([cv2.resize(flow[..., 0], (w, h), interpolation=cv2.INTER_AREA),
                       cv2.resize(flow[..., 1], (w, h), interpolation=cv2.INTER_AREA),
                       cv2.resize(nz, (w, h), interpolation=cv2.INTER_AREA)]).astype(np.float32)
    return (field, flow, nz) if full else field


FIELD_STAT_NAMES = ["total normal", "peak normal", "contact area", "mean pressure in contact",
                    "concentration (peak/mean)", "spatial spread", "eccentricity",
                    "mean |shear|", "net shear", "shear divergence", "|shear curl|"]


def field_stats(field):
    """Rotation/flip-invariant physical summary of one force field."""
    fx, fy, fz = field[..., 0], field[..., 1], field[..., 2]
    zc = np.clip(fz - np.median(fz), 0, None)                  # load above background
    tot = zc.sum() + 1e-6
    contact = fz > AREA_LEVEL
    area = contact.mean()
    yy, xx = np.mgrid[0:fz.shape[0], 0:fz.shape[1]].astype(np.float32)
    my, mx = (zc * yy).sum() / tot, (zc * xx).sum() / tot
    cyy = (zc * (yy - my) ** 2).sum() / tot; cxx = (zc * (xx - mx) ** 2).sum() / tot
    cxy = (zc * (xx - mx) * (yy - my)).sum() / tot
    ev = np.linalg.eigvalsh(np.array([[cxx, cxy], [cxy, cyy]]))
    spread = np.sqrt(ev.sum()); ecc = 1 - ev[0] / (ev[1] + 1e-6)
    mag = np.hypot(fx, fy)
    w = zc / tot
    div = np.gradient(fx, axis=1) + np.gradient(fy, axis=0)
    curl = np.gradient(fy, axis=1) - np.gradient(fx, axis=0)
    return np.array([np.log1p(tot), fz.max(), area, fz[contact].mean() if contact.any() else 0.0,
                     fz.max() / (fz.mean() + 1e-6), spread, ecc,
                     (mag * w).sum(), np.hypot((fx * w).sum(), (fy * w).sum()),
                     (div * w).sum(), (np.abs(curl) * w).sum()], np.float32)


def field_viz(frame, base, size=(640, 480), shear_gain=None):
    """Force-vector field view on the 12x16 grid:
       background = normal-component heatmap; per cell: dot (size ~ normal) + arrow (shear, auto-scaled)."""
    field, flow, nz = force_field(frame, base, full=True)
    W, H = size
    img = cv2.resize(cv2.applyColorMap(np.clip(nz * 6, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO), size)
    img = cv2.addWeighted(img, 0.75, cv2.resize(np.asarray(frame).astype(np.uint8), size), 0.25, 0)
    gh, gw = FIELD_GRID
    cw, ch = W / gw, H / gh
    fx, fy, fz = field[..., 0], field[..., 1], field[..., 2]
    zc = np.clip(fz - np.median(fz), 0, None)
    smax = float(np.hypot(fx, fy).max())
    if shear_gain is None:                                  # auto: longest arrow ~ 0.9 cell
        shear_gain = 0.9 * cw / max(smax, 0.25)   # floor: sub-0.25px noise stays tiny
    for r in range(gh):
        for c in range(gw):
            x, y = (c + 0.5) * cw, (r + 0.5) * ch
            rad = int(min(cw, ch) * 0.45 * min(zc[r, c] / 25.0, 1.0))
            if rad >= 1:
                cv2.circle(img, (int(x), int(y)), rad, (255, 255, 255), 1, cv2.LINE_AA)
            u, v = fx[r, c] * shear_gain, fy[r, c] * shear_gain
            if np.hypot(u, v) > 3:
                cv2.arrowedLine(img, (int(x), int(y)), (int(x + u), int(y + v)), (80, 255, 80), 2, cv2.LINE_AA, tipLength=0.35)
    st = field_stats(field)
    txt = [f"load {st[0]:.1f}  peak {st[1]:.0f}  area {st[2]*100:.0f}%  conc {st[4]:.1f}",
           f"shear max {smax:.2f}px  (arrows x{shear_gain:.0f})"]
    for i, t in enumerate(txt):
        cv2.putText(img, t, (8, 20 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, t, (8, 20 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "o = normal   -> = shear", (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img
