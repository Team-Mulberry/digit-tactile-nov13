"""Minimal DIGIT tactile episode collector.  Run: .venv/bin/python collect.py [cam_index]
Open http://localhost:5050.  Episodes saved to episodes/<id>.npz (frames uint8 [T,H,W,3], t, label)."""
import json, os, sys, threading, time
import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

IDX = int(sys.argv[1]) if len(sys.argv) > 1 else 0
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "episodes")
os.makedirs(OUT, exist_ok=True)

app = Flask(__name__)
lock = threading.Lock()
state = {"frame": None, "rec": False, "frames": [], "ts": [], "label": "", "t0": 0, "cam": "connecting", "reconnect": False,
         "base": None, "rebase": True, "pred": None,
         "mode": "manual", "padding": 0.5, "min_press": 0.5, "sensitivity": 1.35, "rest": None, "contact": 0.0, "thresh": None, "auto_saved": None}
from collections import deque
AUTO_K = 1.5          # auto threshold = AUTO_K / sensitivity x resting contact noise (sensitivity 1.0 = default)
THRESH_FLOOR = 3.0


def update_thresh():
    """Caller holds lock. Recompute threshold from measured rest noise + current sensitivity."""
    if state["rest"] is not None:
        state["thresh"] = max(AUTO_K / state["sensitivity"] * state["rest"], THRESH_FLOOR)


def row_profile(f):
    a = f.astype(np.float32).mean(axis=(1, 2)); return a - a.mean()


def detect_roll(prof, ref):
    """Vertical wrap-around shift of a frame vs reference row-brightness profile (circular x-corr)."""
    c = np.real(np.fft.ifft(np.fft.fft(prof) * np.conj(np.fft.fft(ref))))
    k = int(np.argmax(c)); return k if k <= len(prof) // 2 else k - len(prof)


def open_cam():
    """Try preferred index, then others; DIGIT streams 640x480 natively."""
    for i in [IDX] * 4 + [j for j in range(5) if j != IDX]:   # retry DIGIT before falling back
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # explicit format/size/fps:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)   # reduces UVC
        cap.set(cv2.CAP_PROP_FPS, 30)                                                    # frame desync
        ok, f = cap.read() if cap.isOpened() else (False, None)
        if ok and f.shape[:2] == (480, 640):
            return cap, i
        cap.release()
    return None, None


def capture():
    cap, fails, ref, warm = None, 0, None, []
    while True:
        if cap is None or state["reconnect"]:
            if cap is not None:
                cap.release()
            with lock:
                state.update(cam="connecting", reconnect=False, frame=None)
            cap, i = open_cam()
            with lock:
                state["cam"] = f"connected (cam {i})" if cap else "disconnected"
            if cap is None:
                time.sleep(2); continue          # auto-retry every 2s
            fails = 0
            for _ in range(30):                  # let auto-exposure settle (~1s) before baseline
                cap.read()
            ref, warm = None, []
            state["rebase"] = True
        ok, f = cap.read()
        if not ok:
            fails += 1
            if fails > 30:                       # ~1s of failures -> reopen
                cap.release(); cap = None
            time.sleep(0.03); continue
        fails = 0
        f = cv2.resize(f, (320, 240))
        # --- de-roll: UVC desync sometimes wraps the image vertically (bottom shows at top) ---
        prof = row_profile(f)
        if ref is None:                          # reference = median profile of first 15 frames
            warm.append(prof)
            if len(warm) >= 15:
                ref = np.median(warm, axis=0)
        else:
            k = detect_roll(prof, ref)
            if abs(k) > 3:
                f = np.roll(f, -k, axis=0)
                state["rolls"] = state.get("rolls", 0) + 1
                if state["rolls"] % 20 == 1:
                    print(f"de-rolled frame by {k} rows (total {state['rolls']})")
            else:
                ref = 0.98 * ref + 0.02 * prof   # track slow lighting drift
        with lock:
            state["frame"] = f
            if state["rebase"]:
                state["_bq"] = state.get("_bq", []) + [f.astype(np.float32)]
                if len(state["_bq"]) >= 10:
                    state["base"], state["_bq"], state["rebase"] = np.mean(state["_bq"], 0), [], False
                    state["base_ver"] = state.get("base_ver", 0) + 1
            now = time.time()
            if state["rec"]:
                state["frames"].append(f); state["ts"].append(now)
            base, mode, pad, bver = state["base"], state["mode"], state["padding"], state.get("base_ver", 0)
            hold = state["min_press"]
        # ---- contact sensing (for auto mode) ----
        if base is None:
            continue
        import tactile_model as tm
        c = tm.contact_score(f, base)
        if bver != auto["ver"]:                            # new baseline -> recalibrate rest noise (~1s)
            auto.update(ver=bver, calib=[], run_start=None, last_contact=0)
            with lock:
                state["thresh"] = state["rest"] = None
        if state["thresh"] is None:
            auto["calib"].append(c)
            if len(auto["calib"]) >= 30:
                with lock:
                    state["rest"] = float(np.median(auto["calib"])); update_thresh()
            with lock:
                state["contact"] = c
            continue
        thr = state["thresh"]
        touching = c >= thr
        if touching:
            if auto["run_start"] is None:
                auto["run_start"] = now                      # start of current continuous press
            auto["last_contact"] = now
        else:
            auto["run_start"] = None
        auto["buf"].append((f, now))
        while auto["buf"] and now - auto["buf"][0][1] > pad + hold + 0.2:   # pre-roll + press-confirm window
            auto["buf"].popleft()
        to_save = None
        with lock:
            state["contact"] = c
            if mode == "auto":
                pressed_for = now - auto["run_start"] if auto["run_start"] is not None else -1
                if not state["rec"] and pressed_for >= max(hold, 0.05):   # continuous press >= min_press -> start
                    pre = [p for p in auto["buf"] if p[1] >= auto["run_start"] - pad]   # include `padding` before press began
                    state.update(rec=True, frames=[p[0] for p in pre], ts=[p[1] for p in pre],
                                 t0=pre[0][1] if pre else now, auto_rec=True)
                elif state["rec"] and state.get("auto_rec") and now - auto["last_contact"] >= pad:
                    to_save = stop_locked()                       # quiet for `padding` s -> stop (tail = padding)
            if not state["rec"] and not touching:            # slow baseline drift tracking while idle
                state["base"] = 0.98 * base + 0.02 * f.astype(np.float32)
        if to_save:
            meta = write_episode(*to_save)
            with lock:
                state["auto_saved"] = meta


auto = {"ver": -1, "calib": [], "run_start": None, "last_contact": 0, "buf": deque()}


def stop_locked():
    """Stop recording; return (frames, ts, label). Caller holds `lock`."""
    state["rec"] = False; state["auto_rec"] = False
    out = (state["frames"], state["ts"], state["label"])
    state["frames"], state["ts"] = [], []
    return out


def write_episode(frames, ts, label):
    if not frames:
        return None
    eid = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time()*1000)%1000:03d}"
    np.savez_compressed(os.path.join(OUT, eid + ".npz"), frames=np.stack(frames),
                        t=np.array(ts) - ts[0], label=label)
    meta = {"id": eid, "label": label, "frames": len(frames), "duration": round(ts[-1] - ts[0], 2)}
    with open(os.path.join(OUT, eid + ".json"), "w") as fh:
        json.dump(meta, fh)
    return meta


MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.joblib")


def predictor():
    """Classify the latest live frame continuously (frame by frame)."""
    import joblib
    import tactile_model as tm
    model, mtime, ema = None, 0, None
    calib, last_base, thresh = [], None, None
    while True:
        if os.path.exists(MODEL_PATH) and os.path.getmtime(MODEL_PATH) != mtime:   # hot-reload after retrain
            mtime = os.path.getmtime(MODEL_PATH); model = joblib.load(MODEL_PATH); ema = None
            print("loaded model", model["labels"])
        with lock:
            f, base = state["frame"], state["base"]
        if model is None or f is None or base is None:
            time.sleep(0.2); continue
        d = tm.diff_map(f, base); c = tm.contact_score(f, base)
        if state.get("base_ver", 0) != last_base:  # new baseline -> recalibrate rest noise (~1s, don't touch)
            last_base, calib, thresh = state.get("base_ver", 0), [], None
        if thresh is None:
            calib.append(c)
            if len(calib) >= 30:
                thresh = 1.4 * float(np.median(calib))   # live rest noise; model value only a fallback
            with lock:
                state["pred"] = {"contact": round(c, 2), "label": None, "probs": {}, "thresh": 0, "calibrating": True}
            time.sleep(0.03); continue
        if c < thresh:
            ema = None
            pred = {"contact": round(c, 2), "label": None, "probs": {}}
        else:
            feats = model.get("features", "resnet")
            if feats == "field":
                z = tm.field_stats(tm.force_field(f, base))[None]
            else:
                z = tm.features(tm.norm(d), tm.intensity(f, base))  # diff latent (512) + intensity (4)
                if feats == "both":
                    z = np.hstack([z, tm.field_stats(tm.force_field(f, base))[None]])
            if "ovo" in model:                                         # one-vs-one: a model per pair
                pairs = list(model["ovo"])
                p = np.array([model["ovo"][k].predict_proba(z)[0, 1] for k in pairs])   # P(first of pair)
            else:
                pairs = None; names = model["clf"].classes_.tolist(); p = model["clf"].predict_proba(z)[0]
            ema = p if ema is None else 0.6 * ema + 0.4 * p            # light smoothing across frames
            if pairs:
                sc = {l: 0.0 for l in model["labels"]}
                for (a, b), v in zip(pairs, ema):
                    sc[a] += v; sc[b] += 1 - v
                probs = {str(l): round(float(v) / (len(model["labels"]) - 1), 3) for l, v in sc.items()}   # share of pairwise votes
                duels = [{"a": str(a), "b": str(b), "p": round(float(v), 3)} for (a, b), v in zip(pairs, ema)]
            else:
                probs = {str(k): float(v) for k, v in zip(names, ema.round(3))}; duels = []
            best = max(probs, key=probs.get)
            pred = {"contact": round(c, 2), "label": best, "probs": probs, "duels": duels}
        pred["thresh"] = round(thresh, 2)
        with lock:
            state["pred"] = pred
        time.sleep(0.01)


_eps_cache = {"key": None, "eps": []}


@app.after_request
def _invalidate_eps(resp):
    from flask import request as _rq
    if _rq.method == "POST":
        _eps_cache["key"] = None          # any change (label/delete/archive...) -> re-scan next time
    return resp


def episodes():
    key = os.stat(OUT).st_mtime_ns        # dir mtime changes when episodes are added/removed
    if _eps_cache["key"] == key:
        return _eps_cache["eps"]
    eps = []
    for fn in sorted(os.listdir(OUT), reverse=True):
        if fn.endswith(".json"):
            with open(os.path.join(OUT, fn)) as fh:
                eps.append(json.load(fh))
    _eps_cache.update(key=key, eps=eps)
    return eps


@app.route("/")
def index():
    return render_template("index.html")


# (old full review UI archived in templates/_archived/review.html)


# ---------- classes: name + reference photo (classes/<name>.jpg, order in classes/classes.json) ----------
CLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "classes")
os.makedirs(CLS, exist_ok=True)
CLS_JSON = os.path.join(CLS, "classes.json")


def class_names():
    return json.load(open(CLS_JSON)) if os.path.exists(CLS_JSON) else []


def safe(name):
    name = name.strip()
    return name if name and all(ch.isalnum() or ch in " _-" for ch in name) else None


# ---------- datasets: archive current episodes+classes, restore later ----------
import shutil
ARCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "archives")
os.makedirs(ARCH, exist_ok=True)


def summarize(ep_dir):
    counts = {}
    for fn in os.listdir(ep_dir) if os.path.isdir(ep_dir) else []:
        if fn.endswith(".json"):
            l = json.load(open(os.path.join(ep_dir, fn))).get("label", "")
            counts[l] = counts.get(l, 0) + 1
    return counts


def move_dataset(src_ep, src_cls, dst_ep, dst_cls):
    os.makedirs(dst_ep, exist_ok=True); os.makedirs(dst_cls, exist_ok=True)
    for a, b in ((src_ep, dst_ep), (src_cls, dst_cls)):
        for fn in os.listdir(a) if os.path.isdir(a) else []:
            shutil.move(os.path.join(a, fn), os.path.join(b, fn))


def archive_current(name):
    if not os.listdir(OUT) and not class_names():
        return None
    aid = time.strftime("%Y-%m-%d_%H%M%S") + ("_" + name if name else "")
    move_dataset(OUT, CLS, os.path.join(ARCH, aid, "episodes"), os.path.join(ARCH, aid, "classes"))
    _cache.clear()
    return aid


@app.route("/api/datasets")
def api_datasets():
    arcs = []
    for aid in sorted(os.listdir(ARCH), reverse=True):
        d = os.path.join(ARCH, aid)
        if os.path.isdir(d):
            arcs.append({"id": aid, "counts": summarize(os.path.join(d, "episodes")),
                         "classes": [c[:-4] for c in os.listdir(os.path.join(d, "classes")) if c.endswith(".jpg")]
                         if os.path.isdir(os.path.join(d, "classes")) else []})
    return jsonify(current=summarize(OUT), archives=arcs)


@app.post("/api/archive")
def api_archive():
    if state["rec"]:
        return jsonify(ok=False, error="stop recording first"), 400
    n = request.json.get("name", "").strip()
    if n and not safe(n):
        return jsonify(ok=False, error="letters, numbers, space, _ - only"), 400
    aid = archive_current((n or "").replace(" ", "-"))
    return jsonify(ok=bool(aid), id=aid, error=None if aid else "nothing to archive")


@app.post("/api/restore/<aid>")
def api_restore(aid):
    d = os.path.join(ARCH, aid)
    if state["rec"] or "/" in aid or aid.startswith(".") or not os.path.isdir(d):
        return jsonify(ok=False), 400
    saved = archive_current("autosave")            # never lose what's currently there
    move_dataset(os.path.join(d, "episodes"), os.path.join(d, "classes"), OUT, CLS)
    shutil.rmtree(d)
    _cache.clear()
    return jsonify(ok=True, autosaved=saved)


@app.route("/api/classes")
def api_classes():
    counts = {}
    for e in episodes():
        counts[e["label"]] = counts.get(e["label"], 0) + 1
    return jsonify([{"name": n, "count": counts.get(n, 0),
                     "img": os.path.exists(os.path.join(CLS, n + ".jpg"))} for n in class_names()])


@app.post("/api/classes")
def api_add_class():
    n = safe(request.json.get("name", ""))
    if not n:
        return jsonify(ok=False, error="letters, numbers, space, _ - only"), 400
    names = class_names()
    if n not in names:
        names.append(n); json.dump(names, open(CLS_JSON, "w"))
    return jsonify(ok=True, name=n)


@app.post("/api/classes/<name>/delete")
def api_del_class(name):
    """Remove class + its episodes + photo. Moved to trash/<time>_<name>/ (not destroyed)."""
    if name not in class_names() or state["rec"]:
        return jsonify(ok=False), 400
    import shutil
    tdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trash", time.strftime("%Y-%m-%d_%H%M%S_") + name)
    os.makedirs(tdir, exist_ok=True)
    n = 0
    for e in episodes():
        if e["label"] == name:
            for ext in (".npz", ".json"):
                p = os.path.join(OUT, e["id"] + ext)
                if os.path.exists(p):
                    shutil.move(p, tdir)
            n += 1
    img = os.path.join(CLS, name + ".jpg")
    if os.path.exists(img):
        shutil.move(img, tdir)
    json.dump([c for c in class_names() if c != name], open(CLS_JSON, "w"))
    _cache.clear()
    return jsonify(ok=True, episodes=n)


@app.post("/api/classes/<name>/image")
def api_class_image(name):
    if name not in class_names():
        return jsonify(ok=False), 404
    open(os.path.join(CLS, name + ".jpg"), "wb").write(request.get_data())
    return jsonify(ok=True)


@app.route("/classes/<name>.jpg")
def class_img(name):
    from flask import send_from_directory
    return send_from_directory(CLS, name + ".jpg", max_age=0)


@app.route("/stream")
def stream():
    view = request.args.get("view", "raw")
    force = view == "force"
    def gen():
        import tactile_model as tm
        while True:
            with lock:
                f, base = state["frame"], state["base"]
            if f is not None:
                if force and base is not None:
                    img = tm.field_viz(f, base)
                elif view == "diff" and base is not None:       # viewing only; recordings stay raw
                    m = tm._contact_map(f, base)
                    img = cv2.addWeighted(f, 0.35, cv2.applyColorMap(np.clip(m * 6, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO), 0.9, 0)
                else:
                    img = f
                jpg = cv2.imencode(".jpg", img)[1].tobytes()
                yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            time.sleep(1 / 30)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=f")


@app.route("/api/state")
def api_state():
    with lock:
        s = {"rec": state["rec"], "label": state["label"], "n": len(state["frames"]),
             "elapsed": time.time() - state["t0"] if state["rec"] else 0, "cam": state["cam"], "pred": state["pred"], "rolls": state.get("rolls", 0),
             "mode": state["mode"], "padding": state["padding"], "min_press": state["min_press"], "sensitivity": state["sensitivity"], "contact": round(state["contact"], 2),
             "thresh": state["thresh"] and round(state["thresh"], 2), "auto_saved": state["auto_saved"]}
    return jsonify(s)


@app.get("/api/episodes")
def api_episodes():
    return jsonify(episodes())


@app.post("/api/rebase")
def api_rebase():
    with lock:
        state["rebase"] = True
    return jsonify(ok=True)


@app.post("/api/reconnect")
def api_reconnect():
    with lock:
        state["reconnect"] = True
    return jsonify(ok=True)


@app.post("/api/label")
def api_label():
    with lock:
        state["label"] = request.json.get("label", "").strip()
    return jsonify(ok=True)


@app.post("/api/toggle")
def api_toggle():
    with lock:
        if not state["rec"]:
            state.update(rec=True, frames=[], ts=[], t0=time.time(), auto_rec=False)
            return jsonify(rec=True)
        data = stop_locked()
    meta = write_episode(*data)
    return jsonify(rec=False, saved=meta) if meta else jsonify(rec=False)


@app.post("/api/settings")
def api_settings():
    j = request.json or {}
    with lock:
        if j.get("mode") in ("manual", "auto"):
            if state["rec"] and j["mode"] != state["mode"]:
                return jsonify(ok=False, error="stop recording first"), 400
            state["mode"] = j["mode"]
        if "padding" in j:
            state["padding"] = float(min(5.0, max(0.0, float(j["padding"]))))
        if "sensitivity" in j:
            state["sensitivity"] = float(min(3.0, max(0.3, float(j["sensitivity"]))))
            update_thresh()
        if "min_press" in j:
            state["min_press"] = float(min(5.0, max(0.0, float(j["min_press"]))))
        return jsonify(ok=True, mode=state["mode"], padding=state["padding"], min_press=state["min_press"],
                       sensitivity=state["sensitivity"])


@app.post("/api/delete/<eid>")
def api_delete(eid):
    if not eid.replace("-", "").isdigit():
        return jsonify(ok=False), 400
    for ext in (".npz", ".json"):
        p = os.path.join(OUT, eid + ext)
        if os.path.exists(p):
            os.remove(p)
    _cache.pop(eid, None)
    return jsonify(ok=True)


# ---------- review ----------
_cache = {}

def load_ep(eid):
    if eid not in _cache:
        _cache.clear()
        d = np.load(os.path.join(OUT, eid + ".npz"))
        fr = d["frames"].astype(np.float32)
        base = fr[: min(5, len(fr))].mean(axis=0)
        diff = np.abs(fr - base).mean(axis=3)                    # [T,H,W]
        _cache[eid] = dict(frames=d["frames"], t=d["t"], diff=diff,
                           signal=diff.mean(axis=(1, 2)), raster=diff.mean(axis=2).T)  # raster [H,T]
    return _cache[eid]


def heat(a, gain, cmap):
    return cv2.applyColorMap(np.clip(a * gain, 0, 255).astype(np.uint8), cmap)


@app.route("/ep/<eid>/info")
def ep_info(eid):
    e = load_ep(eid)
    return jsonify(n=len(e["frames"]), t=e["t"].tolist(), signal=e["signal"].round(3).tolist())


@app.route("/ep/<eid>/frame/<int:i>")
def ep_frame(eid, i):
    e = load_ep(eid)
    i = max(0, min(i, len(e["frames"]) - 1))
    raw, d = e["frames"][i], e["diff"][i]
    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0); gy = cv2.Sobel(d, cv2.CV_32F, 0, 1)
    views = {"raw": raw, "diff": heat(d, 4, cv2.COLORMAP_INFERNO),
             "edges": heat(np.hypot(gx, gy), 1, cv2.COLORMAP_VIRIDIS),
             "overlay": cv2.addWeighted(raw, 0.5, heat(d, 4, cv2.COLORMAP_INFERNO), 0.7, 0)}
    v = request.args.get("view", "grid")
    if v == "force":
        import tactile_model as tm
        base = e["frames"][:5].astype(np.float32).mean(0)
        return Response(cv2.imencode(".jpg", tm.field_viz(raw, base))[1].tobytes(), mimetype="image/jpeg")
    if v == "grid":
        img = np.vstack([np.hstack([views["raw"], views["diff"]]), np.hstack([views["overlay"], views["edges"]])])
        for (x, y), name in zip([(0, 0), (320, 0), (0, 240), (320, 240)], ["raw", "diff", "overlay", "edges"]):
            cv2.putText(img, name, (x + 8, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    else:
        img = cv2.resize(views.get(v, raw), (640, 480))
    return Response(cv2.imencode(".jpg", img)[1].tobytes(), mimetype="image/jpeg")


@app.route("/ep/<eid>/raster")
def ep_raster(eid):
    r = load_ep(eid)["raster"]
    img = cv2.resize(heat(r, 8, cv2.COLORMAP_VIRIDIS), (640, 120), interpolation=cv2.INTER_NEAREST)
    return Response(cv2.imencode(".png", img)[1].tobytes(), mimetype="image/png")


@app.post("/api/relabel/<eid>")
def api_relabel(eid):
    p = os.path.join(OUT, eid + ".json")
    if not eid.replace("-", "").isdigit() or not os.path.exists(p):
        return jsonify(ok=False), 404
    with open(p) as fh:
        meta = json.load(fh)
    meta["label"] = request.json.get("label", "").strip()
    with open(p, "w") as fh:
        json.dump(meta, fh)
    return jsonify(ok=True)   # .json is the source of truth for labels


if __name__ == "__main__":
    threading.Thread(target=capture, daemon=True).start()
    if os.environ.get("CLASSIFY") == "1":            # live classifier off by default (CLASSIFY=1 to enable)
        threading.Thread(target=predictor, daemon=True).start()
    print("Open http://localhost:5050")
    app.run(port=5050, threaded=True)
