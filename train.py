"""Frame-level classifier on DIGIT diffs.  Run: .venv/bin/python train.py
- uses frames with contact (>= CONTACT_FRAC of the episode's peak), labelled by episode
- leave-whole-episodes-out CV (GroupKFold) so frames of a test episode are never in training
- augmentation (flip/rotate/shift) applied to training frames only
- compares: contact-score only | 32x32 pixels+PCA | ResNet-18 frozen embedding
- saves model.joblib for live use in collect.py"""
import glob, json, os
import joblib
import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import tactile_model as tm

import sys
CONTACT_FRAC, N_AUG, SEED = 0.4, 4, 0
# which labels to train on (others in episodes/ are ignored). Override: python train.py hard soft
LABELS = sys.argv[1:] or ["hard", "soft"]
ROOT = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(SEED)

# ---------- load frames ----------
D, S, I, Y, G = [], [], [], [], []    # normalised diffs, contact score, intensity stats, label, episode
FS, FG = [], []                        # force-field stats, force-field grid (flattened 12x16x3)
for j in sorted(glob.glob(os.path.join(ROOT, "episodes/*.json"))):
    meta = json.load(open(j))
    if meta["label"] not in LABELS:
        continue
    fr = np.load(j[:-5] + ".npz")["frames"]
    base = fr[:5].astype(np.float32).mean(0)
    diffs = [tm.diff_map(f, base) for f in fr]
    c = np.array([tm.contact_score(f, base) for f in fr])     # same detector as live
    for i in np.where(c >= CONTACT_FRAC * c.max())[0]:
        D.append(tm.norm(diffs[i])); S.append(c[i]); I.append(tm.intensity(fr[i], base))
        ff = tm.force_field(fr[i], base); FS.append(tm.field_stats(ff)); FG.append(ff.ravel())
        Y.append(meta["label"]); G.append(meta["id"])
Y, G, S, I = np.array(Y), np.array(G), np.array(S), np.array(I)
FS, FG = np.array(FS), np.array(FG)
if len(set(Y)) < 2 or len(set(G)) < 4:
    sys.exit(f"need >=2 labels and >=4 episodes among {LABELS}; found {dict(zip(*np.unique(Y, return_counts=True))) if len(Y) else 'none'}")
labels = sorted(set(Y))
print(f"{len(D)} contact frames from {len(set(G))} episodes:", {l: int((Y == l).sum()) for l in labels})
REST = []                              # contact level just after baseline, sensor at rest
for j in glob.glob(os.path.join(ROOT, "episodes/*.json")):
    if json.load(open(j))["label"] not in LABELS:
        continue
    fr = np.load(j[:-5] + ".npz")["frames"].astype(np.float32)
    REST.append(np.mean([tm.contact_score(f, fr[:5].mean(0)) for f in fr[5:8]]))
noise = float(np.median(REST))

# ---------- features ----------
def pix(ds):
    return np.stack([cv2.resize(d, (32, 32)).ravel() for d in ds])

print("embedding originals...")
E = tm.embed(D)
print(f"embedding {N_AUG}x augmented copies...")
AUG = [[tm.augment(d, rng) for d in D] for _ in range(N_AUG)]
EA = [tm.embed(a) for a in AUG]
PA = [pix(a) for a in AUG]
P = pix(D)
E1, EA1 = E, EA                                    # diff-only latent (512)
# Which features the saved/live model uses:  "resnet" = ResNet diff + intensity (516)
#   "field" = force-field stats only (11)   "both" = ResNet diff + intensity + force-field stats (527)
FEATURES = os.environ.get("FEATURES", "both")
EB = np.hstack([E1, I, FS]); EBA = [np.hstack([a, I, FS]) for a in EA1]      # field stats are aug-invariant
E = np.hstack([E1, I]); EA = [np.hstack([a, I]) for a in EA1]

def clf(pca=None):
    steps = [StandardScaler()] + ([PCA(pca, random_state=SEED)] if pca else []) + [LogisticRegression(max_iter=3000, C=0.5)]
    return make_pipeline(*steps)

methods = {
    "contact score only": (lambda: clf(), S[:, None], None),
    "intensity stats only (4)": (lambda: clf(), I, None),
    "pixels 32x32 + PCA": (lambda: clf(20), P, PA),
    "ResNet-18 diff only":   (lambda: clf(), E1, EA1),
    "ResNet-18 diff + intensity": (lambda: clf(), E, EA),
    "force field: stats (11)":   (lambda: clf(), FS, None),
    "force field: grid 12x16x3 + PCA": (lambda: clf(20), FG, None),
    "force field: stats + grid": (lambda: clf(30), np.hstack([FS, FG]), None),
    "ResNet + intensity + field stats": (lambda: clf(), EB, EBA),
}

def ep_vote(pred_frames, g):
    out = {}
    for e in set(g):
        v, n = np.unique(pred_frames[g == e], return_counts=True)
        out[e] = v[n.argmax()]
    return out

cv = GroupKFold(n_splits=min(6, len(set(G))))
print("\nleave-episodes-out CV (6 folds)            frame acc   episode acc (majority vote)")
for name, (mk, X, XA) in methods.items():
    pred = np.empty(len(Y), dtype=object)
    for tr, te in cv.split(X, Y, G):
        Xtr, ytr = X[tr], Y[tr]
        if XA is not None:
            Xtr = np.concatenate([Xtr] + [xa[tr] for xa in XA]); ytr = np.concatenate([ytr] * (1 + len(XA)))
        pred[te] = mk().fit(Xtr, ytr).predict(X[te])
    votes = ep_vote(pred, G)
    true = {g: y for g, y in zip(G, Y)}
    ea = np.mean([votes[e] == true[e] for e in votes])
    print(f"  {name:38s} {np.mean(pred == Y):6.1%}      {ea:6.1%}")
    if name == "ResNet + intensity + field stats":
        print("  confusion (rows=true, cols=pred)", labels)
        print("  " + str(confusion_matrix(Y, pred.astype(str), labels=labels)).replace("\n", "\n  "))
print(f"  chance ≈ {max((Y == l).mean() for l in labels):.1%}")

# ---------- which physical quantities matter? (field stats, standardised logistic weights) ----------
fs_m = clf().fit(FS, Y)
coef = fs_m[-1].coef_[0] if len(labels) == 2 else np.abs(fs_m[-1].coef_).mean(0)
print("\nforce-field stat weights" + (f" (+ => {labels[1]})" if len(labels) == 2 else " (|mean|)") + ":")
for n_, w_ in sorted(zip(tm.FIELD_STAT_NAMES, coef), key=lambda t: -abs(t[1])):
    print(f"  {n_:28s} {w_:+.2f}   " + "  ".join(f"{l}:{FS[Y == l, tm.FIELD_STAT_NAMES.index(n_)].mean():.2f}" for l in labels))

if FEATURES == "field":
    E, EA = FS, [FS] * N_AUG
elif FEATURES == "both":
    E, EA = EB, EBA
print(f"\nsaved/live model features: {FEATURES}")

# ---------- one-vs-one: a binary model for each pair of classes (ResNet-18 features) ----------
from itertools import combinations
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
PAIRS = list(combinations(labels, 2))
def ovo_model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced"))
print("\none-vs-one (ResNet-18, leave-episodes-out CV)         AUC   bal.acc   (frames of those 2 classes only)")
PP = {}                                   # P(first class of pair) for every frame, out-of-fold
for a_, b_ in PAIRS:
    PP[(a_, b_)] = np.zeros(len(Y))
    for tr, te in cv.split(E, Y, G):
        tr = tr[np.isin(Y[tr], [a_, b_])]
        Xtr = np.concatenate([E[tr]] + [ea[tr] for ea in EA]); ytr = np.tile((Y[tr] == a_).astype(int), 1 + N_AUG)
        PP[(a_, b_)][te] = ovo_model().fit(Xtr, ytr).predict_proba(E[te])[:, 1]
    m = np.isin(Y, [a_, b_]); yb = (Y[m] == a_).astype(int); p_ = PP[(a_, b_)][m]
    print(f"  {a_:8s} vs {b_:8s}                                {roc_auc_score(yb, p_):.3f}  {balanced_accuracy_score(yb, p_ > .5):6.1%}")

def ovo_vote(pp):                          # pp: {(a,b): P(a)} -> soft-vote scores per class
    sc = {l: 0.0 for l in labels}
    for (a_, b_), p_ in pp.items():
        sc[a_] += p_; sc[b_] += 1 - p_
    return sc
ovo_pred = np.array([max((sc := ovo_vote({k: v[i] for k, v in PP.items()})), key=sc.get) for i in range(len(Y))])
print(f"  combined (pairwise vote) frame acc: {np.mean(ovo_pred == Y):.1%}   episode acc: "
      f"{np.mean([v == dict(zip(G, Y))[e] for e, v in ep_vote(ovo_pred, G).items()]):.1%}")
print("  confusion (rows=true, cols=pred)", labels)
print("  " + str(confusion_matrix(Y, ovo_pred, labels=labels)).replace("\n", "\n  "))

OVO = {}
for a_, b_ in PAIRS:
    m = np.isin(Y, [a_, b_])
    OVO[(a_, b_)] = ovo_model().fit(np.concatenate([E[m]] + [ea[m] for ea in EA]), np.tile((Y[m] == a_).astype(int), 1 + N_AUG))

# ---------- final model on everything (+aug) ----------
final = clf().fit(np.concatenate([E] + EA), np.concatenate([Y] * (1 + N_AUG)))
thresh = 2.0 * noise                    # on contact_score (drift-compensated local p99)
joblib.dump({"clf": final, "ovo": OVO, "labels": labels, "contact_thresh": thresh, "features": FEATURES}, os.path.join(ROOT, "model.joblib"))
print(f"\nsaved model.joblib  (live contact threshold = {thresh:.2f}, noise ≈ {noise:.2f})")
