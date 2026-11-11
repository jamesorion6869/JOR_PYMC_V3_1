"""LSS-HSR-L runs v1 (VAR unset, 24 Doppler-shape features) and v2 (VAR=v2, 4 features). Exploratory. Detector code copied from parity_fold.py."""
import os, sys, pickle, importlib.util, numpy as np
_sp = importlib.util.spec_from_file_location("lib", os.path.join(os.path.dirname(os.path.abspath(__file__)), "jor_tuned_study_v1.1.py"))
Lb = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(Lb)
SEG = 20; Lb.TIME_STEPS = SEG
segs = [s[:4] for s in pickle.load(open("segs.pkl", "rb"))]
VAR = os.environ.get("VAR", "")
if VAR == "v2": segs = [(a, b[:, :4], c, d) for a, b, c, d in segs]  # PROTOCOL_hsr_v2_weakP: bw, ent, pm, b5
tracks = sorted({s[0] for s in segs})
from sklearn.ensemble import HistGradientBoostingClassifier as HG
from sklearn.model_selection import GroupKFold
SEED = int(os.environ["SEED"]); K = int(os.environ["FOLD"])
rng = np.random.default_rng(SEED); perm = rng.permutation(tracks); fold = {tr: k % 5 for k, tr in enumerate(perm)}
CANDS0 = Lb.make_weight_candidates()
MEM = [(r, a) for r in (0.70, 0.85, 0.95, 0.97, 0.99, 0.995, 0.999) for a in (0.30, 0.10, 0.05, 0.02, 0.01)]
MEM0 = {(r, a) for r in (0.70, 0.85, 0.95, 0.99) for a in (0.30, 0.10)}
CANDS = [dict(c, RET=r, ALPHA=a, orig=(r, a) in MEM0, name=c["name"] + f"_r{r}_a{a}") for r, a in MEM for c in CANDS0]
PNH0 = Lb.P_PRIOR_NH_INITIAL
THR = np.round(np.arange(0.200, 0.9995, 0.001), 3)
THR0 = np.isin(THR, np.round(np.arange(0.200, 0.900, 0.002), 3))
assert THR0.sum() == 350
DET0 = 2*7 + 4  # (3, 0.04, 0.06)
DET = [(s, r, m) for s in (1, 2, 3, 4) for (r, m) in [(None, 0.0)] + [(r, m) for r in (0.02, 0.04, 0.08) for m in (0.03, 0.06)]]
assert DET[DET0] == (3, 0.04, 0.06), DET[DET0]
ARMS = ("jor0", "jor1", "jor2")  # jor0 original grid; jor1 wider memory + threshold, original detector; jor2 adds detector options
def jor_vec(C, E, P, w):
    n, T = P.shape; pnh = np.full(n, PNH0); sm = None; out = np.empty((n, T))
    for t in range(T):
        sop = w["W_C"]*C[:, t] + w["W_E"]*E[:, t] + w["W_P"]*P[:, t]
        lh = np.clip(1 - sop + w["K"]*sop, 0, 1)
        num = sop*pnh; den = num + lh*(1-pnh)
        post = np.where(den > 0, num/np.where(den > 0, den, 1), 0.0)
        sm = post if sm is None else w["ALPHA"]*post + (1-w["ALPHA"])*sm
        out[:, t] = sm; pnh = w["RET"]*post + (1-w["RET"])*PNH0
    return out
def det_stats(S):
    """per detector: segment statistic s such that flag <=> s > threshold"""
    R = np.full(S.shape, -np.inf); R[:, 3:] = S[:, 3:] - S[:, :-3]
    out = []
    for s in (1, 2, 3, 4):
        W = np.full(S.shape, -np.inf); W[:, s-1:] = np.lib.stride_tricks.sliding_window_view(S, s, axis=1).min(axis=2)
        lev = W.max(axis=1)
        for (ss, r, m) in DET:
            if ss != s: continue
            out.append(lev if r is None else np.maximum(lev, np.where(R >= r, W, -np.inf).max(axis=1) + m))
    return out
Z = 1.96
def wilson_up(k, n):
    n = np.maximum(n, 1); p = k/n; d = 1 + Z*Z/n
    return (p + Z*Z/(2*n) + Z*np.sqrt(p*(1-p)/n + Z*Z/(4*n*n)))/d
assert abs(wilson_up(np.array([7]), np.array([100]))[0] - Lb.wilson_interval(7, 100)[1]) < 1e-9
NU = [round(x, 2) for x in np.arange(0.30, 0.951, 0.05)]
H = [0.01,0.02,0.03,0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60,0.80,1.00,1.20,1.50,2.00,2.50,3.00,4.0,5.0,6.0,8.0,10.0,12.0,15.0,20.0]
RISE = [(None, 0.0)] + [(r, f) for r in (0.05, 0.10, 0.20, 0.40) for f in (0.5, 1.0)]
def cusum(X, nu):
    S = np.zeros_like(X); s = np.zeros(X.shape[0])
    for t in range(X.shape[1]): s = np.maximum(0, s + X[:, t] - nu); S[:, t] = s
    return S
def cus_flags(S):
    out = {}; R = np.full(S.shape, -np.inf); R[:, 3:] = S[:, 3:] - S[:, :-3]
    for k in (1, 2, 3, 4):
        W = np.full(S.shape, -np.inf); W[:, k-1:] = np.lib.stride_tricks.sliding_window_view(S, k, axis=1).min(axis=2)
        for h in H:
            lev = W > h
            for r, f in RISE: out[(h, k, r, f)] = (lev if r is None else lev | ((R >= r) & (W > h - f*h))).any(axis=1)
    return out
BUD = (0.12, 0.08, 0.05)
tr_s = [s for s in segs if fold[s[0]] != K]; te_s = [s for s in segs if fold[s[0]] == K]
Xtr = np.vstack([s[1] for s in tr_s]); ytr = np.concatenate([[s[3]]*SEG for s in tr_s]); gtr = np.concatenate([[s[0]]*SEG for s in tr_s])
def fitP(X, yy): return HG(max_iter=200, learning_rate=0.08, class_weight="balanced", random_state=0).fit(X, yy)
oof = np.zeros(len(ytr))
for a_, b_ in GroupKFold(5).split(Xtr, ytr, gtr): oof[b_] = fitP(Xtr[a_], ytr[a_]).predict_proba(Xtr[b_])[:, 1]
m = fitP(Xtr, ytr)
snr_tr = np.concatenate([s[2] for s in tr_s]); q05, q95 = np.percentile(snr_tr, [5, 95])
def cep(ss, Pfr=None):
    P = (Pfr.reshape(len(ss), SEG) if Pfr is not None else np.array([m.predict_proba(s[1])[:, 1] for s in ss]))
    sn = np.array([s[2] for s in ss])
    return np.full(P.shape, 0.65), np.clip((sn-q05)/(q95-q05), 0, 1)*0.55 + 0.30, 0.30 + 0.65*P
Ctr, Etr, Ptr = cep(tr_s, oof); Cte, Ete, Pte = cep(te_s)
y = np.array([s[3] for s in tr_s]); yt = np.array([s[3] for s in te_s]); trk = [s[0] for s in te_s]
npos, nneg = y.sum(), (~y).sum()
best = {(a, b): None for a in ARMS for b in BUD}
for ci, w in enumerate(CANDS):
    for di, st in enumerate(det_stats(jor_vec(Ctr, Etr, Ptr, w))):
        arms = ["jor2"] + (["jor1"] if di == DET0 else []) + (["jor0"] if (di == DET0 and w["orig"]) else [])
        sp = np.sort(st[y]); sn_ = np.sort(st[~y])
        tp = npos - np.searchsorted(sp, THR, side="right"); fp = nneg - np.searchsorted(sn_, THR, side="right")
        up = wilson_up(fp, nneg)
        for a in arms:
            for b in BUD:
                ok = (up <= b) & (THR0 if a == "jor0" else True)
                if not ok.any(): continue
                key = np.where(ok, tp*100000 - fp, -10**12); j = int(np.argmax(key))
                kk = (int(tp[j]), -int(fp[j]))
                if best[(a, b)] is None or kk > best[(a, b)][0]: best[(a, b)] = (kk, ci, di, float(THR[j]))
out = {"seed": SEED, "fold": K, "trk": trk, "y": yt, "cus": {}, "picks": {"cus": {}}}
for a in ARMS:
    out[a] = {}; out["picks"][a] = {}
    for b in BUD:
        _, ci, di, th = best[(a, b)]; w = CANDS[ci]
        st = det_stats(jor_vec(Cte, Ete, Pte, w))[di]
        out[a][b] = st > th; out["picks"][a][b] = (w["name"], DET[di], th)
print("jor done", {a: out["picks"][a] for a in ARMS}, flush=True)
bc = {b: None for b in BUD}
for ci, w in enumerate(CANDS0):
    X = w["W_C"]*Ctr + w["W_E"]*Etr + w["W_P"]*Ptr
    for nu in NU:
        for c, f in cus_flags(cusum(X, nu)).items():
            fp = int((f & ~y).sum()); tpc = int((f & y).sum()); up = Lb.wilson_interval(fp, nneg)[1]
            for b in BUD:
                if up <= b:
                    kk = (tpc, -fp, -c[0])
                    if bc[b] is None or kk > bc[b][0]: bc[b] = (kk, ci, nu, c)
for b in BUD:
    _, ci, nu, c = bc[b]; w = CANDS0[ci]
    out["cus"][b] = cus_flags(cusum(w["W_C"]*Cte + w["W_E"]*Ete + w["W_P"]*Pte, nu))[c]
    out["picks"]["cus"][b] = (w["name"], nu) + tuple(c)

from sklearn.metrics import roc_auc_score
FE = 2*7  # (3, None, 0)
assert DET[FE] == (3, None, 0.0)
st_tr = det_stats(Ptr)[FE]; st_te = det_stats(Pte)[FE]
out["fe"] = {}; out["picks"]["fe"] = {}
sp = np.sort(st_tr[y]); sn_ = np.sort(st_tr[~y]); tp = npos - np.searchsorted(sp, THR, side="right"); fp = nneg - np.searchsorted(sn_, THR, side="right"); up = wilson_up(fp, nneg)
for b in BUD:
    ok = up <= b; key = np.where(ok, tp*100000 - fp, -10**12); j = int(np.argmax(key)); out["fe"][b] = st_te > THR[j]; out["picks"]["fe"][b] = float(THR[j])
yfr = np.repeat(yt, SEG); out["auc_frame_test"] = float(roc_auc_score(yfr, (Pte.ravel()-0.30)/0.65)); out["auc_frame_oof"] = float(roc_auc_score(ytr, oof))
pickle.dump(out, open(f"hsr_{VAR + '_' if VAR else ''}s{SEED}_f{K}.pkl", "wb"))
print("done", out["picks"], flush=True)
