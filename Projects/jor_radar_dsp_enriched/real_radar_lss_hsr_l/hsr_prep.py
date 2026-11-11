import pickle, numpy as np
rows = pickle.load(open('routes.pkl','rb')); SEG = 20
def feats(D):
    D = np.maximum(D.astype(np.float64), 0); pk = D.argmax(1); n = D.shape[1]; idx = np.arange(n)
    Dn = D/(D.max(1, keepdims=True)+1e-9); p = Dn/(Dn.sum(1, keepdims=True)+1e-9)
    cen = (p*idx).sum(1); var = (p*(idx-cen[:,None])**2).sum(1); bw = np.sqrt(var)
    kur = (p*(idx-cen[:,None])**4).sum(1)/(var**2+1e-9); ent = -(p*np.log(p+1e-12)).sum(1)
    pm = np.log(D.max(1)+1e-9) - np.log(np.median(D,1)+1e-9)
    b5 = (Dn > 0.5).sum(1); b1 = (Dn > 0.1).sum(1)
    off = (idx[None,:] - pk[:,None]); far = np.abs(off) > 5
    sp = np.where(far, Dn, 0).max(1)
    asym = (p*np.sign(off)).sum(1)
    win = np.array([np.take(Dn[i], (pk[i]+np.arange(-32,32)) % n) for i in range(len(D))]).reshape(len(D),16,4).mean(2)
    return np.c_[bw, ent, pm, b5, b1, kur, sp, asym, np.log(win+1e-4)]
segs = []
for i, (rid, c, sp, T, D) in enumerate(rows):
    if len(T) < SEG: continue
    F = feats(D)
    for s in range(0, min(len(T)//SEG, 10)*SEG, SEG):
        segs.append((i, F[s:s+SEG].astype(np.float32), T[s:s+SEG, 4].astype(np.float64), c.startswith('UAV'), c))
pickle.dump(segs, open('segs.pkl','wb'))
y = np.array([s[3] for s in segs]); print('segs', len(segs), 'UAV', y.sum(), 'other', (~y).sum(), 'routes', len({s[0] for s in segs}))
import collections; print(collections.Counter(s[4] for s in segs))
