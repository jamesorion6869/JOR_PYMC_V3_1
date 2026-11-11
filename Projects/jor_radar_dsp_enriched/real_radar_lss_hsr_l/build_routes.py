"""LSS-HSR-L audit: routes, lengths, nuisance-only separability (grouped by air route). Exploratory."""
import json, numpy as np, scipy.io as sio, collections
from pathlib import Path
R = json.load(open('LSS-HSR-L/air_routes.json')); root = Path('LSS-HSR-L')
rows = []
for rid, r in R.items():
    T = []; D = []
    for f in r['tracks']:
        a = sio.loadmat(root/f, struct_as_record=False)['Trace_DPL_Data']
        D.append(np.asarray(a[0,0][0,0], np.float32)); T.append(np.asarray(a[0,1][0,0], np.float32))
    cls = Path(r['tracks'][0]).parts[1]; split = Path(r['tracks'][0]).parts[0]
    rows.append((rid, cls, split, np.vstack(T), np.vstack(D)))

import pickle; pickle.dump(rows, open('routes.pkl','wb'))
by = collections.defaultdict(list)
for rid, cls, sp, T, D in rows: by[cls].append((sp, len(T), np.median(T[:,1]), np.median(T[:,3]), np.median(T[:,4]), np.median(np.abs(T[:,0]))))
print('class routes frames(med) train/val/overflow range_km alt_m snr_db |vr|')
for c, v in by.items():
    v = np.array([x[1:] for x in v], float); sp = collections.Counter(x[0] for x in by[c])
    print(f"{c:10s} {len(v):4d} {np.median(v[:,0]):5.0f} {dict(sp)} {np.median(v[:,1]):5.2f} [{v[:,1].min():.2f},{v[:,1].max():.2f}] {np.median(v[:,2]):6.0f} {np.median(v[:,3]):5.1f} {np.median(v[:,4]):5.1f}")
