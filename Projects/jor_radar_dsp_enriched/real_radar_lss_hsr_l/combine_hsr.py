import pickle, sys, json, numpy as np
PRE = __import__('os').environ.get('PRE','')
seeds = [int(s) for s in sys.argv[1:]]; BUD = (0.12, 0.08, 0.05); A = ('jor0','jor1','jor2','cus','fe')
res = {}
for sd in seeds:
    O = [pickle.load(open(f'hsr_{PRE}s{sd}_f{f}.pkl','rb')) for f in range(5)]
    y = np.concatenate([o['y'] for o in O]); trk = np.concatenate([o['trk'] for o in O])
    print(sd, 'frame AUC test', [round(o['auc_frame_test'],3) for o in O])
    for b in BUD:
        F = {a: np.concatenate([o[a][b] for o in O]) for a in A}
        r = {a: (round(F[a][y].mean(),3), round(F[a][~y].mean(),3)) for a in A}
        u = np.unique(trk); idx = {t: np.where(trk==t)[0] for t in u}; rng = np.random.default_rng(0); gaps = {}
        for x, z in (('jor2','cus'), ('jor2','fe'), ('cus','fe')):
            d = F[x].astype(float) - F[z].astype(float); bs = []
            for _ in range(2000):
                ii = np.concatenate([idx[t] for t in rng.choice(u, len(u))]); yy = y[ii]
                bs.append(d[ii][yy].mean())
            gaps[f'{x}-{z}'] = (round(d[y].mean(),3), round(np.percentile(bs,2.5),3), round(np.percentile(bs,97.5),3))
        res[f'{sd}_{b}'] = dict(recall_fpr=r, gaps=gaps); print(sd, b, r, gaps)
json.dump(res, open('hsr_results_' + PRE + '_'.join(map(str,seeds)) + '.json','w'), indent=1)

# plain-text summary next to the JSON
txt = 'hsr_results_' + PRE + '_'.join(map(str, seeds)) + '.txt'
with open(txt, 'w') as fh:
    fh.write(f"LSS-HSR-L JOR vs tuned CUSUM ({'v2 weak front end' if PRE else 'v1'}), seeds {seeds}. Exploratory.\n")
    fh.write("Recall / FPR on test folds; gaps are recall differences with 95% route-grouped bootstrap CIs.\n\n")
    for b in BUD:
        fh.write(f"FPR budget {b}\n")
        for a in A:
            rr = [res[f'{s}_{b}']['recall_fpr'][a] for s in seeds]
            fh.write(f"  {a:5s} mean recall {np.mean([x[0] for x in rr]):.3f}  mean FPR {np.mean([x[1] for x in rr]):.3f}   per seed: " + ", ".join(f"{x[0]:.3f}/{x[1]:.3f}" for x in rr) + "\n")
        for g in ('jor2-cus', 'jor2-fe', 'cus-fe'):
            fh.write(f"  {g:9s} " + ", ".join(f"s{s} {res[f'{s}_{b}']['gaps'][g][0]:+.3f} [{res[f'{s}_{b}']['gaps'][g][1]:+.3f}, {res[f'{s}_{b}']['gaps'][g][2]:+.3f}]" for s in seeds) + "\n")
        fh.write("\n")
print('wrote', txt)
