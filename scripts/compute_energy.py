import sys
import numpy as np
import pandas as pd

power_csv = sys.argv[1] if len(sys.argv) > 1 else "results/power_run1.csv"
log = pd.read_csv("results/runs_log.csv")

d = pd.read_csv(power_csv, encoding="latin1", low_memory=False)
d["P"] = pd.to_numeric(d["CPU Package Power [W]"], errors="coerce")
d["t"] = pd.to_datetime(d["Date"].astype(str) + " " + d["Time"].astype(str),
                        dayfirst=True, errors="coerce")
d = d.dropna(subset=["P", "t"]).sort_values("t").reset_index(drop=True)
t0 = d["t"].iloc[0]
ts = (d["t"] - t0).dt.total_seconds().to_numpy()
ps = d["P"].to_numpy()

def sec(x):
    return (pd.to_datetime(x) - t0).total_seconds()

def energy(a, b):
    """Intégrale de la puissance (en joules) entre a et b, par trapèzes."""
    grid = np.concatenate(([a], ts[(ts > a) & (ts < b)], [b]))
    p = np.interp(grid, ts, ps)
    return float(np.sum((p[1:] + p[:-1]) / 2 * np.diff(grid)))

print(f"Fichier de puissance : {len(d)} lignes, de {d['t'].iloc[0]} à {d['t'].iloc[-1]}\n")
rows = []
for _, r in log.iterrows():
    a, b = sec(r["start"]), sec(r["end"])
    if a < ts[0] or b > ts[-1]:
        print(f"rep {r['rep']}: HORS du fichier de puissance (vérifiez le fichier choisi)")
        continue
    dur = b - a
    # Repos : fenêtre juste avant l'encodage (en laissant 2 s de marge au début)
    i0, i1 = sec(r["idle_start"]) + 2.0, a - 0.5
    idle_mask = (ts >= i0) & (ts <= i1)
    p_idle = float(np.median(ps[idle_mask])) if idle_mask.sum() >= 5 else float("nan")
    e_tot = energy(a, b)
    n = int(((ts >= a) & (ts <= b)).sum())
    e_act = e_tot - p_idle * dur
    rows.append([r["label"], r["rep"], round(dur, 2), n, round(e_tot, 1),
                 round(e_tot / dur, 1), round(p_idle, 1), round(e_act, 1)])

out = pd.DataFrame(rows, columns=["label", "rep", "duree_s", "n_lectures",
                                  "energie_totale_J", "puissance_moy_W",
                                  "repos_W", "energie_active_J"])
print(out.to_string(index=False))
if len(out) > 1:
    print(f"\nÉnergie totale : moyenne {out['energie_totale_J'].mean():.1f} J, "
          f"écart-type {out['energie_totale_J'].std():.1f} J")
    print(f"Énergie active : moyenne {out['energie_active_J'].mean():.1f} J, "
          f"écart-type {out['energie_active_J'].std():.1f} J")
out.to_csv("results/energy_test.csv", index=False)
print("\nTableau enregistré dans results/energy_test.csv")