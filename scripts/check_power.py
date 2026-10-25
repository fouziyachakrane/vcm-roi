import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = pd.read_csv("results/power_test.csv", encoding="latin1", low_memory=False)
d["P"] = pd.to_numeric(d["CPU Package Power [W]"], errors="coerce")
d["t"] = pd.to_datetime(d["Date"].astype(str) + " " + d["Time"].astype(str),
                        dayfirst=True, errors="coerce")
d = d.dropna(subset=["P", "t"]).reset_index(drop=True)
d["s"] = (d["t"] - d["t"].iloc[0]).dt.total_seconds()

dt = d["s"].diff().dropna()
print("Lignes valides      :", len(d))
print("Durée               :", round(d["s"].iloc[-1], 1), "s")
print("Intervalle médian   :", round(dt.median(), 3), "s")
print("Intervalle max      :", round(dt.max(), 3), "s")
print("Puissance médiane   :", round(d["P"].median(), 2), "W")
print("Puissance 95e perc. :", round(d["P"].quantile(0.95), 2), "W")
print("Puissance max       :", round(d["P"].max(), 2), "W")

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(d["s"], d["P"], linewidth=1)
ax.set_xlabel("Temps (s)")
ax.set_ylabel("CPU Package Power (W)")
ax.set_title("Puissance du processeur pendant le test (HWiNFO, période 200 ms)")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("results/fig_power_test.png", dpi=150)
print("Figure : results/fig_power_test.png")