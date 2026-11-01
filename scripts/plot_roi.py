import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEQS = ["MVI_20011", "MVI_40204", "MVI_40732", "MVI_63521"]
fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=False)
for ax, s in zip(axes.flat, SEQS):
    d = pd.read_csv(f"results/roi_oracle_{s}.csv")
    u = d[d["mode"] == "uniforme"].sort_values("bitrate_kbps")
    r = d[d["mode"] == "roi"].sort_values("bitrate_kbps")
    ax.plot(u["bitrate_kbps"], u["ap50"], "o-", label="Compression uniforme")
    ax.plot(r["bitrate_kbps"], r["ap50"], "s", color="tab:red", label="ROI oracle")
    ax.set_xscale("log")
    ax.set_title(s)
    ax.set_xlabel("Débit (kbit/s, échelle log)")
    ax.set_ylabel("AP50")
    ax.grid(True, alpha=0.3)
axes[0, 0].legend()
fig.suptitle("ROI oracle vs compression uniforme (une exécution par point)")
fig.tight_layout()
fig.savefig("results/fig3_roi_oracle.png", dpi=150)
print("Figure : results/fig3_roi_oracle.png")