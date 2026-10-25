import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("results/rd_dev.csv")
seqs = list(df["seq"].unique())
enc = df[df["name"] != "original"]
orig = df[df["name"] == "original"].set_index("seq")

# ---- Figure 1 : score de détection en fonction du débit ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, col, titre in ((axes[0], "ap50", "AP50"), (axes[1], "ap50_95", "AP50-95")):
    for s in seqs:
        d = enc[enc["seq"] == s].sort_values("bitrate_kbps")
        line = ax.plot(d["bitrate_kbps"], d[col], marker="o", label=s)[0]
        ax.axhline(orig.loc[s, col], color=line.get_color(), linestyle=":", alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("Débit (kbit/s, échelle logarithmique)")
    ax.set_ylabel(titre)
    ax.set_title(f"{titre} en fonction du débit")
    ax.grid(True, alpha=0.3)
axes[0].legend(title="Séquence")
fig.suptitle("Compression x265 uniforme : détection YOLO11n (pointillés = images originales)")
fig.tight_layout()
fig.savefig("results/fig1_ap_vs_debit.png", dpi=150)

# ---- Figure 2 : débit selon le crf ----
fig, ax = plt.subplots(figsize=(7, 5))
for s in seqs:
    d = enc[enc["seq"] == s].sort_values("crf")
    ax.plot(d["crf"], d["bitrate_kbps"], marker="o", label=s)
ax.set_yscale("log")
ax.set_xlabel("CRF (plus grand = plus compressé)")
ax.set_ylabel("Débit (kbit/s, échelle logarithmique)")
ax.set_title("Débit obtenu selon le CRF")
ax.grid(True, alpha=0.3)
ax.legend(title="Séquence")
fig.tight_layout()
fig.savefig("results/fig2_debit_vs_crf.png", dpi=150)

# ---- Tableau de synthèse ----
print(enc.pivot(index="crf", columns="seq", values="bitrate_kbps").round(0))
print()
print(enc.pivot(index="crf", columns="seq", values="ap50").round(3))
print("\nFigures créées dans results/")