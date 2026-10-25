import os, csv, shutil, subprocess
from evaluate import evaluate

SEQS = ["MVI_20011", "MVI_40204", "MVI_40732", "MVI_63521"]
CRFS = [18, 23, 28, 33, 38, 43]
FPS = 25
N = 200
DURATION = N / FPS   # 8 secondes

os.makedirs("results", exist_ok=True)
rows = []

for seq in SEQS:
    # Référence : images originales (sans encodage x265)
    ap50, ap, ms = evaluate(seq, f"data/{seq}/images", "original")
    rows.append([seq, "original", "", "", ap50, ap, ms])

    os.makedirs(f"data/encoded/{seq}", exist_ok=True)
    for crf in CRFS:
        video = f"data/encoded/{seq}/crf{crf}.mp4"
        dec = f"data/decoded/{seq}/crf{crf}"
        os.makedirs(dec, exist_ok=True)

        # 1) Encoder en x265
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(FPS), "-i", f"data/{seq}/images/img%05d.jpg",
            "-frames:v", str(N),
            "-c:v", "libx265", "-crf", str(crf), "-preset", "medium",
            "-x265-params", "log-level=error",
            "-pix_fmt", "yuv420p", video], check=True)

        # 2) Décoder en images PNG
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", video, f"{dec}/img%05d.png"], check=True)

        # 3) Débit et score
        size = os.path.getsize(video)
        kbps = size * 8 / DURATION / 1000
        ap50, ap, ms = evaluate(seq, dec, f"crf{crf}")
        rows.append([seq, f"crf{crf}", crf, round(kbps, 1), ap50, ap, ms])

        # 4) Effacer les images décodées pour économiser le disque
        shutil.rmtree(dec, ignore_errors=True)

    with open("results/rd_dev.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq", "name", "crf", "bitrate_kbps", "ap50", "ap50_95", "ms_per_image"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], r[3], round(r[4], 4), round(r[5], 4), round(r[6], 1)])
    print(f"--- {seq} terminée, résultats enregistrés ---\n")

print("Terminé. Fichier : results/rd_dev.csv")