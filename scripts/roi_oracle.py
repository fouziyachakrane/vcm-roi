import os, csv, subprocess, shutil
import numpy as np
import cv2
from evaluate import evaluate, load_gt
import sys
SEQ = sys.argv[1] if len(sys.argv) > 1 else "MVI_20011"
N, FPS = 200, 25
SEG = 10          # images par segment : les ROI sont regroupées sur un segment
MARGIN = 16       # pixels ajoutés autour de chaque boîte
W, H = 960, 540
SRC = f"data/{SEQ}/images/img%05d.jpg"

UNIFORM_CRFS = [33, 38, 43, 48]
ROI_GRID = [(43, -0.1), (43, -0.2), (43, -0.3),
            (48, -0.1), (48, -0.2), (48, -0.3)]   # (crf, décalage de qualité)


def segment_rects(frames, gt):
    """Rectangles ROI d'un segment : union des boîtes (+ marge) sur ses images."""
    mask = np.zeros((H, W), np.uint8)
    for f in frames:
        for x1, y1, x2, y2 in gt.get(f, []):
            a, b = max(int(x1) - MARGIN, 0), max(int(y1) - MARGIN, 0)
            c, d = min(int(x2) + MARGIN, W), min(int(y2) + MARGIN, H)
            if c > a and d > b:
                mask[b:d, a:c] = 1
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    rects = [tuple(int(v) for v in stats[i][:4]) for i in range(1, n)]
    return rects, float(mask.mean())


def encode_segment(first, count, crf, rects, qoff, out):
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
           "-start_number", str(first), "-i", SRC, "-frames:v", str(count)]
    if rects:
        cmd += ["-vf", ",".join(
            f"addroi=x={x}:y={y}:w={w}:h={h}:qoffset={qoff}" for x, y, w, h in rects)]
    cmd += ["-c:v", "libx265", "-crf", str(crf), "-preset", "medium",
            "-x265-params", "log-level=error", "-pix_fmt", "yuv420p",
            "-f", "hevc", out]
    subprocess.run(cmd, check=True)


def run_config(name, crf, qoff, use_roi, gt):
    work = f"data/roi_runs/{SEQ}/{name}"
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work + "/dec")
    parts, covers = [], []
    for first in range(1, N + 1, SEG):
        count = min(SEG, N - first + 1)
        rects, cov = (segment_rects(range(first, first + count), gt)
                      if use_roi else ([], 0.0))
        out = f"{work}/seg{first:03d}.h265"
        encode_segment(first, count, crf, rects, qoff, out)
        parts.append(out)
        covers.append(cov)

    stream = f"{work}/all.h265"
    with open(stream, "wb") as f:
        for p in parts:
            with open(p, "rb") as g:
                f.write(g.read())
            os.remove(p)

    kbps = os.path.getsize(stream) * 8 / (N / FPS) / 1000
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", stream,
                    "-frames:v", str(N), f"{work}/dec/img%05d.png"], check=True)
    n_dec = len(os.listdir(f"{work}/dec"))
    if n_dec != N:
        print(f"ATTENTION {name}: {n_dec} images décodées au lieu de {N}")
        return None
    ap50, ap, ms = evaluate(SEQ, f"{work}/dec", name)
    shutil.rmtree(f"{work}/dec", ignore_errors=True)
    return dict(mode="roi" if use_roi else "uniforme", name=name, crf=crf,
                qoffset=qoff if use_roi else "", bitrate_kbps=round(kbps, 1),
                roi_pct=round(100 * float(np.mean(covers)), 1),
                ap50=round(ap50, 4), ap50_95=round(ap, 4))


gt, _ = load_gt(SEQ)
configs = [(f"uniforme_crf{c}", c, 0, False) for c in UNIFORM_CRFS]
configs += [(f"roi_crf{c}_q{abs(q)}", c, q, True) for c, q in ROI_GRID]

rows = []
os.makedirs("results", exist_ok=True)
for name, crf, qoff, use_roi in configs:
    r = run_config(name, crf, qoff, use_roi, gt)
    if r is None:
        continue
    rows.append(r)
    with open("results/roi_oracle.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(r.keys()))
        w.writeheader()
        w.writerows(rows)

print("\nmode     | crf | décalage | kbit/s | ROI % | AP50  | AP50-95")
for r in sorted(rows, key=lambda r: r["bitrate_kbps"], reverse=True):
    print(f"{r['mode']:<8} | {r['crf']:>3} | {str(r['qoffset']):>8} | "
          f"{r['bitrate_kbps']:>6} | {r['roi_pct']:>5} | {r['ap50']:.3f} | {r['ap50_95']:.3f}")