import subprocess, os
import cv2
import numpy as np

SRC = "data/MVI_20011/images/img%05d.jpg"
N = 200
CRF = 33
BOX = (300, 150, 360, 300)   # x, y, largeur, hauteur de la zone test
QOFF = -0.5                  # négatif = meilleure qualité dans la zone

os.makedirs("data/roi_test", exist_ok=True)


def encode(out, vf=None):
    cmd = ["ffmpeg", "-y", "-loglevel", "warning", "-framerate", "25",
           "-i", SRC, "-frames:v", str(N)]
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-c:v", "libx265", "-crf", str(CRF), "-preset", "medium",
            "-x265-params", "log-level=error", "-pix_fmt", "yuv420p", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.stderr.strip():
        print("Message de ffmpeg :", r.stderr.strip()[:500])
    return os.path.getsize(out)


def decode(video, folder):
    os.makedirs(folder, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video,
                    f"{folder}/img%05d.png"], check=True)


def psnr_in_out(folder):
    """PSNR (en dB) dans la zone et en dehors, vs les images d'origine.
    Plus c'est haut, plus l'image est fidèle."""
    x, y, w, h = BOX
    se_in = se_out = 0.0
    n_in = n_out = 0
    for i in range(1, N + 1, 5):
        a = cv2.imread(f"data/MVI_20011/images/img{i:05d}.jpg").astype(np.float64)
        b = cv2.imread(f"{folder}/img{i:05d}.png").astype(np.float64)
        d = (a - b) ** 2
        mask = np.zeros(d.shape[:2], bool)
        mask[y:y + h, x:x + w] = True
        se_in += d[mask].sum()
        n_in += d[mask].size
        se_out += d[~mask].sum()
        n_out += d[~mask].size
    f = lambda se, n: 10 * np.log10(255 ** 2 / (se / n))
    return f(se_in, n_in), f(se_out, n_out)


x, y, w, h = BOX
vf = f"addroi=x={x}:y={y}:w={w}:h={h}:qoffset={QOFF}"

s0 = encode("data/roi_test/base.mp4")
s1 = encode("data/roi_test/roi.mp4", vf)
decode("data/roi_test/base.mp4", "data/roi_test/dec_base")
decode("data/roi_test/roi.mp4", "data/roi_test/dec_roi")
b_in, b_out = psnr_in_out("data/roi_test/dec_base")
r_in, r_out = psnr_in_out("data/roi_test/dec_roi")

print()
print(f"Sans ROI : {s0 * 8 / 8 / 1000:.0f} kbit/s | PSNR dans la zone {b_in:.2f} dB | hors zone {b_out:.2f} dB")
print(f"Avec ROI : {s1 * 8 / 8 / 1000:.0f} kbit/s | PSNR dans la zone {r_in:.2f} dB | hors zone {r_out:.2f} dB")
print(f"Différence (ROI - sans ROI) : zone {r_in - b_in:+.2f} dB | hors zone {r_out - b_out:+.2f} dB | débit {100 * (s1 - s0) / s0:+.1f} %")